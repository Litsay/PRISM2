"""
PRISM2 框架主类

Proactive Recognition of Inconsistencies between 
Spatio-temporal Macro-Micro scales

一个Model-Agnostic的概念漂移主动感知与适应框架
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple
from collections import deque

from adapter.prism2_modules import (
    PRISM2Config,
    PRISM2State,
    SystemState,
    ManifoldRepresentationBuilder,
    GeometricFeatureExtractor,
    DivergenceDetector,
    GradedResponseController
)


class PRISM2(nn.Module):
    """
    PRISM2 框架主类
    
    使用示例:
    ```python
    # 创建Backbone
    backbone = Model(args)
    
    # 创建PRISM2框架
    prism2 = PRISM2(backbone, args)
    
    # 在线预测
    Y_hat = prism2(X_t)
    ```
    """
    
    def __init__(self, backbone: nn.Module, args):
        """
        参数:
            backbone: 时序预测模型
            args: 命令行参数
        """
        super().__init__()
        
        self.args = args
        self.config = PRISM2Config.from_args(args)
        
        # Backbone
        if args.freeze:
            backbone.requires_grad_(False)
        self.backbone = backbone
        
        # 设备
        try:
            self.device = next(backbone.parameters()).device
        except StopIteration:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化各模块
        self.manifold_builder = ManifoldRepresentationBuilder(self.config, self.device)
        self.feature_extractor = GeometricFeatureExtractor(self.config)
        self.divergence_detector = DivergenceDetector(self.config)
        self.response_controller = GradedResponseController(self.config, self.device)
        
        # 初始化状态
        self.state = PRISM2State()
        self.state.update_window_size(self.config.window_size)
        
        # 日志
        self.states_log: List[SystemState] = []
        self.divergence_log: List[float] = []
        self.H_diff_log: List[float] = []
        self.R_tail_log: List[float] = []
        
        # 控制标志
        self.flag_online_learning = False
        self.flag_update = False

        # 存储recent_batch用于在线学习
        self.register_buffer('recent_batch', None, persistent=False)
        self.criterion = nn.MSELoss()

        # 存储当前x_mark供后续推理使用
        self._current_x_mark = None

    def load_state_dict(self, state_dict, strict=False):
        """
        加载模型权重，支持加载原始backbone的权重

        如果state_dict的key没有'backbone.'前缀，则假设是原始backbone的权重，
        需要添加'backbone.'前缀
        """
        # 检查是否需要添加backbone前缀
        sample_key = list(state_dict.keys())[0] if state_dict else ''
        model_keys = list(self.state_dict().keys())
        model_key = model_keys[0] if model_keys else ''

        print(f"PRISM2.load_state_dict: sample_key='{sample_key[:50]}...', model_key='{model_key[:50]}...'")

        if model_key.startswith('backbone.') and not sample_key.startswith('backbone.'):
            # 需要添加backbone前缀
            print("PRISM2.load_state_dict: Adding 'backbone.' prefix to state_dict keys")
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = 'backbone.' + key
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        # 调用父类的load_state_dict，使用非严格模式忽略PRISM2特有的参数
        result = super().load_state_dict(state_dict, strict=strict)
        print(f"PRISM2.load_state_dict: Loaded {len(state_dict)} keys, missing_keys: {len(result.missing_keys)}, unexpected_keys: {len(result.unexpected_keys)}")
        return result

    def forward(self, x: Tensor, x_mark: Tensor = None, *args, **kwargs) -> Tensor:
        """
        前向传播
        
        在非在线学习模式下，直接调用backbone
        在在线学习模式下，执行PRISM2的状态感知预测
        """
        if not self.flag_online_learning:
            # 标准前向传播
            if x_mark is not None:
                return self.backbone(x, x_mark, *args, **kwargs)
            else:
                return self.backbone(x, *args, **kwargs)
        else:
            # 在线学习模式：执行状态感知预测
            return self._online_forward(x, x_mark, *args, **kwargs)
    
    def _online_forward(self, x: Tensor, x_mark: Tensor = None, *args, **kwargs) -> Tensor:
        """
        在线学习模式的前向传播
        """
        batch_size = x.size(0)
        outputs = []
        
        for i in range(batch_size):
            X_t = x[i]  # (seq_len, num_variates)
            X_mark_t = x_mark[i] if x_mark is not None else None
            
            # 单步在线预测
            Y_hat = self._online_step(X_t, X_mark_t)
            
            # 检查并处理NaN
            if torch.isnan(Y_hat).any():
                Y_hat = torch.zeros_like(Y_hat)
            
            outputs.append(Y_hat)
        
        return torch.stack(outputs, dim=0)
    
    def _online_step(self, X_t: Tensor, X_mark_t: Tensor = None) -> Tensor:
        """
        在线预测的单步执行
        """
        # 获取backbone的实际设备
        try:
            device = next(self.backbone.parameters()).device
        except StopIteration:
            device = self.device
        
        if X_t.device != device:
            X_t = X_t.to(device)
        if X_mark_t is not None and X_mark_t.device != device:
            X_mark_t = X_mark_t.to(device)
        
        # 保存当前x_mark供后续推理使用
        self._current_x_mark = X_mark_t
        
        # 步骤1: 提取流形嵌入（使用Backbone的语义表示）
        z_t = self.manifold_builder.extract_embedding(self.backbone, X_t, X_mark_t)
        
        # 步骤2: 更新滑动窗口
        self.state.embedding_buffer.append(z_t.detach())
        self.state.input_buffer.append(X_t.detach())
        self.state.timestamp_buffer.append(self.state.total_steps)
        self.state.total_steps += 1
        
        # 步骤3: 预热阶段检查
        if len(self.state.embedding_buffer) < self.config.warmup_steps:
            return self._simple_inference(X_t, X_mark_t)
        
        # 步骤4: 构建几何结构
        embeddings = torch.stack(list(self.state.embedding_buffer))
        K, sigma = self.manifold_builder.compute_similarity_matrix(embeddings)
        knn_indices, knn_distances = self.manifold_builder.compute_knn(
            embeddings, self.config.lid_k
        )
        
        # 步骤5: 提取宏-微尺度特征
        H_diff = self.feature_extractor.compute_diffusion_spectral_entropy(K)
        R_tail = self.feature_extractor.compute_window_level_lirti(knn_distances)
        
        self.H_diff_log.append(H_diff)
        self.R_tail_log.append(R_tail)
        
        # 步骤6: 计算分歧与状态判定
        epsilon = self.divergence_detector.compute_divergence(H_diff, R_tail)
        state = self.divergence_detector.determine_state(epsilon, H_diff)
        
        self.divergence_log.append(epsilon)
        self.states_log.append(state)
        self.state.divergence_history.append(epsilon)
        self.state.current_state = state
        
        # 更新状态计数
        if state == SystemState.STABLE:
            self.state.stable_count += 1
        elif state == SystemState.PRECURSOR:
            self.state.precursor_count += 1
        else:
            self.state.drift_count += 1
        
        # 步骤7: 分级响应
        Y_hat, updated = self.response_controller.respond(
            state=state,
            backbone=self.backbone,
            X_t=X_t,
            embeddings=embeddings,
            z_t=z_t,
            available_samples=list(self.state.available_samples),
            divergence_history=self.state.divergence_history,
            criterion=self.criterion,
            x_mark=self._current_x_mark
        )
        
        return Y_hat
    
    def _simple_inference(self, X_t: Tensor, X_mark_t: Tensor = None) -> Tensor:
        """预热阶段的简单推理"""
        self.backbone.eval()
        with torch.no_grad():
            if X_t.dim() == 2:
                X_t = X_t.unsqueeze(0)
            if X_mark_t is not None and X_mark_t.dim() == 2:
                X_mark_t = X_mark_t.unsqueeze(0)
            # 确保数据在正确的设备上 - 使用backbone的实际设备
            try:
                device = next(self.backbone.parameters()).device
            except StopIteration:
                device = self.device
            if X_t.device != device:
                X_t = X_t.to(device)
            if X_mark_t is not None and X_mark_t.device != device:
                X_mark_t = X_mark_t.to(device)
            # 调用backbone，支持需要x_mark的模型（如iTransformer）
            if X_mark_t is not None:
                output = self.backbone(X_t, X_mark_t)
            else:
                output = self.backbone(X_t)
            if isinstance(output, tuple):
                output = output[0]
        result = output.squeeze(0) if output.size(0) == 1 else output
        # 检查NaN
        if torch.isnan(result).any():
            # 如果有NaN，返回零
            result = torch.zeros_like(result)
        return result
    
    def add_labeled_sample(self, X: Tensor, Y: Tensor, x_mark: Tensor = None, epsilon: float = None):
        """
        添加带标签的样本到可用训练样本池
        
        参数:
            X: 输入, shape = (seq_len, num_variates)
            Y: 标签, shape = (pred_len, num_variates)
            x_mark: 时间特征, shape = (seq_len, mark_dim) 或 None
            epsilon: 该样本对应时刻的分歧值（用于流形感知加权）
        """
        # 确保数据在正确的设备上
        if X.device != self.device:
            X = X.to(self.device)
        if Y.device != self.device:
            Y = Y.to(self.device)
        if x_mark is not None and x_mark.device != self.device:
            x_mark = x_mark.to(self.device)
        
        # 提取嵌入 - backbone需要在eval模式
        self.backbone.eval()
        z = self.manifold_builder.extract_embedding(self.backbone, X, x_mark)
        
        # 添加到样本池: (X, Y, timestamp, embedding, x_mark, epsilon)
        t = self.state.total_steps
        # 如果没有提供epsilon，使用当前最新的分歧值
        if epsilon is None and len(self.state.divergence_history) > 0:
            epsilon = self.state.divergence_history[-1]
        elif epsilon is None:
            epsilon = 0.0
        
        x_mark_detached = x_mark.detach() if x_mark is not None else None
        self.state.available_samples.append((
            X.detach(), Y.detach(), t, z.detach(), x_mark_detached, epsilon
        ))
    
    def process_delayed_label(self, Y_delayed: Tensor, delay: int):
        """
        处理延迟到达的标签
        
        参数:
            Y_delayed: 延迟到达的真实标签
            delay: 延迟的时间步数（通常为pred_len）
        """
        if Y_delayed is None:
            return
        
        Y_delayed = Y_delayed.to(self.device)
        
        # 延迟标签对应的是 delay 步之前的输入
        if len(self.state.input_buffer) > delay:
            idx = -(delay + 1)
            X_past = self.state.input_buffer[idx]
            z_past = self.state.embedding_buffer[idx]
            t_past = self.state.timestamp_buffer[idx]
            
            self.state.available_samples.append((X_past, Y_delayed.detach(), t_past, z_past))
    
    def update_recent_batch(self, batch_x: Tensor, batch_y: Tensor):
        """
        更新recent_batch缓存
        """
        self.recent_batch = torch.cat([batch_x, batch_y], dim=1)
    
    def reset_state(self):
        """
        重置运行时状态
        """
        self.state = PRISM2State()
        self.state.update_window_size(self.config.window_size)
        self.divergence_detector.reset()

        # 重置响应控制器的更新计数器
        if hasattr(self.response_controller, 'reset_counters'):
            self.response_controller.reset_counters()

        self.states_log = []
        self.divergence_log = []
        self.H_diff_log = []
        self.R_tail_log = []
    
    def get_statistics(self) -> dict:
        """获取运行统计信息"""
        total = max(self.state.total_steps, 1)
        return {
            "total_steps": self.state.total_steps,
            "stable_count": self.state.stable_count,
            "precursor_count": self.state.precursor_count,
            "drift_count": self.state.drift_count,
            "stable_ratio": self.state.stable_count / total,
            "precursor_ratio": self.state.precursor_count / total,
            "drift_ratio": self.state.drift_count / total,
            "avg_divergence": sum(self.divergence_log) / max(len(self.divergence_log), 1),
            "current_state": self.state.current_state.value if self.state.current_state else "WARMUP"
        }
    
    def get_current_state(self) -> str:
        """获取当前系统状态"""
        if len(self.state.embedding_buffer) < self.config.warmup_steps:
            return "WARMUP"
        return self.state.current_state.value


class PRISM2Wrapper(nn.Module):
    """
    PRISM2包装器 - 用于与现有实验框架集成
    
    兼容Exp_Online的接口
    """
    
    def __init__(self, backbone: nn.Module, args):
        super().__init__()
        self.args = args
        self.prism2 = PRISM2(backbone, args)
        self.backbone = self.prism2.backbone
        
    def forward(self, x: Tensor, x_mark: Tensor = None, *args, **kwargs):
        return self.prism2(x, x_mark, *args, **kwargs)
    
    def set_online_mode(self, online: bool = True):
        """设置在线学习模式"""
        self.prism2.flag_online_learning = online
        
    def add_sample(self, X: Tensor, Y: Tensor):
        """添加带标签样本"""
        self.prism2.add_labeled_sample(X, Y)
        
    def process_label(self, Y: Tensor, delay: int):
        """处理延迟标签"""
        self.prism2.process_delayed_label(Y, delay)
        
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.prism2.get_statistics()
    
    def reset(self):
        """重置状态"""
        self.prism2.reset_state()
