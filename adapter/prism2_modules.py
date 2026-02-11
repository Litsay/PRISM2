"""
PRISM2 核心模块实现

包含四大核心模块：
1. ManifoldRepresentationBuilder - 流形表征构建
2. GeometricFeatureExtractor - 宏-微尺度几何特征提取
3. DivergenceDetector - 分歧计算与状态判定
4. GradedResponseController - 流形敏感型分级响应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from collections import deque


class SystemState(Enum):
    """系统状态枚举"""
    STABLE = "STABLE"           # 稳态：无需更新
    PRECURSOR = "PRECURSOR"     # 前驱态：快速微调
    DRIFT = "DRIFT"             # 漂移态：多步适应


@dataclass
class PRISM2Config:
    """PRISM2 框架超参数配置（优化版本）"""

    # 流形表征参数
    window_size: int = 256          # 滑动窗口大小 W
    k_neighbors: int = 15           # k近邻数量

    # 微尺度参数 (局部内秩尾指数)
    lid_k: int = 20                 # LID估计的k值

    # 分歧计算参数（更保守的默认值）
    gamma: float = 0.3              # 指示增强系数（降低以减少误报）
    tau_R: float = 2.5              # 微尺度异常阈值 (Z-score)
    tau_H: float = 2.0              # 宏尺度稳定阈值 (Z-score)
    ema_alpha: float = 0.05         # 指数移动平均系数（更慢的适应）

    # 状态判定阈值（更保守）
    theta_epsilon: float = 2.0      # 分歧阈值 (触发PRECURSOR)
    theta_H: float = 2.5            # 宏尺度变化阈值 (触发DRIFT)

    # 分级响应参数（保守学习率）
    precursor_lr: float = 0.0005    # PRECURSOR阶段学习率
    precursor_neighbors: int = 32   # PRECURSOR检索样本数 m
    drift_lr: float = 0.0001        # DRIFT阶段学习率
    drift_neighbors: int = 64       # DRIFT检索样本数 M (M > m)
    drift_steps: int = 2            # DRIFT更新步数（减少以防过拟合）

    # 流形感知权重
    weight_lambda: float = 0.5      # 权重系数 λ（降低以减少过度强调）

    # 其他
    warmup_steps: int = 100         # 预热步数（增加以获得更稳定的统计）

    # 安全机制参数
    max_consecutive_updates: int = 5    # 连续更新次数上限
    min_stable_ratio: float = 0.5       # 最小稳定比例（低于此值降低更新频率）

    # STABLE状态更新策略
    stable_update_interval: int = 5     # STABLE状态每N步更新一次（0表示不更新）
    stable_lr_scale: float = 0.5        # STABLE状态学习率缩放因子

    @classmethod
    def from_args(cls, args):
        """从命令行参数创建配置，支持模型和数据集特定配置"""
        # 尝试导入settings获取优化配置
        try:
            from settings import get_prism2_config
            dataset = getattr(args, 'dataset', None) or getattr(args, 'data', None)
            model = getattr(args, 'model', None)
            base_config = get_prism2_config(dataset, model)
        except ImportError:
            base_config = {}

        def get_param(name, default):
            """获取参数：命令行参数优先（如果不是None），否则用base_config，最后用default"""
            arg_val = getattr(args, f'prism2_{name}', None)
            if arg_val is not None:
                return arg_val
            return base_config.get(name, default)

        # 命令行参数优先（如果不是None），否则使用settings配置，最后使用默认值
        return cls(
            window_size=get_param('window_size', 256),
            k_neighbors=get_param('k_neighbors', 15),
            lid_k=get_param('lid_k', 20),
            gamma=get_param('gamma', 0.3),
            tau_R=get_param('tau_R', 2.5),
            tau_H=get_param('tau_H', 2.0),
            ema_alpha=get_param('ema_alpha', 0.05),
            theta_epsilon=get_param('theta_epsilon', 2.0),
            theta_H=get_param('theta_H', 2.5),
            precursor_lr=get_param('precursor_lr', 0.0005),
            precursor_neighbors=get_param('precursor_neighbors', 32),
            drift_lr=get_param('drift_lr', 0.0001),
            drift_neighbors=get_param('drift_neighbors', 64),
            drift_steps=get_param('drift_steps', 2),
            weight_lambda=get_param('weight_lambda', 0.5),
            warmup_steps=get_param('warmup_steps', 100),
            stable_update_interval=get_param('stable_update_interval', 5),
            stable_lr_scale=get_param('stable_lr_scale', 0.5),
        )


@dataclass
class PRISM2State:
    """PRISM2 运行时状态"""
    
    # 滑动窗口缓冲区
    embedding_buffer: deque = field(default_factory=lambda: deque(maxlen=256))
    input_buffer: deque = field(default_factory=lambda: deque(maxlen=256))
    label_buffer: deque = field(default_factory=lambda: deque(maxlen=256))
    timestamp_buffer: deque = field(default_factory=lambda: deque(maxlen=256))
    
    # 可用训练样本（带标签）: (X, Y, timestamp, embedding)
    available_samples: deque = field(default_factory=lambda: deque(maxlen=512))
    
    # 分歧历史
    divergence_history: List[float] = field(default_factory=list)
    
    # 当前系统状态
    current_state: SystemState = SystemState.STABLE
    
    # 计数器
    total_steps: int = 0
    stable_count: int = 0
    precursor_count: int = 0
    drift_count: int = 0
    
    def update_window_size(self, new_size: int):
        """动态调整窗口大小"""
        self.embedding_buffer = deque(self.embedding_buffer, maxlen=new_size)
        self.input_buffer = deque(self.input_buffer, maxlen=new_size)
        self.label_buffer = deque(self.label_buffer, maxlen=new_size)
        self.timestamp_buffer = deque(self.timestamp_buffer, maxlen=new_size)


class ManifoldRepresentationBuilder:
    """
    流形表征构建器
    
    负责：
    1. 从Backbone提取特征嵌入（使用模型的语义表示）
    2. 管理滑动窗口
    3. 构建几何结构（相似性矩阵、k近邻图）
    """
    
    def __init__(self, config: PRISM2Config, device: str = 'cuda'):
        self.config = config
        self.device = device
        self._embedding_dim = None  # 缓存embedding维度
        
    def extract_embedding(self, backbone: nn.Module, X: Tensor, x_mark: Tensor = None, target_dim: int = 64) -> Tensor:
        """
        从Backbone提取流形嵌入（使用模型的语义表示）
        
        参数:
            backbone: 时序预测模型
            X: 输入序列, shape = (batch_size, seq_len, num_variates) 或 (seq_len, num_variates)
            x_mark: 时间特征, shape与X对应
            target_dim: 目标embedding维度
               
        输出:
            嵌入向量, shape = (batch_size, embedding_dim) 或 (embedding_dim,)
        """
        single_sample = X.dim() == 2
        if single_sample:
            X = X.unsqueeze(0)
            if x_mark is not None:
                x_mark = x_mark.unsqueeze(0)
        
        # 确保数据在正确的设备上
        try:
            device = next(backbone.parameters()).device
        except StopIteration:
            device = self.device
        X = X.to(device)
        if x_mark is not None:
            x_mark = x_mark.to(device)
        
        batch_size = X.size(0)
        
        # 使用Backbone的语义表示
        with torch.no_grad():
            backbone.eval()
            embeddings = self._extract_from_backbone(backbone, X, x_mark)
            
            # 如果embedding维度过大，进行降维
            if embeddings.size(-1) > target_dim * 4:
                # 使用均匀采样降维
                step = embeddings.size(-1) // target_dim
                indices = torch.arange(0, target_dim * step, step, device=device)[:target_dim]
                embeddings = embeddings[..., indices]
            
            # L2归一化，提高距离可比性
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        if single_sample:
            embeddings = embeddings.squeeze(0)
            
        return embeddings
    
    def _extract_from_backbone(self, backbone: nn.Module, X: Tensor, x_mark: Tensor = None) -> Tensor:
        """
        从Backbone提取语义特征
        
        支持多种模型类型：
        1. 支持return_emb参数的模型（iTransformer, PatchTST）
        2. 有encoder属性的模型（TCN）
        3. 回退方案：使用hook提取中间层输出
        """
        batch_size = X.size(0)
        device = X.device
        
        # 方案1：尝试使用return_emb参数
        try:
            if x_mark is not None:
                output = backbone(X, x_mark, return_emb=True)
            else:
                output = backbone(X, return_emb=True)
            
            if isinstance(output, tuple) and len(output) >= 2:
                # 第二个元素通常是embedding
                emb = output[1]
                if emb.dim() == 3:
                    # (batch, seq, dim) -> (batch, dim) 通过平均池化
                    emb = emb.mean(dim=1)
                elif emb.dim() > 3:
                    emb = emb.view(batch_size, -1)
                return emb
        except (TypeError, RuntimeError):
            pass
        
        # 方案2：检查是否有encoder属性（如TCN）
        if hasattr(backbone, 'encoder'):
            try:
                encoder = backbone.encoder
                # TCN的encoder需要concat x_mark
                if hasattr(backbone, 'regressor'):  # TCN结构
                    if x_mark is None:
                        x_mark = torch.zeros(*X.shape[:2], 7, device=device)
                    x_concat = torch.cat([X, x_mark], dim=-1)
                    emb = encoder(x_concat)
                else:
                    emb = encoder(X)
                
                if emb.dim() == 3:
                    emb = emb.mean(dim=1)
                elif emb.dim() > 3:
                    emb = emb.view(batch_size, -1)
                return emb
            except (TypeError, RuntimeError):
                pass
        
        # 方案3：使用hook提取中间层输出
        return self._extract_from_forward_hook(backbone, X, x_mark)
    
    def _extract_from_forward_hook(self, backbone: nn.Module, X: Tensor, x_mark: Tensor = None) -> Tensor:
        """回退方案：从前向传播中使用hook提取特征"""
        batch_size = X.size(0)
        device = X.device
        features = []
        
        def hook(module, input, output):
            if isinstance(output, Tensor) and output.dim() >= 2:
                features.append(output.detach())
        
        # 注册hook到所有LayerNorm和Linear层
        handles = []
        for name, module in backbone.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.Linear)):
                handles.append(module.register_forward_hook(hook))
        
        # 前向传播
        try:
            if x_mark is not None:
                _ = backbone(X, x_mark)
            else:
                _ = backbone(X)
        except TypeError:
            _ = backbone(X)
        
        # 移除hooks
        for h in handles:
            h.remove()
        
        if features:
            # 选择合适维度的特征（倒数第二个或最后一个LayerNorm输出）
            for feat in reversed(features):
                if feat.dim() == 3 and feat.size(0) == batch_size:
                    # (batch, seq, dim) -> (batch, dim)
                    return feat.mean(dim=1)
                elif feat.dim() == 2 and feat.size(0) == batch_size:
                    return feat
            
            # 使用最后一个有效特征
            feat = features[-1]
            if feat.dim() > 2:
                feat = feat.view(batch_size, -1)
            return feat
        
        # 最终回退：使用输入的flatten（但这不应该发生）
        print("Warning: Failed to extract semantic embedding, falling back to input flatten")
        X_flat = X.view(batch_size, -1)
        if X_flat.size(1) > 64:
            step = X_flat.size(1) // 64
            indices = torch.arange(0, 64 * step, step, device=device)[:64]
            return X_flat[:, indices]
        return X_flat
    
    def compute_similarity_matrix(self, embeddings: Tensor, sigma: float = None) -> Tuple[Tensor, float]:
        """
        构建高斯核相似性矩阵
        
        公式: K_{i,j} = exp(-||z_i - z_j||^2 / (2σ^2))
        """
        # 确保embeddings是2D
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
        
        # 计算成对欧氏距离
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # 中值启发式选择sigma
        if sigma is None:
            nonzero_dists = dist_matrix[dist_matrix > 1e-10]
            if len(nonzero_dists) > 0:
                sigma = torch.median(nonzero_dists).item()
            else:
                sigma = 1.0
        
        # 计算高斯核
        K = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
        
        return K, sigma
    
    def compute_knn(self, embeddings: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        计算每个点的k近邻
        
        输出:
            knn_indices: 近邻索引, shape = (W, k)
            knn_distances: 近邻距离, shape = (W, k)
        """
        # 确保embeddings是2D
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
        
        W = embeddings.shape[0]
        k = min(k, W - 1)  # 确保k不超过可用邻居数
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # 排序（第一个是自身，距离为0）
        distances, indices = torch.sort(dist_matrix, dim=1)
        
        # 排除自身
        knn_distances = distances[:, 1:k+1]
        knn_indices = indices[:, 1:k+1]
        
        return knn_indices, knn_distances


class GeometricFeatureExtractor:
    """
    几何特征提取器
    
    负责计算：
    1. 宏尺度特征：扩散谱熵 (DSE)
    2. 微尺度特征：局部内秩尾指数 (LIRTI)
    """
    
    def __init__(self, config: PRISM2Config):
        self.config = config
    
    def compute_diffusion_spectral_entropy(self, K: Tensor) -> float:
        """
        计算扩散谱熵 H_diff
        
        物理意义:
        - 量化流形全局拓扑结构的复杂程度
        - 对局部扰动具有鲁棒性
        """
        W = K.shape[0]
        
        # 步骤1: 构建归一化扩散算子 P = D^{-1} K
        degree = K.sum(dim=1)
        degree = torch.clamp(degree, min=1e-10)
        D_inv = torch.diag(1.0 / degree)
        P = D_inv @ K
        
        # 步骤2: 计算特征值
        try:
            eigenvalues = torch.linalg.eigvals(P).real
        except:
            return 0.0
        
        # 检查NaN
        if torch.isnan(eigenvalues).any():
            return 0.0
        
        eigenvalues = torch.sort(eigenvalues, descending=True)[0]
        
        # 步骤3: 排除主特征值 λ_1 ≈ 1，对剩余特征值归一化
        eigenvalues_sub = eigenvalues[1:]
        eigenvalues_abs = torch.abs(eigenvalues_sub)
        
        total = eigenvalues_abs.sum()
        if total < 1e-10 or torch.isnan(total):
            return 0.0
        
        eigenvalues_norm = eigenvalues_abs / total
        
        # 步骤4: 计算熵
        eigenvalues_norm = torch.clamp(eigenvalues_norm, min=1e-10)
        H_diff = -torch.sum(eigenvalues_norm * torch.log(eigenvalues_norm))
        
        result = H_diff.item()
        return result if not (result != result) else 0.0  # NaN check
    
    def compute_local_intrinsic_rank_tail_index(self, knn_distances: Tensor) -> Tensor:
        """
        计算每个数据点的局部内秩尾指数 R̂_tail^(i)
        
        公式:
        R̂_tail^(i) = [1/(k-1) * Σ_{j=1}^{k-1} log(d_{i,k} / d_{i,j})]^{-1}
        """
        # 确保是2D张量
        if knn_distances.dim() > 2:
            knn_distances = knn_distances.view(-1, knn_distances.size(-1))
        W, k = knn_distances.shape
        
        if k < 2:
            return torch.ones(W, device=knn_distances.device)
        
        # d_{i,k}: 到第k近邻的距离（最远）
        d_k = knn_distances[:, -1:]  # shape = (W, 1)
        
        # d_{i,j}: 到前k-1个近邻的距离
        d_j = knn_distances[:, :-1]  # shape = (W, k-1)
        
        # 避免除零和log(0)
        d_j = torch.clamp(d_j, min=1e-10)
        d_k = torch.clamp(d_k, min=1e-10)
        
        # 计算对数比值的均值
        log_ratios = torch.log(d_k / d_j)
        mean_log_ratio = log_ratios.mean(dim=1)
        
        # 计算局部内秩尾指数（倒数）
        mean_log_ratio = torch.clamp(mean_log_ratio, min=1e-10)
        R_tail = 1.0 / mean_log_ratio
        
        return R_tail
    
    def compute_window_level_lirti(self, knn_distances: Tensor) -> float:
        """
        计算窗口级局部内秩尾指数
        
        公式: R̂_tail^(t) = (1/|W_t|) * Σ_{z_i ∈ W_t} R̂_tail^(i)
        """
        R_tail_per_point = self.compute_local_intrinsic_rank_tail_index(knn_distances)
        return R_tail_per_point.mean().item()


class DivergenceDetector:
    """
    分歧检测器
    
    负责：
    1. 在线更新统计量
    2. 计算宏-微尺度分歧
    3. 判定系统状态
    """
    
    def __init__(self, config: PRISM2Config):
        self.config = config
        self.alpha = config.ema_alpha
        
        # 在线统计量
        self.mu_H = 0.0
        self.var_H = 1.0
        self.mu_R = 0.0
        self.var_R = 1.0
        self.initialized = False
        
        # 缓存最近一次计算的z-score和delta_H（用于状态判定）
        self._last_z_H = 0.0
        self._last_z_R = 0.0
        self._last_delta_H = 0.0
        
    def reset(self):
        """重置统计量"""
        self.mu_H = 0.0
        self.var_H = 1.0
        self.mu_R = 0.0
        self.var_R = 1.0
        self.initialized = False
        self._last_z_H = 0.0
        self._last_z_R = 0.0
        self._last_delta_H = 0.0
        
    def update_statistics(self, H_diff: float, R_tail: float):
        """
        使用指数移动平均更新统计量
        """
        if not self.initialized:
            self.mu_H = H_diff
            self.mu_R = R_tail
            self.var_H = 0.01
            self.var_R = 0.01
            self.initialized = True
        else:
            # 更新扩散谱熵统计
            delta_H = H_diff - self.mu_H
            self.mu_H = (1 - self.alpha) * self.mu_H + self.alpha * H_diff
            self.var_H = (1 - self.alpha) * self.var_H + self.alpha * (delta_H ** 2)
            
            # 更新局部内秩统计
            delta_R = R_tail - self.mu_R
            self.mu_R = (1 - self.alpha) * self.mu_R + self.alpha * R_tail
            self.var_R = (1 - self.alpha) * self.var_R + self.alpha * (delta_R ** 2)
    
    @property
    def sigma_H(self) -> float:
        return max(self.var_H ** 0.5, 1e-6)
    
    @property
    def sigma_R(self) -> float:
        return max(self.var_R ** 0.5, 1e-6)
    
    def compute_z_scores(self, H_diff: float, R_tail: float) -> Tuple[float, float]:
        """计算Z-score标准化值"""
        z_H = (H_diff - self.mu_H) / self.sigma_H
        z_R = (R_tail - self.mu_R) / self.sigma_R
        return z_H, z_R
    
    def compute_divergence(self, H_diff: float, R_tail: float) -> float:
        """
        计算宏-微尺度分歧 ε_t
        
        公式:
        ε_t = |z_R - z_H| + γ * I[z_R > τ_R ∧ |z_H| < τ_H]
        
        注意：先用历史统计量计算z-score，再更新统计量
        这样可以正确检测当前值相对于历史的偏离程度
        """
        # 先计算Z-score（使用历史统计量）
        z_H, z_R = self.compute_z_scores(H_diff, R_tail)
        
        # 计算delta_H（用于状态判定，使用更新前的统计量）
        delta_H = abs(H_diff - self.mu_H) / self.sigma_H
        
        # 缓存z-score和delta_H供状态判定使用
        self._last_z_H = z_H
        self._last_z_R = z_R
        self._last_delta_H = delta_H
        
        # 再更新统计量（用于下一步的计算）
        self.update_statistics(H_diff, R_tail)
        
        # 基础分歧：两个尺度偏离程度的绝对差异
        base_divergence = abs(z_R - z_H)
        
        # 指示增强项：典型前驱特征（微观剧烈波动，宏观稳定）
        is_precursor_pattern = (z_R > self.config.tau_R) and (abs(z_H) < self.config.tau_H)
        enhancement = self.config.gamma * float(is_precursor_pattern)
        
        epsilon = base_divergence + enhancement
        
        return epsilon
    
    def determine_state(self, epsilon: float, H_diff: float = None) -> SystemState:
        """
        三级系统状态判定
        
        状态定义:
        1. STABLE:    ε_t < θ_ε ∧ ΔH < θ_H
        2. PRECURSOR: ε_t ≥ θ_ε ∧ ΔH < θ_H
        3. DRIFT:     ΔH ≥ θ_H
        
        注意：使用缓存的delta_H（在compute_divergence中基于更新前的统计量计算）
        """
        # 使用缓存的delta_H（已在compute_divergence中计算）
        delta_H = self._last_delta_H
        
        # 状态判定（按优先级）
        if delta_H >= self.config.theta_H:
            return SystemState.DRIFT
        elif epsilon >= self.config.theta_epsilon:
            return SystemState.PRECURSOR
        else:
            return SystemState.STABLE


class GradedResponseController:
    """
    分级响应控制器（优化版本）

    根据系统状态动态调整Backbone的适应策略
    增加安全机制防止过度更新
    """

    def __init__(self, config: PRISM2Config, device: str = 'cuda'):
        self.config = config
        self.device = device

        # 安全机制：跟踪更新状态
        self._consecutive_updates = 0       # 连续更新次数
        self._total_updates = 0             # 总更新次数
        self._total_steps = 0               # 总步数
        self._stable_steps = 0              # STABLE状态计数器（用于间隔更新）
        self._update_cooldown = 0           # 更新冷却计数器
        self._max_consecutive = getattr(config, 'max_consecutive_updates', 5)

        # STABLE更新策略
        self._stable_update_interval = getattr(config, 'stable_update_interval', 5)
        self._stable_lr_scale = getattr(config, 'stable_lr_scale', 0.5)

    def respond(
        self,
        state: SystemState,
        backbone: nn.Module,
        X_t: Tensor,
        embeddings: Tensor,
        z_t: Tensor,
        available_samples: list,
        divergence_history: list,
        criterion: nn.Module = None,
        x_mark: Tensor = None
    ) -> Tuple[Tensor, bool]:
        """
        根据状态执行相应的响应策略

        返回:
            Y_hat: 预测结果
            updated: 是否进行了模型更新
        """
        if criterion is None:
            criterion = nn.MSELoss()

        self._total_steps += 1

        if state == SystemState.STABLE:
            # 保守更新策略：STABLE状态每N步更新一次
            self._consecutive_updates = 0
            self._stable_steps += 1

            # 检查是否应该更新（每stable_update_interval步更新一次）
            should_update = (self._stable_update_interval > 0 and
                           self._stable_steps % self._stable_update_interval == 0)

            if should_update and len(available_samples) >= 2:
                self._total_updates += 1
                # STABLE状态使用较低学习率的轻量更新
                return self._stable_light_response(
                    backbone, X_t, embeddings, z_t, available_samples,
                    divergence_history, criterion, x_mark
                ), True
            else:
                # 只推理，不更新
                return self._stable_response(backbone, X_t, x_mark), False
        elif state == SystemState.PRECURSOR:
            self._stable_steps = 0  # 重置稳定计数器
            self._consecutive_updates += 1
            self._total_updates += 1
            return self._precursor_response(
                backbone, X_t, embeddings, z_t, available_samples,
                divergence_history, criterion, x_mark
            ), True
        else:  # DRIFT
            self._stable_steps = 0  # 重置稳定计数器
            self._consecutive_updates += 1
            self._total_updates += 1
            return self._drift_response(
                backbone, X_t, embeddings, z_t, available_samples, criterion, x_mark
            ), True

    def reset_counters(self):
        """重置更新计数器"""
        self._consecutive_updates = 0
        self._total_updates = 0
        self._total_steps = 0
        self._stable_steps = 0
        self._update_cooldown = 0
    
    def _stable_response(self, backbone: nn.Module, X_t: Tensor, x_mark: Tensor = None) -> Tensor:
        """
        稳态响应：Inference Only
        """
        backbone.eval()
        with torch.no_grad():
            if X_t.dim() == 2:
                X_t = X_t.unsqueeze(0)
            if x_mark is not None and x_mark.dim() == 2:
                x_mark = x_mark.unsqueeze(0)
            # 确保数据在正确的设备上
            try:
                device = next(backbone.parameters()).device
            except StopIteration:
                device = self.device
            if X_t.device != device:
                X_t = X_t.to(device)
            if x_mark is not None and x_mark.device != device:
                x_mark = x_mark.to(device)
            # 调用backbone，支持需要x_mark的模型（如iTransformer）
            if x_mark is not None:
                Y_hat = backbone(X_t, x_mark)
            else:
                Y_hat = backbone(X_t)
            # 处理元组输出
            if isinstance(Y_hat, tuple):
                Y_hat = Y_hat[0]
        return Y_hat.squeeze(0) if Y_hat.size(0) == 1 else Y_hat

    def _stable_light_response(
        self,
        backbone: nn.Module,
        X_t: Tensor,
        embeddings: Tensor,
        z_t: Tensor,
        available_samples: list,
        divergence_history: list,
        criterion: nn.Module,
        x_mark: Tensor = None
    ) -> Tensor:
        """
        STABLE状态的轻量更新响应

        使用较低的学习率和较少的样本进行保守更新
        """
        # 只使用最近的少量样本进行更新
        n_samples = min(4, len(available_samples))
        recent_samples = []
        for i in range(n_samples):
            sample = available_samples[-(i+1)]
            X, Y = sample[0], sample[1]
            x_mark_sample = sample[4] if len(sample) > 4 else None
            epsilon = sample[5] if len(sample) > 5 else 0.0
            recent_samples.append((X, Y, x_mark_sample, epsilon))

        if len(recent_samples) < 2:
            return self._stable_response(backbone, X_t, x_mark)

        # 使用降低的学习率进行更新
        original_lr = self.config.precursor_lr
        try:
            # 临时降低学习率
            self.config.precursor_lr = original_lr * self._stable_lr_scale

            # 计算均匀权重（STABLE状态不需要流形感知加权）
            weights = torch.ones(len(recent_samples), device=self.device)

            # 执行轻量更新
            self._fast_weight_update(backbone, recent_samples, weights, criterion)
        finally:
            # 恢复原始学习率
            self.config.precursor_lr = original_lr

        # 预测
        return self._stable_response(backbone, X_t, x_mark)

    def _precursor_response(
        self,
        backbone: nn.Module,
        X_t: Tensor,
        embeddings: Tensor,
        z_t: Tensor,
        available_samples: list,
        divergence_history: list,
        criterion: nn.Module,
        x_mark: Tensor = None
    ) -> Tensor:
        """
        前驱态响应：流形敏感的Fast-Weight Tuning
        """
        # 流形邻域检索 - 返回完整样本 (X, Y, x_mark, epsilon)
        relevant_samples, sample_indices = self._manifold_neighbor_retrieval(
            z_t, available_samples, k=self.config.precursor_neighbors
        )
        
        if len(relevant_samples) == 0:
            return self._stable_response(backbone, X_t, x_mark)
        
        # 计算流形感知权重（使用样本自带的epsilon）
        weights = self._compute_manifold_weights(relevant_samples)
        
        # 快速权重更新
        self._fast_weight_update(backbone, relevant_samples, weights, criterion)
        
        # 预测
        return self._stable_response(backbone, X_t, x_mark)
    
    def _drift_response(
        self,
        backbone: nn.Module,
        X_t: Tensor,
        embeddings: Tensor,
        z_t: Tensor,
        available_samples: list,
        criterion: nn.Module,
        x_mark: Tensor = None
    ) -> Tensor:
        """
        漂移态响应：Multi-Step Adaptation
        """
        # 扩展邻域检索
        extended_samples, _ = self._manifold_neighbor_retrieval(
            z_t, available_samples, k=self.config.drift_neighbors
        )
        
        if len(extended_samples) == 0:
            return self._stable_response(backbone, X_t, x_mark)
        
        # 多步适应
        self._multi_step_adaptation(backbone, extended_samples, criterion)
        
        # 预测
        return self._stable_response(backbone, X_t, x_mark)
    
    def _manifold_neighbor_retrieval(
        self,
        z_t: Tensor,
        available_samples: list,
        k: int,
        prioritize_recent: bool = True
    ) -> Tuple[list, list]:
        """
        流形邻域检索 (优化版：混合最近样本和流形邻居)

        返回完整样本信息包括 x_mark 和 epsilon
        样本格式: (X, Y, timestamp, embedding, x_mark, epsilon)

        优化策略：总是包含最近的几个样本以确保适应最新数据分布
        """
        if len(available_samples) == 0:
            return [], []

        n_samples = len(available_samples)

        # 优化策略：混合最近样本和流形邻居
        if prioritize_recent:
            # 首先选取最近的样本 (最近的1-2个样本)
            n_recent = min(2, n_samples)
            recent_indices = list(range(n_samples - n_recent, n_samples))

            # 然后从剩余样本中选取流形邻居
            k_remaining = min(k - n_recent, n_samples - n_recent)

            if k_remaining > 0 and n_samples > n_recent:
                # 只在非最近样本中查找邻居
                older_samples = available_samples[:-n_recent] if n_recent > 0 else available_samples

                sample_embeddings = torch.stack([s[3] for s in older_samples])
                if sample_embeddings.dim() > 2:
                    sample_embeddings = sample_embeddings.view(sample_embeddings.size(0), -1)

                if z_t.dim() > 2:
                    z_t_flat = z_t.view(-1)
                else:
                    z_t_flat = z_t
                if z_t_flat.dim() == 1:
                    z_t_flat = z_t_flat.unsqueeze(0)
                if z_t_flat.dim() > 2:
                    z_t_flat = z_t_flat.view(1, -1)

                distances = torch.norm(sample_embeddings - z_t_flat, dim=1)
                k_actual = min(k_remaining, len(older_samples))
                _, neighbor_indices = torch.topk(distances, k_actual, largest=False)

                if neighbor_indices.dim() > 1:
                    neighbor_indices = neighbor_indices.squeeze()
                neighbor_list = neighbor_indices.tolist()
                if not isinstance(neighbor_list, list):
                    neighbor_list = [neighbor_list]

                # 合并索引 (最近样本 + 流形邻居)
                indices_list = recent_indices + neighbor_list
            else:
                indices_list = recent_indices
        else:
            # 原始的纯流形邻居检索
            sample_embeddings = torch.stack([s[3] for s in available_samples])
            if sample_embeddings.dim() > 2:
                sample_embeddings = sample_embeddings.view(sample_embeddings.size(0), -1)

            if z_t.dim() > 2:
                z_t = z_t.view(-1)
            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)
            if z_t.dim() > 2:
                z_t = z_t.view(1, -1)

            distances = torch.norm(sample_embeddings - z_t, dim=1)
            k = min(k, len(available_samples))
            _, indices = torch.topk(distances, k, largest=False)

            if indices.dim() > 1:
                indices = indices.squeeze()
            indices_list = indices.tolist()
            if not isinstance(indices_list, list):
                indices_list = [indices_list]

        # 提取完整样本: (X, Y, x_mark, epsilon)
        # 样本格式: (X, Y, timestamp, embedding, x_mark, epsilon)
        relevant_samples = []
        for i in indices_list:
            sample = available_samples[i]
            X, Y = sample[0], sample[1]
            x_mark = sample[4] if len(sample) > 4 else None
            epsilon = sample[5] if len(sample) > 5 else 0.0
            relevant_samples.append((X, Y, x_mark, epsilon))

        return relevant_samples, indices_list
    
    def _compute_manifold_weights(
        self,
        samples: list
    ) -> Tensor:
        """
        计算流形感知的样本权重
        
        公式: w_i = 1 + tanh(λ * ε_i)
        
        参数:
            samples: 样本列表，每个样本格式为 (X, Y, x_mark, epsilon)
        """
        weights = []
        for sample in samples:
            # 从样本中直接获取epsilon（第4个元素，索引3）
            epsilon = sample[3] if len(sample) > 3 else 0.0
            w = 1.0 + torch.tanh(torch.tensor(self.config.weight_lambda * epsilon))
            weights.append(w)
        
        return torch.stack(weights).to(self.device)
    
    def _fast_weight_update(
        self,
        backbone: nn.Module,
        samples: list,
        weights: Tensor,
        criterion: nn.Module
    ):
        """
        快速权重更新 (PRECURSOR阶段) - 优化版本

        参数:
            samples: 样本列表，每个样本格式为 (X, Y, x_mark, epsilon)
        """
        # Debug flag - set to True to enable debug output
        _DEBUG = False  # Disabled for production

        # 安全检查：样本数量最小阈值（降低以支持小批量更新）
        if len(samples) < 2:
            if _DEBUG:
                print(f"  [_fast_weight_update] Early return: len(samples)={len(samples)} < 4")
            return

        backbone.train()

        # 确保backbone参数需要梯度
        has_grad = False
        for param in backbone.parameters():
            param.requires_grad_(True)
            has_grad = True

        if not has_grad:
            return

        # 获取backbone的实际设备
        try:
            device = next(backbone.parameters()).device
        except StopIteration:
            device = self.device

        # 保存当前参数用于回滚
        param_backup = {name: param.data.clone() for name, param in backbone.named_parameters()}

        try:
            # 构建批次
            X_batch = torch.stack([s[0].clone().detach() for s in samples]).to(device)
            Y_batch = torch.stack([s[1].clone().detach() for s in samples]).to(device)
            weights = weights.to(device)

            if _DEBUG:
                print(f"  [_fast_weight_update] X_batch shape: {X_batch.shape}, Y_batch shape: {Y_batch.shape}")

            # 检查输入有效性
            if torch.isnan(X_batch).any() or torch.isnan(Y_batch).any():
                if _DEBUG:
                    print(f"  [_fast_weight_update] Early return: NaN in X_batch or Y_batch")
                return

            # 检查是否有x_mark
            has_x_mark = any(s[2] is not None for s in samples)
            x_mark_batch = None
            if has_x_mark:
                x_marks = []
                for s in samples:
                    if s[2] is not None:
                        x_marks.append(s[2].clone().detach())
                    else:
                        x_marks.append(torch.zeros(*s[0].shape[:-1], 7, device=device))
                x_mark_batch = torch.stack(x_marks).to(device)

            # 前向传播
            with torch.enable_grad():
                try:
                    if x_mark_batch is not None:
                        Y_pred = backbone(X_batch, x_mark_batch)
                    else:
                        Y_pred = backbone(X_batch)
                except TypeError:
                    try:
                        default_x_mark = torch.zeros(*X_batch.shape[:-1], 7, device=device)
                        Y_pred = backbone(X_batch, default_x_mark)
                    except:
                        if _DEBUG:
                            print(f"  [_fast_weight_update] Early return: TypeError in forward")
                        return

                if isinstance(Y_pred, tuple):
                    Y_pred = Y_pred[0]

                # 检查输出有效性
                if torch.isnan(Y_pred).any():
                    if _DEBUG:
                        print(f"  [_fast_weight_update] Early return: NaN in Y_pred")
                    return

                # 计算加权损失
                losses = ((Y_pred - Y_batch) ** 2).mean(dim=list(range(1, Y_pred.dim())))
                weighted_loss = (weights * losses).mean()

                if _DEBUG:
                    print(f"  [_fast_weight_update] weighted_loss: {weighted_loss.item():.6f}")

                # 检查损失有效性
                if torch.isnan(weighted_loss) or weighted_loss.item() > 1e6:
                    if _DEBUG:
                        print(f"  [_fast_weight_update] Early return: invalid loss")
                    return

                backbone.zero_grad()
                weighted_loss.backward()

            # 梯度裁剪 - max_norm=50.0 提供稳定性同时允许有效更新
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=50.0)

            # 应用更新
            update_count = 0
            with torch.no_grad():
                for param in backbone.parameters():
                    if param.grad is not None:
                        param.data -= self.config.precursor_lr * param.grad
                        update_count += 1

            if _DEBUG:
                print(f"  [_fast_weight_update] Updated {update_count} params with lr={self.config.precursor_lr}")

            # 验证更新后的参数没有NaN
            for param in backbone.parameters():
                if torch.isnan(param.data).any():
                    if _DEBUG:
                        print(f"  [_fast_weight_update] Rolling back due to NaN in params")
                    # 回滚到备份
                    for name, param in backbone.named_parameters():
                        param.data.copy_(param_backup[name])
                    return

        except Exception as e:
            # 发生任何错误时回滚
            for name, param in backbone.named_parameters():
                if name in param_backup:
                    param.data.copy_(param_backup[name])
    
    def _multi_step_adaptation(
        self,
        backbone: nn.Module,
        samples: list,
        criterion: nn.Module
    ):
        """
        多步流形引导适应 (DRIFT阶段) - 优化版本

        参数:
            samples: 样本列表，每个样本格式为 (X, Y, x_mark, epsilon)
        """
        # 安全检查：样本数量最小阈值（降低以支持小批量更新）
        if len(samples) < 2:
            return

        backbone.train()

        # 确保backbone参数需要梯度
        has_grad = False
        for param in backbone.parameters():
            param.requires_grad_(True)
            has_grad = True

        if not has_grad:
            return

        # 获取backbone的实际设备
        try:
            device = next(backbone.parameters()).device
        except StopIteration:
            device = self.device

        # 保存当前参数用于回滚
        param_backup = {name: param.data.clone() for name, param in backbone.named_parameters()}

        try:
            # 样本格式: (X, Y, x_mark, epsilon)
            X_batch = torch.stack([s[0].clone().detach() for s in samples]).to(device)
            Y_batch = torch.stack([s[1].clone().detach() for s in samples]).to(device)

            # 检查输入有效性
            if torch.isnan(X_batch).any() or torch.isnan(Y_batch).any():
                return

            # 检查是否有x_mark
            has_x_mark = any(s[2] is not None for s in samples)
            x_mark_batch = None
            if has_x_mark:
                x_marks = []
                for s in samples:
                    if s[2] is not None:
                        x_marks.append(s[2].clone().detach())
                    else:
                        x_marks.append(torch.zeros(*s[0].shape[:-1], 7, device=device))
                x_mark_batch = torch.stack(x_marks).to(device)

            # 记录初始损失用于早停
            initial_loss = None

            for step in range(self.config.drift_steps):
                with torch.enable_grad():
                    try:
                        if x_mark_batch is not None:
                            Y_pred = backbone(X_batch, x_mark_batch)
                        else:
                            Y_pred = backbone(X_batch)
                    except TypeError:
                        try:
                            default_x_mark = torch.zeros(*X_batch.shape[:-1], 7, device=device)
                            Y_pred = backbone(X_batch, default_x_mark)
                        except:
                            return

                    if isinstance(Y_pred, tuple):
                        Y_pred = Y_pred[0]

                    # 检查输出有效性
                    if torch.isnan(Y_pred).any():
                        # 回滚并返回
                        for name, param in backbone.named_parameters():
                            param.data.copy_(param_backup[name])
                        return

                    loss = criterion(Y_pred, Y_batch)

                    # 检查损失有效性
                    if torch.isnan(loss) or loss.item() > 1e6:
                        for name, param in backbone.named_parameters():
                            param.data.copy_(param_backup[name])
                        return

                    # 记录初始损失
                    if initial_loss is None:
                        initial_loss = loss.item()

                    # 早停：如果损失反而增加了很多，停止更新
                    if loss.item() > initial_loss * 2.0:
                        for name, param in backbone.named_parameters():
                            param.data.copy_(param_backup[name])
                        return

                    backbone.zero_grad()
                    loss.backward()

                # 梯度裁剪 - max_norm=50.0 提供稳定性同时允许有效更新
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=50.0)

                with torch.no_grad():
                    for param in backbone.parameters():
                        if param.grad is not None:
                            param.data -= self.config.drift_lr * param.grad

            # 验证更新后的参数没有NaN
            for param in backbone.parameters():
                if torch.isnan(param.data).any():
                    for name, param in backbone.named_parameters():
                        param.data.copy_(param_backup[name])
                    return

        except Exception as e:
            # 发生任何错误时回滚
            for name, param in backbone.named_parameters():
                if name in param_backup:
                    param.data.copy_(param_backup[name])
