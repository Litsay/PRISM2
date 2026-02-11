"""
PRISM2 实验类

继承Exp_Online，实现PRISM2框架的训练和在线学习
"""

import copy
import numpy as np
import torch
from tqdm import tqdm

from adapter import prism2
from adapter.prism2_modules import SystemState
from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_online import Exp_Online
from util.metrics import update_metrics, calculate_metrics


class Exp_PRISM2(Exp_Online):
    """
    PRISM2实验类
    
    实现基于宏-微尺度几何分歧的概念漂移主动感知与适应
    """
    
    def __init__(self, args):
        args = copy.deepcopy(args)
        # PRISM2通过分级响应策略动态控制更新，不需要预先冻结
        # 在STABLE状态下自动跳过更新
        if args.freeze:
            print("PRISM2: Disabling freeze mode - using state-aware selective updates instead")
            args.freeze = False
        super(Exp_PRISM2, self).__init__(args)
        self.online_phases = ['val', 'test', 'online']
        
    def _build_model(self, model=None, framework_class=None):
        """
        构建PRISM2包装的模型
        """
        model = super()._build_model(model, framework_class=prism2.PRISM2)
        print(f"PRISM2 Model Built:")
        print(f"  - Window Size: {model.config.window_size}")
        print(f"  - Theta Epsilon: {model.config.theta_epsilon}")
        print(f"  - Theta H: {model.config.theta_H}")
        print(f"  - Warmup Steps: {model.config.warmup_steps}")
        return model
    
    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        """
        PRISM2在线学习主循环
        
        实现状态感知的分级响应策略
        
        注意：更新逻辑完全由PRISM2内部的GradedResponseController处理
        Exp层只负责数据流转和统计，不进行额外的外部更新
        """
        self.phase = phase
        
        # 重置PRISM2状态
        self._model.reset_state()
        
        if online_data is None:
            online_data = get_dataset(
                self.args, phase, self.device,
                wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                **self.wrap_data_kwargs
            )
        
        online_loader = get_dataloader(online_data, self.args, flag='online')
        
        if self.args.do_predict:
            predictions = []
        
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}
        
        # 状态统计
        state_counts = {'STABLE': 0, 'PRECURSOR': 0, 'DRIFT': 0, 'WARMUP': 0}
        
        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=10, desc=f"PRISM2 {phase}")
        
        for i, (recent_data, current_data) in enumerate(online_loader):
            # 只在测试阶段启用完整的PRISM2在线学习
            if phase == 'test':
                # 设置在线学习模式
                self._model.flag_online_learning = True
                
                # 步骤1: 处理recent_data - 添加到样本池（包含x_mark）
                self._process_recent_data(recent_data)
            
            # 步骤2: 预测当前数据
            # 在线学习模式下，PRISM2内部会自动：
            # 1. 提取流形嵌入
            # 2. 计算宏-微尺度特征
            # 3. 判定系统状态
            # 4. 执行分级响应（PRECURSOR/DRIFT时进行流形邻域检索和更新）
            # 注意：需要设置eval模式以确保推理行为一致
            self._model.backbone.eval()
            outputs = self.forward(current_data)
            
            # 步骤3: 获取当前状态（在forward后状态已更新）
            if phase == 'test':
                current_state = self._model.get_current_state()
                state_counts[current_state] = state_counts.get(current_state, 0) + 1
                self._model.flag_online_learning = False
            else:
                # 验证阶段
                current_state = 'STABLE'
                state_counts[current_state] = state_counts.get(current_state, 0) + 1
            
            # 步骤4: 更新统计指标
            with torch.no_grad():
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs_for_pred = outputs[0]
                    else:
                        outputs_for_pred = outputs
                    predictions.append(outputs_for_pred.detach().cpu().numpy())
                
                update_metrics(
                    outputs, 
                    current_data[self.label_position].to(self.device), 
                    statistics, 
                    target_variate
                )
        
        # 计算指标
        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        
        # 打印PRISM2统计信息
        if phase == 'test':
            print(f'\nmse:{mse}, mae:{mae}')
            prism2_stats = self._model.get_statistics()
            print(f"PRISM2 State Distribution:")
            print(f"  - WARMUP: {state_counts.get('WARMUP', 0)}")
            print(f"  - STABLE: {prism2_stats['stable_count']} ({prism2_stats['stable_ratio']:.2%})")
            print(f"  - PRECURSOR: {prism2_stats['precursor_count']} ({prism2_stats['precursor_ratio']:.2%})")
            print(f"  - DRIFT: {prism2_stats['drift_count']} ({prism2_stats['drift_ratio']:.2%})")
            print(f"  - Avg Divergence: {prism2_stats['avg_divergence']:.4f}")
        
        if self.args.do_predict:
            return mse, mae, online_data, predictions
        else:
            return mse, mae, online_data
    
    def _process_recent_data(self, recent_data):
        """
        处理recent_data，添加到PRISM2的样本池
        
        recent_data格式: (batch_x, batch_y, batch_x_mark, batch_y_mark)
        """
        batch_x = recent_data[0]
        batch_y = recent_data[1]
        # 获取x_mark（如果存在）
        batch_x_mark = recent_data[2] if len(recent_data) > 2 else None
        
        if not self.args.pin_gpu:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.to(self.device)
        
        # 处理batch维度
        if batch_x.dim() == 4:  # (1, recent_num, seq_len, num_var)
            batch_x = batch_x.squeeze(0)
            batch_y = batch_y.squeeze(0)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.squeeze(0)
        elif batch_x.dim() == 3 and batch_x.size(0) == 1:
            batch_x = batch_x.squeeze(0)
            batch_y = batch_y.squeeze(0)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.squeeze(0)
        
        # 添加样本到池中（包含x_mark）
        if batch_x.dim() == 2:  # 单个样本
            x_mark = batch_x_mark if batch_x_mark is not None else None
            self._model.add_labeled_sample(batch_x, batch_y, x_mark=x_mark)
        else:  # 多个样本
            for j in range(batch_x.size(0)):
                x_mark = batch_x_mark[j] if batch_x_mark is not None else None
                self._model.add_labeled_sample(batch_x[j], batch_y[j], x_mark=x_mark)
    
    def update_valid(self, valid_data=None, valid_dataloader=None):
        """
        验证集上的在线学习
        
        注意：更新逻辑完全由PRISM2内部处理
        """
        self.phase = 'online'
        
        if valid_data is None:
            valid_data = get_dataset(
                self.args, 'val', self.device,
                wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                **self.wrap_data_kwargs, 
                take_post=self.args.pred_len - 1
            )
        
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        predictions = []
        
        # 重置状态
        self._model.reset_state()
        self._model.flag_online_learning = True
        
        for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
            # 处理recent数据（添加到样本池）
            self._process_recent_data(recent_batch)
            
            # 预测（PRISM2内部会自动执行状态感知的分级响应）
            outputs = self.forward(current_batch)
            
            if self.args.do_predict:
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
        
        self._model.flag_online_learning = False
        
        trainable_params = sum([param.nelement() if param.requires_grad else 0 
                               for param in self._model.parameters()])
        print(f'Trainable Params: {trainable_params}')
        
        return predictions
    
    def _update(self, batch, criterion, optimizer, scaler=None):
        """
        训练阶段的更新（预训练）
        """
        self._model.flag_update = True
        loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        
        # 更新recent_batch
        batch_x, batch_y = batch[0], batch[1]
        self._model.update_recent_batch(batch_x, batch_y)
        
        self._model.flag_update = False
        return loss, outputs
    
    def _update_online(self, batch, criterion, optimizer, scaler=None):
        """
        在线学习阶段的更新
        
        注意：实际更新由PRISM2内部的GradedResponseController自动处理
        这里只负责前向传播和返回结果
        """
        self._model.flag_online_learning = True
        
        # 前向传播（PRISM2内部会自动执行状态感知的分级响应）
        outputs = self.forward(batch)
        
        # 计算loss用于记录（但不用于更新，更新由PRISM2内部处理）
        batch_y = batch[1]
        if not self.args.pin_gpu:
            batch_y = batch_y.to(self.device)
        
        if isinstance(outputs, tuple):
            outputs_for_loss = outputs[0]
        else:
            outputs_for_loss = outputs
        
        with torch.no_grad():
            loss = criterion(outputs_for_loss, batch_y)
        
        # 更新recent_batch
        batch_x, batch_y = batch[0], batch[1]
        self._model.update_recent_batch(batch_x, batch_y)
        
        self._model.flag_online_learning = False
        return loss, outputs
    
    def analysis_online(self):
        """
        分析在线学习性能
        """
        return super().analysis_online()
    
    def predict(self, path, setting, load=False):
        """
        预测
        """
        self.update_valid()
        res = self.online()
        np.save('./results/' + setting + '_pred.npy', np.vstack(res[-1]))
        return res
