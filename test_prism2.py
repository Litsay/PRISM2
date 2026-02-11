"""
PRISM2 框架测试脚本

用于验证PRISM2各模块的基本功能
"""

import torch
import torch.nn as nn
import sys

# 测试导入
print("Testing imports...")
try:
    from adapter.prism2_modules import (
        PRISM2Config,
        PRISM2State,
        SystemState,
        ManifoldRepresentationBuilder,
        GeometricFeatureExtractor,
        DivergenceDetector,
        GradedResponseController
    )
    from adapter.prism2 import PRISM2, PRISM2Wrapper
    from exp.exp_prism2 import Exp_PRISM2
    print("  All imports successful!")
except ImportError as e:
    print(f"  Import error: {e}")
    sys.exit(1)


def test_config():
    """测试配置类"""
    print("\nTesting PRISM2Config...")
    config = PRISM2Config()
    assert config.window_size == 256
    assert config.theta_epsilon == 1.5
    assert config.theta_H == 2.0
    print("  PRISM2Config test passed!")


def test_geometric_features():
    """测试几何特征计算"""
    print("\nTesting GeometricFeatureExtractor...")
    config = PRISM2Config()
    extractor = GeometricFeatureExtractor(config)
    
    # 测试扩散谱熵
    embeddings = torch.randn(100, 64)
    K = torch.exp(-torch.cdist(embeddings, embeddings) ** 2 / 2)
    H_diff = extractor.compute_diffusion_spectral_entropy(K)
    assert 0 <= H_diff <= 10, f"DSE out of range: {H_diff}"
    print(f"  DSE: {H_diff:.4f}")
    
    # 测试LIRTI
    knn_distances = torch.rand(100, 20) + 0.1
    R_tail = extractor.compute_window_level_lirti(knn_distances)
    assert R_tail > 0, f"LIRTI should be positive: {R_tail}"
    print(f"  LIRTI: {R_tail:.4f}")
    print("  GeometricFeatureExtractor test passed!")


def test_divergence_detector():
    """测试分歧检测器"""
    print("\nTesting DivergenceDetector...")
    config = PRISM2Config(theta_epsilon=1.5, theta_H=2.0)
    detector = DivergenceDetector(config)
    
    # 初始化统计量
    for _ in range(10):
        detector.update_statistics(1.0, 1.0)
    
    # STABLE: 低分歧，低宏观变化
    eps = detector.compute_divergence(1.0, 1.0)
    state = detector.determine_state(eps, 1.0)
    assert state == SystemState.STABLE, f"Expected STABLE, got {state}"
    print(f"  STABLE state: divergence={eps:.4f}")
    
    # PRECURSOR: 高分歧（微观突变），低宏观变化
    eps = detector.compute_divergence(1.0, 5.0)
    state = detector.determine_state(eps, 1.0)
    assert state == SystemState.PRECURSOR, f"Expected PRECURSOR, got {state}"
    print(f"  PRECURSOR state: divergence={eps:.4f}")
    
    # DRIFT: 高宏观变化
    eps = detector.compute_divergence(10.0, 10.0)
    state = detector.determine_state(eps, 10.0)
    assert state == SystemState.DRIFT, f"Expected DRIFT, got {state}"
    print(f"  DRIFT state: divergence={eps:.4f}")
    
    print("  DivergenceDetector test passed!")


def test_prism2_with_simple_backbone():
    """测试PRISM2与简单backbone的集成"""
    print("\nTesting PRISM2 with simple backbone...")
    
    # 创建简单的backbone
    class SimpleBackbone(nn.Module):
        def __init__(self, input_dim, seq_len, pred_len):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.input_dim = input_dim
            self.fc = nn.Linear(seq_len * input_dim, pred_len * input_dim)
        
        def forward(self, x):
            # x: (batch, seq_len, input_dim)
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            out = self.fc(x_flat)
            return out.view(batch_size, self.pred_len, self.input_dim)
        
        def extract_features(self, x):
            batch_size = x.size(0)
            return x.view(batch_size, -1)[:, :64]  # 返回64维特征
    
    # 创建args模拟对象
    class Args:
        freeze = False
        prism2_window_size = 64
        prism2_k_neighbors = 10
        prism2_lid_k = 15
        prism2_gamma = 0.5
        prism2_tau_R = 2.0
        prism2_tau_H = 1.5
        prism2_ema_alpha = 0.1
        prism2_theta_epsilon = 1.5
        prism2_theta_H = 2.0
        prism2_precursor_lr = 0.01
        prism2_precursor_neighbors = 16
        prism2_drift_lr = 0.001
        prism2_drift_neighbors = 32
        prism2_drift_steps = 3
        prism2_weight_lambda = 1.0
        prism2_warmup_steps = 20
    
    args = Args()
    backbone = SimpleBackbone(input_dim=7, seq_len=96, pred_len=24)
    
    # 创建PRISM2
    prism2 = PRISM2(backbone, args)
    print(f"  PRISM2 created with config:")
    print(f"    - Window size: {prism2.config.window_size}")
    print(f"    - Theta epsilon: {prism2.config.theta_epsilon}")
    print(f"    - Warmup steps: {prism2.config.warmup_steps}")
    
    # 测试前向传播
    x = torch.randn(2, 96, 7)
    y = prism2(x)
    assert y.shape == (2, 24, 7), f"Output shape mismatch: {y.shape}"
    print(f"  Forward pass: input={x.shape}, output={y.shape}")
    
    # 测试在线学习模式
    prism2.flag_online_learning = True
    
    # 模拟在线学习循环
    for t in range(50):
        X_t = torch.randn(96, 7)
        Y_hat = prism2._online_step(X_t)
        
        # 添加延迟标签
        if t > 24:
            Y_delayed = torch.randn(24, 7)
            prism2.add_labeled_sample(
                prism2.state.input_buffer[-(24+1)],
                Y_delayed
            )
    
    # 获取统计信息
    stats = prism2.get_statistics()
    print(f"  Online learning stats:")
    print(f"    - Total steps: {stats['total_steps']}")
    print(f"    - Current state: {stats['current_state']}")
    print(f"    - Stable ratio: {stats['stable_ratio']:.2%}")
    print(f"    - Precursor ratio: {stats['precursor_ratio']:.2%}")
    print(f"    - Drift ratio: {stats['drift_ratio']:.2%}")
    
    print("  PRISM2 integration test passed!")


def main():
    print("=" * 50)
    print("PRISM2 Framework Tests")
    print("=" * 50)
    
    test_config()
    test_geometric_features()
    test_divergence_detector()
    test_prism2_with_simple_backbone()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
