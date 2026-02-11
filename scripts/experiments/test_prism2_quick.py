"""
PRISM2 快速测试脚本

测试优化后的配置是否能解决TCN+ETTh2的严重性能问题
"""

import os
import sys
import subprocess
from pathlib import Path

# 设置工作目录
WORK_DIR = Path(__file__).parent.parent.parent
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

from settings import get_prism2_config, pretrain_lr_online_dict

def run_command(cmd, timeout=6000):
    """运行命令并返回输出"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORK_DIR)
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

def test_single_config(model, dataset, pred_len):
    """测试单个配置"""
    # 获取优化后的配置
    config = get_prism2_config(dataset, model)

    print(f"\n{'='*60}")
    print(f"Testing: {model} | {dataset} | pred_len={pred_len}")
    print(f"{'='*60}")
    print(f"Config:")
    print(f"  - theta_epsilon: {config.get('theta_epsilon', 2.0)}")
    print(f"  - theta_H: {config.get('theta_H', 2.5)}")
    print(f"  - precursor_lr: {config.get('precursor_lr', 0.0005)}")
    print(f"  - drift_lr: {config.get('drift_lr', 0.0001)}")
    print(f"  - drift_steps: {config.get('drift_steps', 2)}")
    print(f"  - warmup_steps: {config.get('warmup_steps', 100)}")
    print(f"  - gamma: {config.get('gamma', 0.3)}")

    # 获取预训练学习率
    lr = pretrain_lr_online_dict.get(model, {}).get(dataset, 0.0001)

    # 运行PRISM2测试
    cmd = [
        "python", "run.py",
        "--model", model,
        "--dataset", dataset,
        "--features", "M",
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--online_method", "PRISM2",
        "--pretrain",
        "--only_test",
        "--wo_valid",
        "--learning_rate", str(lr),
        "--online_learning_rate", "0.0001",
        "--itr", "1"
    ]

    output = run_command(cmd)
    print("\n--- Output ---")
    # 只打印关键行
    for line in output.split('\n'):
        if any(key in line.lower() for key in ['mse', 'mae', 'stable', 'precursor', 'drift', 'divergence', 'error']):
            print(line)

    return output

def main():
    print("PRISM2 Quick Test - Optimized Configuration")
    print("="*60)

    # 测试最严重的问题组合：TCN + ETTh2
    test_cases = [
        ("TCN", "ETTh2", 24),
        ("TCN", "ETTh2", 48),
        # ("PatchTST", "ETTh2", 24),
        # ("iTransformer", "ETTh2", 24),
    ]

    for model, dataset, pred_len in test_cases:
        test_single_config(model, dataset, pred_len)

    print("\n" + "="*60)
    print("Quick test completed!")
    print("="*60)

if __name__ == "__main__":
    main()
