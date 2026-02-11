"""
PRISM2 超参数搜索脚本

使用网格搜索或随机搜索来优化PRISM2超参数
"""

import os
import sys
import itertools
import random
import argparse
import subprocess
from datetime import datetime
import csv

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def get_search_space():
    """定义超参数搜索空间"""
    return {
        # 流形表征参数
        'window_size': [64, 128, 256],
        'warmup_steps': [30, 50, 80],

        # 分歧检测参数
        'theta_epsilon': [1.0, 1.5, 2.0],
        'theta_H': [1.5, 2.0, 2.5],
        'gamma': [0.3, 0.5, 0.7],
        'ema_alpha': [0.05, 0.1, 0.15],

        # 学习率参数
        'precursor_lr': [0.00005, 0.0001, 0.0002, 0.0003],
        'drift_lr': [0.0001, 0.0002, 0.0003],
        'drift_steps': [2, 3, 4],

        # 邻域参数
        'precursor_neighbors': [8, 16, 32],
        'drift_neighbors': [16, 32, 64],

        # STABLE更新策略
        'stable_update_interval': [0, 3, 5, 10],  # 0表示不更新
        'stable_lr_scale': [0.2, 0.3, 0.5],
    }


def get_focused_search_space(model, dataset):
    """获取针对特定模型和数据集的聚焦搜索空间"""

    # 基于分析结果的聚焦搜索空间
    base_space = {
        'window_size': [64, 128],
        'warmup_steps': [30, 50],
        'theta_epsilon': [1.2, 1.5, 1.8],
        'theta_H': [1.8, 2.0, 2.2],
        'ema_alpha': [0.08, 0.1, 0.12],
        'stable_update_interval': [3, 5],
        'stable_lr_scale': [0.3, 0.5],
    }

    # 模型特定的学习率空间
    if model == 'TCN':
        base_space['precursor_lr'] = [0.0001, 0.0002, 0.0003]
        base_space['drift_lr'] = [0.0002, 0.0003, 0.0005]
        base_space['drift_steps'] = [2, 3]
    elif model == 'PatchTST':
        base_space['precursor_lr'] = [0.00005, 0.0001, 0.00015]
        base_space['drift_lr'] = [0.0001, 0.0002]
        base_space['drift_steps'] = [2, 3]
    elif model == 'iTransformer':
        base_space['precursor_lr'] = [0.00005, 0.0001, 0.00015]
        base_space['drift_lr'] = [0.0001, 0.0002]
        base_space['drift_steps'] = [2, 3]

    # 数据集特定调整
    if dataset == 'ETTh2':
        # ETTh2之前过于激进，需要更保守
        base_space['theta_epsilon'] = [1.5, 1.8, 2.0]
        base_space['theta_H'] = [2.0, 2.2, 2.5]
    elif dataset == 'Weather':
        # Weather变化频繁，可以更敏感
        base_space['theta_epsilon'] = [1.0, 1.2, 1.5]
        base_space['theta_H'] = [1.5, 1.8, 2.0]

    return base_space


def generate_random_configs(search_space, n_configs=50):
    """生成随机配置"""
    configs = []
    for _ in range(n_configs):
        config = {}
        for param, values in search_space.items():
            config[param] = random.choice(values)
        configs.append(config)
    return configs


def generate_grid_configs(search_space, max_configs=100):
    """生成网格配置（如果太多则随机采样）"""
    keys = list(search_space.keys())
    values = list(search_space.values())

    all_configs = list(itertools.product(*values))

    if len(all_configs) > max_configs:
        all_configs = random.sample(all_configs, max_configs)

    configs = []
    for config_tuple in all_configs:
        config = dict(zip(keys, config_tuple))
        configs.append(config)

    return configs


def build_command(model, dataset, pred_len, config, gpu=0):
    """构建运行命令"""
    cmd = [
        'python', 'run.py',
        '--only_test',
        '--pretrain',
        '--online_method', 'PRISM2',
        '--model', model,
        '--dataset', dataset,
        '--pred_len', str(pred_len),
        '--gpu', str(gpu),
        '--wo_valid',
    ]

    # 添加PRISM2超参数
    for param, value in config.items():
        cmd.extend([f'--prism2_{param}', str(value)])

    return cmd


def run_experiment(cmd, timeout=600):
    """运行单个实验"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root
        )

        # 解析输出获取MSE和MAE
        output = result.stdout + result.stderr

        mse = None
        mae = None

        for line in output.split('\n'):
            if 'mse:' in line.lower() and 'mae:' in line.lower():
                parts = line.split(',')
                for part in parts:
                    if 'mse:' in part.lower():
                        try:
                            mse = float(part.split(':')[1].strip())
                        except:
                            pass
                    if 'mae:' in part.lower():
                        try:
                            mae = float(part.split(':')[1].strip())
                        except:
                            pass

        return mse, mae, output

    except subprocess.TimeoutExpired:
        return None, None, "Timeout"
    except Exception as e:
        return None, None, str(e)


def main():
    parser = argparse.ArgumentParser(description='PRISM2 Hyperparameter Search')
    parser.add_argument('--model', type=str, default='TCN',
                       choices=['TCN', 'PatchTST', 'iTransformer'])
    parser.add_argument('--dataset', type=str, default='ETTh2',
                       choices=['ETTh2', 'ETTm1', 'Weather'])
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--n_configs', type=int, default=30,
                       help='Number of configurations to try')
    parser.add_argument('--search_type', type=str, default='focused',
                       choices=['random', 'grid', 'focused'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path')

    args = parser.parse_args()

    # 生成搜索空间
    if args.search_type == 'focused':
        search_space = get_focused_search_space(args.model, args.dataset)
        configs = generate_random_configs(search_space, args.n_configs)
    elif args.search_type == 'random':
        search_space = get_search_space()
        configs = generate_random_configs(search_space, args.n_configs)
    else:
        search_space = get_search_space()
        configs = generate_grid_configs(search_space, args.n_configs)

    print(f"Running hyperparameter search for {args.model} on {args.dataset}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Number of configurations: {len(configs)}")
    print("=" * 60)

    # 设置输出文件
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/hp_search_{args.model}_{args.dataset}_{args.pred_len}_{timestamp}.csv"

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # 运行实验
    results = []
    best_mse = float('inf')
    best_config = None

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

        cmd = build_command(args.model, args.dataset, args.pred_len, config, args.gpu)
        mse, mae, output = run_experiment(cmd)

        if mse is not None:
            print(f"  Result: MSE={mse:.6f}, MAE={mae:.6f}")

            result = {
                'model': args.model,
                'dataset': args.dataset,
                'pred_len': args.pred_len,
                'mse': mse,
                'mae': mae,
                **config
            }
            results.append(result)

            if mse < best_mse:
                best_mse = mse
                best_config = config
                print(f"  *** New best! ***")
        else:
            print(f"  Failed: {output[:100]}...")

    # 保存结果
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'=' * 60}")
        print(f"Search completed! Results saved to: {args.output}")
        print(f"\nBest configuration (MSE={best_mse:.6f}):")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo successful experiments!")


if __name__ == '__main__':
    main()
