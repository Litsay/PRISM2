"""
PRISM2 批量实验脚本

运行所有模型/数据集/预测长度组合，并与PROCEED基线比较
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import csv

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# PROCEED 基线数据
PROCEED_BASELINE = {
    'TCN': {
        'ETTh2': {'24': {'mse': 2.908, 'mae': 0.659}, '48': {'mse': 4.056, 'mae': 0.767}, '96': {'mse': 5.891, 'mae': 0.890}},
        'ETTm1': {'24': {'mse': 0.531, 'mae': 0.447}, '48': {'mse': 0.704, 'mae': 0.521}, '96': {'mse': 0.780, 'mae': 0.553}},
        'Weather': {'24': {'mse': 0.707, 'mae': 0.382}, '48': {'mse': 0.959, 'mae': 0.493}, '96': {'mse': 1.314, 'mae': 0.637}},
    },
    'PatchTST': {
        'ETTh2': {'24': {'mse': 1.735, 'mae': 0.579}, '48': {'mse': 3.114, 'mae': 0.692}, '96': {'mse': 5.555, 'mae': 0.849}},
        'ETTm1': {'24': {'mse': 0.424, 'mae': 0.392}, '48': {'mse': 0.577, 'mae': 0.463}, '96': {'mse': 0.660, 'mae': 0.505}},
        'Weather': {'24': {'mse': 0.724, 'mae': 0.367}, '48': {'mse': 0.973, 'mae': 0.477}, '96': {'mse': 1.261, 'mae': 0.591}},
    },
    'iTransformer': {
        'ETTh2': {'24': {'mse': 2.387, 'mae': 0.633}, '48': {'mse': 3.969, 'mae': 0.753}, '96': {'mse': 6.291, 'mae': 0.889}},
        'ETTm1': {'24': {'mse': 0.426, 'mae': 0.398}, '48': {'mse': 0.561, 'mae': 0.461}, '96': {'mse': 0.642, 'mae': 0.500}},
        'Weather': {'24': {'mse': 0.742, 'mae': 0.378}, '48': {'mse': 1.015, 'mae': 0.495}, '96': {'mse': 1.294, 'mae': 0.602}},
    },
}


def run_experiment(model, dataset, pred_len, method='PRISM2', gpu=0, timeout=1800):
    """运行单个实验"""
    cmd = [
        'python', 'run.py',
        '--only_test',
        '--pretrain',
        '--online_method', method,
        '--model', model,
        '--dataset', dataset,
        '--pred_len', str(pred_len),
        '--gpu', str(gpu),
        '--wo_valid',
        '--itr', '1',  # 只运行一次迭代
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root
        )

        output = result.stdout + result.stderr

        mse = None
        mae = None
        stable_ratio = None
        precursor_ratio = None
        drift_ratio = None

        # 解析MSE/MAE - 只取第一个匹配的结果（测试结果）
        for line in output.split('\n'):
            # 跳过包含字典格式的行
            if '{' in line and '}' in line:
                continue
            if 'mse:' in line.lower() and 'mae:' in line.lower() and mse is None:
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

            # 解析状态分布
            if 'STABLE:' in line and '%' in line:
                try:
                    stable_ratio = float(line.split('(')[1].split('%')[0])
                except:
                    pass
            if 'PRECURSOR:' in line and '%' in line:
                try:
                    precursor_ratio = float(line.split('(')[1].split('%')[0])
                except:
                    pass
            if 'DRIFT:' in line and '%' in line:
                try:
                    drift_ratio = float(line.split('(')[1].split('%')[0])
                except:
                    pass

        return {
            'mse': mse,
            'mae': mae,
            'stable_ratio': stable_ratio,
            'precursor_ratio': precursor_ratio,
            'drift_ratio': drift_ratio,
            'success': mse is not None
        }

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Run all PRISM2 experiments')
    parser.add_argument('--models', nargs='+', default=['TCN', 'PatchTST', 'iTransformer'])
    parser.add_argument('--datasets', nargs='+', default=['ETTh2', 'ETTm1', 'Weather'])
    parser.add_argument('--pred_lens', nargs='+', type=int, default=[24, 48, 96])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        args.output = f"results/prism2_optimized_{timestamp}.csv"

    os.makedirs('results', exist_ok=True)

    results = []
    total_experiments = len(args.models) * len(args.datasets) * len(args.pred_lens)
    current = 0

    # 统计
    better_count = 0
    worse_count = 0
    total_mse_improvement = 0
    total_mae_improvement = 0

    print("=" * 80)
    print("PRISM2 Optimized Experiments")
    print("=" * 80)

    for model in args.models:
        for dataset in args.datasets:
            for pred_len in args.pred_lens:
                current += 1
                print(f"\n[{current}/{total_experiments}] {model} | {dataset} | pred_len={pred_len}")

                # 运行PRISM2实验
                result = run_experiment(model, dataset, pred_len, 'PRISM2', args.gpu)

                if result['success']:
                    # 获取PROCEED基线
                    baseline = PROCEED_BASELINE.get(model, {}).get(dataset, {}).get(str(pred_len), {})
                    baseline_mse = baseline.get('mse', float('inf'))
                    baseline_mae = baseline.get('mae', float('inf'))

                    # 计算改进
                    mse_improvement = ((baseline_mse - result['mse']) / baseline_mse) * 100 if baseline_mse > 0 else 0
                    mae_improvement = ((baseline_mae - result['mae']) / baseline_mae) * 100 if baseline_mae > 0 else 0

                    # 统计
                    if result['mse'] < baseline_mse:
                        better_count += 1
                        status = "BETTER"
                    else:
                        worse_count += 1
                        status = "WORSE"

                    total_mse_improvement += mse_improvement
                    total_mae_improvement += mae_improvement

                    print(f"  PRISM2:  MSE={result['mse']:.4f}, MAE={result['mae']:.4f}")
                    print(f"  PROCEED: MSE={baseline_mse:.4f}, MAE={baseline_mae:.4f}")
                    print(f"  Improvement: MSE={mse_improvement:+.2f}%, MAE={mae_improvement:+.2f}% [{status}]")
                    if result.get('stable_ratio') is not None:
                        print(f"  States: STABLE={result['stable_ratio']:.1f}%, PRECURSOR={result['precursor_ratio']:.1f}%, DRIFT={result['drift_ratio']:.1f}%")

                    results.append({
                        'model': model,
                        'dataset': dataset,
                        'pred_len': pred_len,
                        'prism2_mse': result['mse'],
                        'prism2_mae': result['mae'],
                        'proceed_mse': baseline_mse,
                        'proceed_mae': baseline_mae,
                        'mse_improvement_pct': mse_improvement,
                        'mae_improvement_pct': mae_improvement,
                        'stable_ratio': result.get('stable_ratio'),
                        'precursor_ratio': result.get('precursor_ratio'),
                        'drift_ratio': result.get('drift_ratio'),
                        'status': status,
                    })
                else:
                    print(f"  Failed: {result.get('error', 'Unknown error')}")

    # 保存结果
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # 打印总结
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(results)}")
    print(f"Better than PROCEED: {better_count} ({better_count/len(results)*100:.1f}%)")
    print(f"Worse than PROCEED: {worse_count} ({worse_count/len(results)*100:.1f}%)")
    print(f"Average MSE improvement: {total_mse_improvement/len(results):+.2f}%")
    print(f"Average MAE improvement: {total_mae_improvement/len(results):+.2f}%")
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
