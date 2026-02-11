"""
PRISM2 完整实验脚本

测试配置:
- Backbone: TCN_RevIN, PatchTST, iTransformer
- 数据集: ETTh2, ETTm1, Weather
- 预测步长: 24, 48, 96

输出:
- MSE, MAE 指标
- 概念漂移预测状态分布 (STABLE/PRECURSOR/DRIFT)
"""

import os
import sys
import subprocess
import re
import csv
from datetime import datetime
from pathlib import Path

# 设置工作目录
WORK_DIR = Path(__file__).parent.parent.parent
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

# 导入settings中的学习率配置和PRISM2配置
from settings import pretrain_lr_online_dict, pretrain_lr_dict, get_prism2_config

def get_pretrain_lr(model, dataset):
    """获取预训练学习率，与Online测试时的学习率保持一致"""
    if model in pretrain_lr_online_dict and dataset in pretrain_lr_online_dict[model]:
        return pretrain_lr_online_dict[model][dataset]
    elif model in pretrain_lr_dict and dataset in pretrain_lr_dict[model]:
        return pretrain_lr_dict[model][dataset]
    else:
        return 0.0001  # 默认学习率

# ============== 实验配置 ==============
MODELS = ["TCN", "PatchTST", "iTransformer"]
DATASETS = ["ETTh2", "ETTm1", "Weather"]
PRED_LENS = [24, 48, 96]

def get_prism2_experiment_config(model, dataset):
    """
    获取针对特定模型和数据集组合的PRISM2配置

    使用settings.py中的优化配置系统
    """
    config = get_prism2_config(dataset, model)

    # 返回实验脚本需要的格式
    return {
        "window_size": config.get('window_size', 256),
        "theta_epsilon": config.get('theta_epsilon', 2.0),
        "theta_H": config.get('theta_H', 2.5),
        "warmup_steps": config.get('warmup_steps', 100),
        "precursor_lr": config.get('precursor_lr', 0.0005),
        "drift_lr": config.get('drift_lr', 0.0001),
        "drift_steps": config.get('drift_steps', 2),
        "k_neighbors": config.get('k_neighbors', 15),
        "lid_k": config.get('lid_k', 20),
        "gamma": config.get('gamma', 0.3),
        "ema_alpha": config.get('ema_alpha', 0.05),
    }

# 默认配置（将被特定配置覆盖）
PRISM2_CONFIG = {
    "window_size": 256,
    "theta_epsilon": 2.0,
    "theta_H": 2.5,
    "warmup_steps": 100,
    "precursor_lr": 0.0005,
    "drift_lr": 0.0001,
    "drift_steps": 2,
    "k_neighbors": 15,
    "lid_k": 20,
}

# ============== 辅助函数 ==============

def run_command(cmd, timeout=3600):
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

def parse_results(output):
    """解析实验输出，提取指标"""
    results = {
        "mse": "N/A",
        "mae": "N/A",
        "stable_ratio": "N/A",
        "precursor_ratio": "N/A", 
        "drift_ratio": "N/A",
        "avg_divergence": "N/A",
        "warmup_count": "N/A",
    }
    
    # 提取 MSE 和 MAE
    mse_mae_match = re.search(r"mse:([0-9.]+), mae:([0-9.]+)", output)
    if mse_mae_match:
        results["mse"] = mse_mae_match.group(1)
        results["mae"] = mse_mae_match.group(2)
    
    # 提取状态分布
    stable_match = re.search(r"STABLE: (\d+) \(([0-9.]+)%\)", output)
    if stable_match:
        results["stable_ratio"] = stable_match.group(2)
    
    precursor_match = re.search(r"PRECURSOR: (\d+) \(([0-9.]+)%\)", output)
    if precursor_match:
        results["precursor_ratio"] = precursor_match.group(2)
    
    drift_match = re.search(r"DRIFT: (\d+) \(([0-9.]+)%\)", output)
    if drift_match:
        results["drift_ratio"] = drift_match.group(2)
    
    # 提取平均分歧度
    divergence_match = re.search(r"Avg Divergence: ([0-9.]+)", output)
    if divergence_match:
        results["avg_divergence"] = divergence_match.group(1)
    
    # 提取WARMUP数量
    warmup_match = re.search(r"WARMUP: (\d+)", output)
    if warmup_match:
        results["warmup_count"] = warmup_match.group(1)
    
    return results

def pretrain_backbone(model, dataset, pred_len):
    """预训练backbone模型"""
    lr = get_pretrain_lr(model, dataset)
    cmd = [
        "python", "run.py",
        "--model", model,
        "--dataset", dataset,
        "--features", "M",
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--learning_rate", str(lr),
        "--train_epochs", "25",
        "--patience", "3",
        "--batch_size", "32",
        "--itr", "1"
    ]
    return run_command(cmd)

def run_prism2_test(model, dataset, pred_len, config=None):
    """运行PRISM2在线测试"""
    # 如果没有提供config，使用针对模型和数据集的优化配置
    if config is None:
        config = get_prism2_experiment_config(model, dataset)

    lr = get_pretrain_lr(model, dataset)
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
        "--prism2_window_size", str(config["window_size"]),
        "--prism2_theta_epsilon", str(config["theta_epsilon"]),
        "--prism2_theta_H", str(config["theta_H"]),
        "--prism2_warmup_steps", str(config["warmup_steps"]),
        "--prism2_precursor_lr", str(config["precursor_lr"]),
        "--prism2_drift_lr", str(config["drift_lr"]),
        "--prism2_drift_steps", str(config["drift_steps"]),
        "--prism2_k_neighbors", str(config["k_neighbors"]),
        "--prism2_lid_k", str(config["lid_k"]),
        "--itr", "1"
    ]
    return run_command(cmd)

def run_online_baseline(model, dataset, pred_len):
    """运行Online基准测试（无适应）"""
    lr = get_pretrain_lr(model, dataset)
    cmd = [
        "python", "run.py",
        "--model", model,
        "--dataset", dataset,
        "--features", "M",
        "--seq_len", "96",
        "--pred_len", str(pred_len),
        "--online_method", "Online",
        "--pretrain",
        "--only_test",
        "--learning_rate", str(lr),
        "--online_learning_rate", "0.0001",
        "--itr", "1"
    ]
    return run_command(cmd)

# ============== 主函数 ==============

def main():
    # 创建结果目录
    results_dir = WORK_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 结果文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"prism2_experiments_{timestamp}.csv"
    log_file = results_dir / f"prism2_experiments_{timestamp}.log"
    
    print("=" * 60)
    print("PRISM2 Experiments")
    print("=" * 60)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Pred Lengths: {PRED_LENS}")
    print(f"Results: {results_file}")
    print("=" * 60)
    
    # 初始化结果
    all_results = []
    
    # CSV头
    headers = [
        "Model", "Dataset", "PredLen", 
        "PRISM2_MSE", "PRISM2_MAE",
        "Online_MSE", "Online_MAE",
        "Improvement_MSE%",
        "Stable%", "Precursor%", "Drift%", 
        "Avg_Divergence"
    ]
    
    total_experiments = len(MODELS) * len(DATASETS) * len(PRED_LENS)
    current = 0
    
    for model in MODELS:
        for dataset in DATASETS:
            for pred_len in PRED_LENS:
                current += 1
                print(f"\n[{current}/{total_experiments}] {model} | {dataset} | pred_len={pred_len}")
                print("-" * 50)
                
                # 步骤1: 预训练backbone
                print("[1/3] Pretraining backbone...")
                pretrain_output = pretrain_backbone(model, dataset, pred_len)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Pretrain: {model} | {dataset} | pred_len={pred_len}\n")
                    f.write(pretrain_output)
                
                # 步骤2: 运行Online基准
                print("[2/3] Running Online baseline...")
                online_output = run_online_baseline(model, dataset, pred_len)
                online_results = parse_results(online_output)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Online: {model} | {dataset} | pred_len={pred_len}\n")
                    f.write(online_output)
                
                # 步骤3: 运行PRISM2测试（使用针对模型和数据集的优化配置）
                print("[3/3] Running PRISM2 test...")
                prism2_config = get_prism2_experiment_config(model, dataset)
                print(f"     Config: theta_eps={prism2_config['theta_epsilon']}, "
                      f"theta_H={prism2_config['theta_H']}, "
                      f"precursor_lr={prism2_config['precursor_lr']}")
                prism2_output = run_prism2_test(model, dataset, pred_len, prism2_config)
                prism2_results = parse_results(prism2_output)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"PRISM2: {model} | {dataset} | pred_len={pred_len}\n")
                    f.write(prism2_output)
                
                # 计算MSE改进
                try:
                    online_mse = float(online_results["mse"])
                    prism2_mse = float(prism2_results["mse"])
                    improvement = (online_mse - prism2_mse) / online_mse * 100
                    improvement_str = f"{improvement:.2f}"
                except:
                    improvement_str = "N/A"
                
                # 保存结果
                row = {
                    "Model": model,
                    "Dataset": dataset,
                    "PredLen": pred_len,
                    "PRISM2_MSE": prism2_results["mse"],
                    "PRISM2_MAE": prism2_results["mae"],
                    "Online_MSE": online_results["mse"],
                    "Online_MAE": online_results["mae"],
                    "Improvement_MSE%": improvement_str,
                    "Stable%": prism2_results["stable_ratio"],
                    "Precursor%": prism2_results["precursor_ratio"],
                    "Drift%": prism2_results["drift_ratio"],
                    "Avg_Divergence": prism2_results["avg_divergence"],
                }
                all_results.append(row)
                
                # 打印当前结果
                print(f"  Online:  MSE={online_results['mse']}, MAE={online_results['mae']}")
                print(f"  PRISM2:  MSE={prism2_results['mse']}, MAE={prism2_results['mae']}")
                print(f"  Improvement: {improvement_str}%")
                print(f"  States: Stable={prism2_results['stable_ratio']}%, "
                      f"Precursor={prism2_results['precursor_ratio']}%, "
                      f"Drift={prism2_results['drift_ratio']}%")
    
    # 写入CSV
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_results)
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    # 表头
    header_fmt = "{:<12} {:<10} {:>8} {:>12} {:>12} {:>12} {:>10}"
    print(header_fmt.format("Model", "Dataset", "PredLen", "PRISM2_MSE", "Online_MSE", "Improv%", "Drift%"))
    print("-" * 80)
    
    for r in all_results:
        print(header_fmt.format(
            r["Model"][:12], 
            r["Dataset"][:10], 
            r["PredLen"],
            r["PRISM2_MSE"][:12] if isinstance(r["PRISM2_MSE"], str) else f"{float(r['PRISM2_MSE']):.4f}",
            r["Online_MSE"][:12] if isinstance(r["Online_MSE"], str) else f"{float(r['Online_MSE']):.4f}",
            r["Improvement_MSE%"],
            r["Drift%"]
        ))
    
    print("=" * 80)
    print(f"\nResults saved to: {results_file}")
    print(f"Log saved to: {log_file}")
    
    return all_results

if __name__ == "__main__":
    main()
