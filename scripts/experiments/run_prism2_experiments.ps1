# PRISM2 完整实验脚本
# 测试 TCN/PatchTST/iTransformer 在 ETTh2/ETTm1/Weather 数据集上的表现
# 预测步长: 24, 48, 96

$ErrorActionPreference = "Continue"

# 设置工作目录
Set-Location "C:\Users\Litsay\Desktop\PRISM2"

# 实验配置
$models = @("TCN", "PatchTST", "iTransformer")
$datasets = @("ETTh2", "ETTm1", "Weather")
$pred_lens = @(24, 48, 96)

# PRISM2 参数
$prism2_params = @{
    "window_size" = 128
    "theta_epsilon" = 1.0
    "theta_H" = 1.5
    "warmup_steps" = 50
    "precursor_lr" = 0.0005
    "drift_lr" = 0.0001
    "drift_steps" = 3
}

# 结果文件
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$results_file = "results/prism2_experiments_$timestamp.csv"
$log_file = "results/prism2_experiments_$timestamp.log"

# 创建results目录
if (!(Test-Path "results")) {
    New-Item -ItemType Directory -Path "results"
}

# 写入CSV头
"Model,Dataset,PredLen,MSE,MAE,Stable_Ratio,Precursor_Ratio,Drift_Ratio,Avg_Divergence" | Out-File $results_file -Encoding UTF8

Write-Host "=========================================="
Write-Host "PRISM2 Experiments Starting"
Write-Host "Models: $($models -join ', ')"
Write-Host "Datasets: $($datasets -join ', ')"
Write-Host "Pred Lengths: $($pred_lens -join ', ')"
Write-Host "Results: $results_file"
Write-Host "=========================================="

foreach ($model in $models) {
    foreach ($dataset in $datasets) {
        foreach ($pred_len in $pred_lens) {
            Write-Host ""
            Write-Host "============================================"
            Write-Host "Running: $model on $dataset with pred_len=$pred_len"
            Write-Host "============================================"
            
            # 步骤1: 预训练backbone (如果checkpoint不存在)
            Write-Host "[Step 1] Checking/Training backbone..."
            $pretrain_cmd = @(
                "python", "run.py",
                "--model", $model,
                "--dataset", $dataset,
                "--features", "M",
                "--seq_len", "96",
                "--pred_len", $pred_len,
                "--learning_rate", "0.0001",
                "--train_epochs", "25",
                "--patience", "3",
                "--batch_size", "32",
                "--itr", "1"
            )
            
            $pretrain_result = & $pretrain_cmd[0] $pretrain_cmd[1..($pretrain_cmd.Length-1)] 2>&1
            $pretrain_result | Out-File $log_file -Append -Encoding UTF8
            
            # 步骤2: 运行PRISM2在线测试
            Write-Host "[Step 2] Running PRISM2 online test..."
            $prism2_cmd = @(
                "python", "run.py",
                "--model", $model,
                "--dataset", $dataset,
                "--features", "M",
                "--seq_len", "96",
                "--pred_len", $pred_len,
                "--online_method", "PRISM2",
                "--pretrain",
                "--only_test",
                "--wo_valid",
                "--learning_rate", "0.0001",
                "--online_learning_rate", "0.0001",
                "--prism2_window_size", $prism2_params["window_size"],
                "--prism2_theta_epsilon", $prism2_params["theta_epsilon"],
                "--prism2_theta_H", $prism2_params["theta_H"],
                "--prism2_warmup_steps", $prism2_params["warmup_steps"],
                "--prism2_precursor_lr", $prism2_params["precursor_lr"],
                "--prism2_drift_lr", $prism2_params["drift_lr"],
                "--prism2_drift_steps", $prism2_params["drift_steps"],
                "--itr", "1"
            )
            
            $output = & $prism2_cmd[0] $prism2_cmd[1..($prism2_cmd.Length-1)] 2>&1
            $output_str = $output -join "`n"
            $output_str | Out-File $log_file -Append -Encoding UTF8
            
            # 解析结果
            $mse = "N/A"
            $mae = "N/A"
            $stable_ratio = "N/A"
            $precursor_ratio = "N/A"
            $drift_ratio = "N/A"
            $avg_divergence = "N/A"
            
            # 提取MSE和MAE
            if ($output_str -match "mse:([0-9.]+), mae:([0-9.]+)") {
                $mse = $matches[1]
                $mae = $matches[2]
            }
            
            # 提取状态分布
            if ($output_str -match "STABLE: \d+ \(([0-9.]+)%\)") {
                $stable_ratio = $matches[1]
            }
            if ($output_str -match "PRECURSOR: \d+ \(([0-9.]+)%\)") {
                $precursor_ratio = $matches[1]
            }
            if ($output_str -match "DRIFT: \d+ \(([0-9.]+)%\)") {
                $drift_ratio = $matches[1]
            }
            if ($output_str -match "Avg Divergence: ([0-9.]+)") {
                $avg_divergence = $matches[1]
            }
            
            # 写入结果
            "$model,$dataset,$pred_len,$mse,$mae,$stable_ratio,$precursor_ratio,$drift_ratio,$avg_divergence" | Out-File $results_file -Append -Encoding UTF8
            
            Write-Host "Result: MSE=$mse, MAE=$mae"
            Write-Host "States: Stable=$stable_ratio%, Precursor=$precursor_ratio%, Drift=$drift_ratio%"
            Write-Host ""
        }
    }
}

Write-Host "=========================================="
Write-Host "All experiments completed!"
Write-Host "Results saved to: $results_file"
Write-Host "Log saved to: $log_file"
Write-Host "=========================================="

# 显示结果汇总
Write-Host ""
Write-Host "Results Summary:"
Get-Content $results_file | Format-Table
