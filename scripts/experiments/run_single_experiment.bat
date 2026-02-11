@echo off
REM PRISM2 单次实验脚本
REM 用法: run_single_experiment.bat [model] [dataset] [pred_len]
REM 示例: run_single_experiment.bat PatchTST ETTh2 96

setlocal enabledelayedexpansion

cd /d C:\Users\Litsay\Desktop\PRISM2

set MODEL=%1
set DATASET=%2
set PRED_LEN=%3

if "%MODEL%"=="" set MODEL=PatchTST
if "%DATASET%"=="" set DATASET=ETTh2
if "%PRED_LEN%"=="" set PRED_LEN=96

echo ==========================================
echo PRISM2 Single Experiment
echo Model: %MODEL%
echo Dataset: %DATASET%
echo Pred Length: %PRED_LEN%
echo ==========================================

echo.
echo [Step 1] Pretraining backbone...
python run.py --model %MODEL% --dataset %DATASET% --features M --seq_len 96 --pred_len %PRED_LEN% --learning_rate 0.0001 --train_epochs 25 --patience 3 --batch_size 32 --itr 1

echo.
echo [Step 2] Running Online baseline...
python run.py --model %MODEL% --dataset %DATASET% --features M --seq_len 96 --pred_len %PRED_LEN% --online_method Online --pretrain --only_test --learning_rate 0.0001 --online_learning_rate 0.0001 --itr 1

echo.
echo [Step 3] Running PRISM2 test...
python run.py --model %MODEL% --dataset %DATASET% --features M --seq_len 96 --pred_len %PRED_LEN% --online_method PRISM2 --pretrain --only_test --wo_valid --learning_rate 0.0001 --online_learning_rate 0.0001 --prism2_window_size 128 --prism2_theta_epsilon 1.0 --prism2_theta_H 1.5 --prism2_warmup_steps 50 --prism2_precursor_lr 0.0005 --prism2_drift_lr 0.0001 --prism2_drift_steps 3 --itr 1

echo.
echo ==========================================
echo Experiment completed!
echo ==========================================

endlocal
