#!/bin/bash

# PRISM2 on ETTh2 Dataset (无需预训练，从头训练)
# 用于测试PRISM2框架是否正常工作

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST
dataset=ETTh2
pred_len=96

python -u run.py \
    --model $model_name \
    --dataset $dataset \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
    --online_method PRISM2 \
    --learning_rate 0.0001 \
    --online_learning_rate 0.001 \
    --prism2_window_size 128 \
    --prism2_theta_epsilon 1.2 \
    --prism2_theta_H 2.0 \
    --prism2_warmup_steps 30 \
    --train_epochs 10 \
    --itr 1
