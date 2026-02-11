#!/bin/bash

# PRISM2 with iTransformer on ETTh2 Dataset
# 宏微尺度几何分歧感知的概念漂移适应框架

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
dataset=ETTh2

for pred_len in 96 192 336 720
do
    python -u run.py \
        --model $model_name \
        --dataset $dataset \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --online_method PRISM2 \
        --pretrain \
        --freeze \
        --learning_rate 0.0001 \
        --online_learning_rate 0.001 \
        --prism2_window_size 128 \
        --prism2_theta_epsilon 1.2 \
        --prism2_theta_H 2.0 \
        --prism2_warmup_steps 30 \
        --prism2_precursor_lr 0.01 \
        --prism2_drift_lr 0.001 \
        --prism2_drift_steps 5 \
        --itr 1
done
