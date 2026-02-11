#!/bin/bash

# Pretrain PatchTST on ETTh2 Dataset
# 预训练PatchTST模型

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST
dataset=ETTh2

for pred_len in 96 192 336 720
do
    python -u run.py \
        --model $model_name \
        --dataset $dataset \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --learning_rate 0.0001 \
        --train_epochs 25 \
        --patience 3 \
        --batch_size 32 \
        --itr 1
done
