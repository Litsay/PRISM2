"""
PRISM2 诊断脚本

比较Online基线和PRISM2的输出差异
"""

import os
import sys
from pathlib import Path

# 设置工作目录
WORK_DIR = Path(__file__).parent.parent.parent
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_online import Exp_Online
from exp.exp_prism2 import Exp_PRISM2
import argparse

def create_args():
    """创建测试参数"""
    args = argparse.Namespace(
        model='TCN',
        dataset='ETTh2',
        features='M',
        seq_len=96,
        pred_len=24,
        label_len=48,
        border_type='online',
        root_path='./dataset/',
        target='OT',
        freq='h',
        use_gpu=True,
        gpu=0,
        pin_gpu=True,
        batch_size=1,  # 用batch_size=1测试
        learning_rate=0.003,
        online_learning_rate=0.0001,
        checkpoints='./checkpoints/',
        freeze=False,
        pretrain=True,
        only_test=True,
        wo_valid=True,
        online_method='PRISM2',
        wrap_data_class=[],
        do_predict=False,
        use_amp=False,
        enc_in=7,
        c_out=7,
        dec_in=7,
        data='ETTh2',
        data_path='ETT-small/ETTh2.csv',
        des='test',
        train_only=False,
        timeenc=2,
        num_workers=0,
        embed='timeF',
        local_rank=-1,
        optim='AdamW',
        patience=3,
        train_epochs=25,
        begin_valid_epoch=0,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        use_multi_gpu=False,
        devices='0',
        factor=3,
        distil=True,
        activation='gelu',
        output_attention=False,
        output_enc=False,
        loss='mse',
        lradj='type3',
        pct_start=0.3,
        warmup_epochs=5,
        ensemble=False,
        save_opt=True,
        find_unused_parameters=False,
        normalization=None,
        compile=False,
    )

    # 添加PRISM2参数
    args.prism2_window_size = None
    args.prism2_theta_epsilon = None
    args.prism2_theta_H = None
    args.prism2_warmup_steps = None
    args.prism2_precursor_lr = None
    args.prism2_drift_lr = None
    args.prism2_drift_steps = None
    args.prism2_k_neighbors = None
    args.prism2_lid_k = None
    args.prism2_gamma = None
    args.prism2_ema_alpha = None

    return args

def test_direct_inference(args):
    """测试直接推理（不经过PRISM2框架）"""
    print("\n" + "="*60)
    print("Testing Direct Inference (without PRISM2)")
    print("="*60)

    from settings import get_borders
    get_borders(args)

    # 创建模型
    from models.TCN import Model
    model = Model(args).cuda()

    # 加载预训练权重
    pretrain_path = f'./checkpoints/ETTh2_96_24_TCN_online_ftM_sl96_ll48_pl24_lr0.003_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth'

    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {pretrain_path}")
    else:
        print(f"Pretrained model not found at {pretrain_path}")
        return

    model.eval()

    # 获取测试数据
    wrap_data_kwargs = {'recent_num': 1, 'gap': args.pred_len, 'borders': args.borders}
    test_data = get_dataset(
        args, 'test', 'cuda',
        wrap_class=[Dataset_Recent],
        **wrap_data_kwargs
    )
    test_loader = get_dataloader(test_data, args, flag='online')

    # 测试
    total_mse = 0
    total_mae = 0
    count = 0

    for i, (recent_data, current_data) in enumerate(tqdm(test_loader, desc="Direct Inference")):
        if i >= 100:  # 只测试前100个样本
            break

        batch_x = current_data[0].cuda()
        batch_y = current_data[1].cuda()
        batch_x_mark = current_data[2].cuda() if len(current_data) > 2 else None

        with torch.no_grad():
            output = model(batch_x, batch_x_mark)

        mse = ((output - batch_y) ** 2).mean().item()
        mae = (torch.abs(output - batch_y)).mean().item()

        total_mse += mse
        total_mae += mae
        count += 1

        if i < 3:
            print(f"\nSample {i}:")
            print(f"  batch_x shape: {batch_x.shape}, range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
            print(f"  batch_y shape: {batch_y.shape}, range: [{batch_y.min():.3f}, {batch_y.max():.3f}]")
            print(f"  output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")

    print(f"\nDirect Inference Results (first {count} samples):")
    print(f"  Avg MSE: {total_mse/count:.4f}")
    print(f"  Avg MAE: {total_mae/count:.4f}")

def test_prism2_inference(args):
    """测试PRISM2框架的推理"""
    print("\n" + "="*60)
    print("Testing PRISM2 Framework Inference")
    print("="*60)

    from settings import get_borders
    get_borders(args)

    # 创建PRISM2模型
    exp = Exp_PRISM2(args)

    # 加载预训练权重
    pretrain_path = f'./checkpoints/ETTh2_96_24_TCN_online_ftM_sl96_ll48_pl24_lr0.003_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth'

    if os.path.exists(pretrain_path):
        exp.load_checkpoint(pretrain_path)
        print(f"Loaded pretrained model from {pretrain_path}")
    else:
        print(f"Pretrained model not found at {pretrain_path}")
        return

    exp._model.backbone.eval()

    # 获取测试数据
    wrap_data_kwargs = {'recent_num': 1, 'gap': args.pred_len, 'borders': args.borders}
    test_data = get_dataset(
        args, 'test', 'cuda',
        wrap_class=[Dataset_Recent],
        **wrap_data_kwargs
    )
    test_loader = get_dataloader(test_data, args, flag='online')

    # 测试
    total_mse = 0
    total_mae = 0
    count = 0

    exp._model.flag_online_learning = False  # 禁用在线学习模式

    for i, (recent_data, current_data) in enumerate(tqdm(test_loader, desc="PRISM2 Inference")):
        if i >= 100:
            break

        batch_x = current_data[0].cuda()
        batch_y = current_data[1].cuda()
        batch_x_mark = current_data[2].cuda() if len(current_data) > 2 else None

        with torch.no_grad():
            if batch_x_mark is not None:
                output = exp._model(batch_x, batch_x_mark)
            else:
                output = exp._model(batch_x)

        if isinstance(output, tuple):
            output = output[0]

        mse = ((output - batch_y) ** 2).mean().item()
        mae = (torch.abs(output - batch_y)).mean().item()

        total_mse += mse
        total_mae += mae
        count += 1

        if i < 3:
            print(f"\nSample {i}:")
            print(f"  batch_x shape: {batch_x.shape}, range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
            print(f"  batch_y shape: {batch_y.shape}, range: [{batch_y.min():.3f}, {batch_y.max():.3f}]")
            print(f"  output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")

    print(f"\nPRISM2 Framework Results (first {count} samples):")
    print(f"  Avg MSE: {total_mse/count:.4f}")
    print(f"  Avg MAE: {total_mae/count:.4f}")

def test_prism2_online_mode(args):
    """测试PRISM2在线学习模式"""
    print("\n" + "="*60)
    print("Testing PRISM2 Online Learning Mode")
    print("="*60)

    from settings import get_borders
    get_borders(args)

    # 创建PRISM2模型
    exp = Exp_PRISM2(args)

    # 加载预训练权重
    pretrain_path = f'./checkpoints/ETTh2_96_24_TCN_online_ftM_sl96_ll48_pl24_lr0.003_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth'

    if os.path.exists(pretrain_path):
        exp.load_checkpoint(pretrain_path)
        print(f"Loaded pretrained model from {pretrain_path}")
    else:
        print(f"Pretrained model not found at {pretrain_path}")
        return

    # 获取测试数据
    wrap_data_kwargs = {'recent_num': 1, 'gap': args.pred_len, 'borders': args.borders}
    test_data = get_dataset(
        args, 'test', 'cuda',
        wrap_class=[Dataset_Recent],
        **wrap_data_kwargs
    )
    test_loader = get_dataloader(test_data, args, flag='online')

    # 测试
    total_mse = 0
    total_mae = 0
    count = 0

    exp._model.reset_state()
    exp._model.flag_online_learning = True  # 启用在线学习模式

    for i, (recent_data, current_data) in enumerate(tqdm(test_loader, desc="PRISM2 Online")):
        if i >= 100:
            break

        # 处理recent_data
        exp._process_recent_data(recent_data)

        batch_x = current_data[0].cuda()
        batch_y = current_data[1].cuda()
        batch_x_mark = current_data[2].cuda() if len(current_data) > 2 else None

        exp._model.backbone.eval()
        with torch.no_grad():
            if batch_x_mark is not None:
                output = exp._model(batch_x, batch_x_mark)
            else:
                output = exp._model(batch_x)

        if isinstance(output, tuple):
            output = output[0]

        mse = ((output - batch_y) ** 2).mean().item()
        mae = (torch.abs(output - batch_y)).mean().item()

        total_mse += mse
        total_mae += mae
        count += 1

        if i < 3:
            print(f"\nSample {i}:")
            print(f"  batch_x shape: {batch_x.shape}, range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
            print(f"  batch_y shape: {batch_y.shape}, range: [{batch_y.min():.3f}, {batch_y.max():.3f}]")
            print(f"  output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")
            print(f"  State: {exp._model.get_current_state()}")

    print(f"\nPRISM2 Online Mode Results (first {count} samples):")
    print(f"  Avg MSE: {total_mse/count:.4f}")
    print(f"  Avg MAE: {total_mae/count:.4f}")

def main():
    args = create_args()

    test_direct_inference(args)
    test_prism2_inference(args)
    test_prism2_online_mode(args)

if __name__ == "__main__":
    main()
