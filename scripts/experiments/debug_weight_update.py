"""
Debug PRISM2 weight update mechanism
"""

import os
import sys
from pathlib import Path

WORK_DIR = Path(__file__).parent.parent.parent
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse

from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp.exp_prism2 import Exp_PRISM2
from settings import get_borders


def create_args():
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
        batch_size=1,
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

    # PRISM2 parameters - will be loaded from settings
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
    args.prism2_tau_R = None
    args.prism2_tau_H = None
    args.prism2_precursor_neighbors = None
    args.prism2_drift_neighbors = None
    args.prism2_weight_lambda = None

    return args


def debug_weight_update():
    """Detailed debug of weight update mechanism"""
    args = create_args()
    get_borders(args)

    # Create experiment
    exp = Exp_PRISM2(args)

    # Load pretrain weights
    pretrain_path = './checkpoints/ETTh2_96_24_TCN_online_ftM_sl96_ll48_pl24_lr0.003_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth'
    if os.path.exists(pretrain_path):
        exp.load_checkpoint(pretrain_path)
        print(f"Loaded pretrained model from {pretrain_path}")
    else:
        print(f"ERROR: Pretrained model not found at {pretrain_path}")
        return

    # Get test data
    wrap_data_kwargs = {'recent_num': 1, 'gap': args.pred_len, 'borders': args.borders}
    test_data = get_dataset(
        args, 'test', 'cuda',
        wrap_class=[Dataset_Recent],
        **wrap_data_kwargs
    )
    test_loader = get_dataloader(test_data, args, flag='online')

    # Get the PRISM2 model and its components
    model = exp._model
    backbone = model.backbone
    response_controller = model.response_controller

    print(f"\n=== PRISM2 Config ===")
    print(f"window_size: {model.config.window_size}")
    print(f"warmup_steps: {model.config.warmup_steps}")
    print(f"theta_epsilon: {model.config.theta_epsilon}")
    print(f"theta_H: {model.config.theta_H}")
    print(f"precursor_lr: {model.config.precursor_lr}")
    print(f"drift_lr: {model.config.drift_lr}")
    print(f"precursor_neighbors: {model.config.precursor_neighbors}")
    print(f"drift_neighbors: {model.config.drift_neighbors}")

    # Reset state
    model.reset_state()
    model.flag_online_learning = True

    # Enable debug output in response controller
    response_controller._debug_updates = True

    # Track weight changes
    def get_weight_snapshot():
        """Get a snapshot of model weights"""
        return {name: param.data.clone() for name, param in backbone.named_parameters()}

    # Iterate through samples
    print(f"\n=== Debug Run ===")

    for i, (recent_data, current_data) in enumerate(test_loader):
        if i >= 30:  # Test first 30 samples
            break

        # Get weight snapshot before
        weights_before = get_weight_snapshot()

        # Process recent data (adds to sample pool)
        exp._process_recent_data(recent_data)

        # Get current sample pool size
        sample_pool_size = len(model.state.available_samples)

        batch_x = current_data[0].cuda()
        batch_y = current_data[1].cuda()
        batch_x_mark = current_data[2].cuda() if len(current_data) > 2 else None

        # Forward pass
        backbone.eval()
        with torch.no_grad():
            if batch_x_mark is not None:
                output = model(batch_x, batch_x_mark)
            else:
                output = model(batch_x)

        if isinstance(output, tuple):
            output = output[0]

        # Get weight snapshot after
        weights_after = get_weight_snapshot()

        # Calculate weight difference
        weight_diff = 0.0
        for name in weights_before:
            weight_diff += (weights_after[name] - weights_before[name]).abs().sum().item()

        # Get state
        current_state = model.get_current_state()

        # Calculate MSE
        mse = ((output - batch_y) ** 2).mean().item()

        # Debug output
        if i < 5 or (current_state != "WARMUP" and current_state != "STABLE") or weight_diff > 0:
            print(f"\nStep {i}:")
            print(f"  State: {current_state}")
            print(f"  Sample pool size: {sample_pool_size}")
            print(f"  Weight diff: {weight_diff:.8f}")
            print(f"  MSE: {mse:.4f}")

            # Additional debug for PRECURSOR/DRIFT
            if current_state in ["PRECURSOR", "DRIFT"]:
                print(f"  Response controller stats:")
                print(f"    _total_updates: {response_controller._total_updates}")
                print(f"    _total_steps: {response_controller._total_steps}")
                print(f"    _consecutive_updates: {response_controller._consecutive_updates}")
                print(f"    _update_cooldown: {response_controller._update_cooldown}")

    # Final statistics
    print(f"\n=== Final Stats ===")
    stats = model.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")


def debug_manual_update():
    """Manually test the update mechanism"""
    args = create_args()
    get_borders(args)

    # Create experiment
    exp = Exp_PRISM2(args)

    # Load pretrain weights
    pretrain_path = './checkpoints/ETTh2_96_24_TCN_online_ftM_sl96_ll48_pl24_lr0.003_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth'
    if os.path.exists(pretrain_path):
        exp.load_checkpoint(pretrain_path)
        print(f"Loaded pretrained model")
    else:
        print(f"ERROR: Pretrained model not found")
        return

    # Get test data
    wrap_data_kwargs = {'recent_num': 1, 'gap': args.pred_len, 'borders': args.borders}
    test_data = get_dataset(
        args, 'test', 'cuda',
        wrap_class=[Dataset_Recent],
        **wrap_data_kwargs
    )
    test_loader = get_dataloader(test_data, args, flag='online')

    model = exp._model
    backbone = model.backbone

    print("\n=== Manual Update Test ===")

    # Collect some samples first
    samples = []
    for i, (recent_data, current_data) in enumerate(test_loader):
        if i >= 10:
            break
        batch_x = current_data[0].cuda()
        batch_y = current_data[1].cuda()
        batch_x_mark = current_data[2].cuda() if len(current_data) > 2 else None

        if batch_x.dim() == 3:
            batch_x = batch_x[0]
            batch_y = batch_y[0]
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark[0]

        samples.append((batch_x, batch_y, batch_x_mark, 0.5))  # (X, Y, x_mark, epsilon)

    print(f"Collected {len(samples)} samples")

    # Get weight snapshot before
    def get_weight_sum():
        total = 0.0
        for param in backbone.parameters():
            total += param.data.abs().sum().item()
        return total

    weight_sum_before = get_weight_sum()
    print(f"Weight sum before: {weight_sum_before:.4f}")

    # Manually call _fast_weight_update
    print("\nCalling _fast_weight_update manually...")

    criterion = nn.MSELoss()
    weights = torch.ones(len(samples), device='cuda') / len(samples)

    # Check backbone requires_grad
    grad_enabled = False
    for param in backbone.parameters():
        if param.requires_grad:
            grad_enabled = True
            break
    print(f"Backbone requires_grad: {grad_enabled}")

    # Enable grad if needed
    if not grad_enabled:
        print("Enabling requires_grad on backbone...")
        for param in backbone.parameters():
            param.requires_grad_(True)

    # Call update function directly
    model.response_controller._fast_weight_update(backbone, samples, weights, criterion)

    weight_sum_after = get_weight_sum()
    print(f"Weight sum after: {weight_sum_after:.4f}")
    print(f"Difference: {abs(weight_sum_after - weight_sum_before):.8f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Part 1: Debug Weight Update Flow")
    print("=" * 60)
    debug_weight_update()

    print("\n" + "=" * 60)
    print("Part 2: Manual Update Test")
    print("=" * 60)
    debug_manual_update()
