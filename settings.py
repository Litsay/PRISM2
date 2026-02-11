
need_x_y_mark = ['Autoformer', 'Transformer', 'Informer']
need_x_mark = ['TCN', 'FSNet', 'OneNet', 'iTransformer']
need_x_mark += [name + '_Ensemble' for name in need_x_mark]
no_extra_param = ['Online', 'ER', 'DERpp']
peft_methods = ['lora', 'adapter', 'ssf', 'mam_adapter', 'basic_tuning']
# PRISM2不在peft_methods中，因为它支持从头训练

# PRISM2 默认超参数配置（优化版本 - 平衡稳定性和适应性）
prism2_default_config = {
    'window_size': 128,           # 中等窗口大小
    'k_neighbors': 15,
    'lid_k': 20,
    'gamma': 0.5,                 # 适中的增强系数
    'tau_R': 2.0,                 # 微尺度异常阈值
    'tau_H': 1.5,                 # 宏尺度稳定阈值
    'ema_alpha': 0.1,             # 适中的EMA系数
    'theta_epsilon': 1.5,         # 适中的分歧阈值
    'theta_H': 2.0,               # DRIFT触发阈值
    'precursor_lr': 0.0003,       # 学习率与Online基线接近
    'precursor_neighbors': 16,    # 减少邻居数量提高效率
    'drift_lr': 0.0002,           # DRIFT阶段学习率
    'drift_neighbors': 32,        # DRIFT阶段邻居数量
    'drift_steps': 3,             # DRIFT更新步数
    'weight_lambda': 0.8,         # 流形感知权重系数
    'warmup_steps': 50,           # 预热步数
    'stable_update_interval': 5,  # STABLE每5步更新一次
    'stable_lr_scale': 0.3,       # STABLE状态学习率缩放
}

# PRISM2 模型特定学习率配置（基于Online基线学习率优化）
prism2_model_lr_config = {
    'TCN': {
        'precursor_lr': 0.0002,    # TCN：接近Online基线的0.0001，略高以支持更快适应
        'drift_lr': 0.0003,        # DRIFT需要更强的适应
        'drift_steps': 3,
        'stable_update_interval': 3,  # TCN响应较慢，更频繁更新
    },
    'PatchTST': {
        'precursor_lr': 0.0001,    # PatchTST：与Online基线一致
        'drift_lr': 0.0002,
        'drift_steps': 2,
        'stable_update_interval': 5,
    },
    'iTransformer': {
        'precursor_lr': 0.0001,    # iTransformer：与Online基线一致
        'drift_lr': 0.0002,
        'drift_steps': 2,
        'stable_update_interval': 5,
    },
}

# PRISM2 数据集特定配置（优化版本 - 针对每个数据集调优）
prism2_dataset_config = {
    'ETTh1': {
        'window_size': 128,
        'warmup_steps': 50,
        'theta_epsilon': 1.8,
        'theta_H': 2.2,
        'ema_alpha': 0.08,
        'stable_update_interval': 5,
    },
    'ETTh2': {
        # ETTh2优化配置 - 基于实验验证 (MSE=2.936 for TCN)
        'window_size': 128,
        'warmup_steps': 50,
        'theta_epsilon': 1.0,
        'theta_H': 1.5,
        'ema_alpha': 0.1,
        'gamma': 0.5,
        'tau_R': 1.5,
        'tau_H': 1.2,
        # 学习率配置 - 这些值在测试中表现最佳
        'precursor_lr': 0.001,       # TCN需要较高学习率
        'drift_lr': 0.002,
        'drift_steps': 2,
        'precursor_neighbors': 8,
        'drift_neighbors': 16,
        'stable_update_interval': 1,
        'stable_lr_scale': 0.5,
    },
    'ETTm1': {
        # ETTm1配置 - 基于实验验证，保持默认theta以支持适度更新
        'window_size': 128,
        'warmup_steps': 50,
        'theta_epsilon': 1.0,        # 保持敏感度支持更新
        'theta_H': 1.5,
        'ema_alpha': 0.1,
        'precursor_lr': 0.0003,      # 适中学习率 (Online baseline TCN=0.001)
        'drift_lr': 0.0005,
        'drift_steps': 2,
        'precursor_neighbors': 16,
        'drift_neighbors': 32,
        'stable_update_interval': 5,  # 减少STABLE更新频率
        'stable_lr_scale': 0.3,
    },
    'ETTm2': {
        'window_size': 128,
        'warmup_steps': 50,
        'theta_epsilon': 1.5,
        'theta_H': 2.0,
        'ema_alpha': 0.1,
        'stable_update_interval': 5,
    },
    'Weather': {
        # Weather配置 - 基于实验验证
        'window_size': 128,
        'warmup_steps': 50,
        'theta_epsilon': 1.0,        # 保持检测敏感度
        'theta_H': 1.5,
        'ema_alpha': 0.1,
        'precursor_lr': 0.0001,      # Weather Online基线lr较低
        'drift_lr': 0.0002,
        'drift_steps': 2,
        'precursor_neighbors': 16,
        'drift_neighbors': 32,
        'stable_update_interval': 5,  # 减少更新频率
        'stable_lr_scale': 0.3,
    },
    'ECL': {
        'window_size': 128,
        'warmup_steps': 50,
        'theta_epsilon': 1.5,
        'theta_H': 2.0,
        'stable_update_interval': 5,
    },
    'Traffic': {
        'window_size': 128,
        'warmup_steps': 50,
        'theta_epsilon': 1.8,
        'theta_H': 2.2,
        'stable_update_interval': 5,
    },
}

def get_prism2_config(dataset, model=None):
    """
    获取PRISM2配置（支持数据集和模型特定配置）

    优先级: 数据集配置 > 模型配置 > 默认配置

    Args:
        dataset: 数据集名称
        model: 模型名称（可选）

    Returns:
        config: 合并后的配置字典
    """
    config = prism2_default_config.copy()

    # 先应用模型特定配置
    if model is not None:
        model_config = None
        if model in prism2_model_lr_config:
            model_config = prism2_model_lr_config[model]
        else:
            for model_key in prism2_model_lr_config:
                if model.startswith(model_key):
                    model_config = prism2_model_lr_config[model_key]
                    break
        if model_config:
            config.update(model_config)

    # 再应用数据集特定配置（数据集配置优先级更高，会覆盖模型配置）
    if dataset in prism2_dataset_config:
        config.update(prism2_dataset_config[dataset])

    return config

data_settings = {
    'wind_N2': {'data': 'wind_N2.csv', 'T':'FR51', 'M':[254, 254], 'prefetch_batch_size': 16},
    'wind': {'data': 'wind.csv', 'T':'UK', 'M':[28,28], 'prefetch_batch_size': 64},
    'ECL':{'data':'electricity.csv','T':'OT','M':[321,321],'S':[1,1],'MS':[321,1], 'prefetch_batch_size': 10},
    'ETTh1':{'data':'ETT-small/ETTh1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTh2':{'data':'ETT-small/ETTh2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm1':{'data':'ETT-small/ETTm1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm2':{'data':'ETT-small/ETTm2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'Solar':{'data':'solar_AL.txt','T': 136,'M':[137,137],'S':[1,1],'MS':[137,1], 'prefetch_batch_size': 32},
    'Weather':{'data':'weather/weather.csv','T':'OT','M':[21,21],'S':[1,1],'MS':[21,1], 'prefetch_batch_size': 64},
    'WTH':{'data':'WTH.csv','T':'OT','M':[12,12],'S':[1,1],'MS':[12,1], 'prefetch_batch_size': 64},
    'Traffic': {'data': 'traffic.csv', 'T':'OT', 'M':[862,862], 'prefetch_batch_size': 2},
    'METR_LA': {'data':'metr-la.csv','T': '773869','M':[207,207],'S':[1,1],'MS':[207,1], 'prefetch_batch_size': 16},
    'PEMS_BAY': {'data':'pems-bay.csv','T': 400001,'M':[325,325],'S':[1,1],'MS':[325,1], 'prefetch_batch_size': 10},
    'NYC_BIKE': {'data':'nyc-bike.h5','T': 0,'M':[500,500],'S':[1,1],'MS':[500,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'NYC_TAXI': {'data':'nyc-taxi.h5','T': 0,'M':[532,532],'S':[1,1],'MS':[532,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'PeMSD4': {'data':'PeMSD4/PeMSD4.npz','T': 0,'M':[921,921],'S':[1,1],'MS':[921,1], 'prefetch_batch_size': 2, 'feat_dim': 3},
    'PeMSD8': {'data':'PeMSD8/PeMSD8.npz','T': 0,'M':[510,510],'S':[1,1],'MS':[510,1], 'prefetch_batch_size': 6, 'feat_dim': 3},
    'Exchange': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'exchange_rate': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'Illness': {'data': 'illness.csv', 'T':'OT', 'M':[7,7], 'prefetch_batch_size': 128},
}

def get_borders(args):
    if args.border_type == 'online':
        if args.data.startswith('ETTh'):
            border1s = [0, 4*30*24 - args.seq_len, 5*30*24 - args.seq_len]
            border2s = [4*30*24, 5*30*24, 20*30*24]
            args.borders = (border1s, border2s)
        elif args.data.startswith('ETTm'):
            border1s = [0, 4*30*24*4 - args.seq_len, 5*30*24*4 - args.seq_len]
            border2s = [4*30*24*4, 5*30*24*4, 20*30*24*4]
            args.borders = (border1s, border2s)
        else:
            args.ratio = (0.2, 0.75)

hyperparams = {
    'PatchTST': {'e_layers': 3},
    'MTGNN': {},
    'LightCTS': {},
    'Crossformer': {'lradj': 'Crossformer', 'e_layers': 3, 'seg_len': 24, 'd_ff': 512, 'd_model': 256, 'n_heads': 4, 'dropout': 0.2},
    'DLinear': {},
    'GPT4TS': {'e_layers': 3, 'd_model': 768, 'n_heads': 4, 'd_ff': 768, 'dropout': 0.3},
    'iTransformer': {'e_layers': 3, 'd_model': 512, 'd_ff': 512, 'activation': 'gelu', 'timeenc': 1, 'patience': 3, 'train_epochs': 10, },
    'Autoformer': {'train_epochs': 10, 'timeenc': 1},
    'Informer': {'train_epochs': 10, 'timeenc': 1},
}

def get_hyperparams(data, model, args, reduce_bs=True):
    hyperparam: dict = hyperparams[model]
    if model == 'iTransformer':
        if data == 'Traffic':
            hyperparam['e_layers'] = 4
        elif 'ETT' in data:
            hyperparam['e_layers'] = 2
            if data == 'ETTh1':
                hyperparam['d_model'] = 256
                hyperparam['d_ff'] = 256
            else:
                hyperparam['d_model'] = 128
                hyperparam['d_ff'] = 128

    if model == 'PatchTST':
        if args.lradj != 'type3':
            if data in ['ETTh1', 'ETTh2', 'Weather', 'Exchange', 'wind']:
                hyperparam['lradj'] = 'type3'
            elif data in ['Illness']:
                hyperparam['lradj'] = 'constant'
            else:
                hyperparam['lradj'] = 'TST'
        if data in ['ETTh1', 'ETTh2', 'Illness']:
            hyperparam.update(**{'dropout': 0.3, 'fc_dropout': 0.3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128})
        elif data in ['ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 128, 'd_ff': 256})
        else:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 64, 'd_ff': 128})

    elif model == 'Crossformer':
        if data == 'ECL' or args.lradj == 'fixed':
            hyperparam['lradj'] = 'fixed'
        if reduce_bs:
            if data in ['PeMSD4']:
                hyperparam['batch_size'] = 4
            elif data in ['Traffic']:
                hyperparam['batch_size'] = 4
            elif data in ['NYC_BIKE', 'NYC_TAXI', 'PeMSD8']:
                hyperparam['batch_size'] = 8
        else:
            if data in ['Traffic', 'PeMSD4'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 24
            if data in ['PeMSD8'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 16

        if data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Illness', 'wind', 'Exchange']:
            hyperparam['d_model'] = 256
            hyperparam['n_heads'] = 4
        else:
            hyperparam['d_model'] = 64
            hyperparam['n_heads'] = 2

        if data in ['Traffic', 'ECL']:
            hyperparam['d_ff'] = 128

        if data in ['Illness']:
            hyperparam['e_layers'] = 2

    elif model == 'GPT4TS':
        if data == 'ETTh1':
            hyperparam['lradj'] = 'typy4'
            hyperparam['tmax'] = 20
            # hyperparam['label_len'] = 168
        elif data == 'ETTh2':
            hyperparam['dropout'] = 1
            hyperparam['tmax'] = 20
            # hyperparam['label_len'] = 168
        elif data == 'Traffic':
            hyperparam['dropout'] = 0.3
        elif data == 'ECL':
            hyperparam['tmax'] = 10
        elif data == 'Illness':
            hyperparam['patch_size'] = 24
            # hyperparam['label_len'] = 18
            hyperparam['batch_size'] = 16

        if data in ['ETTm1', 'ETTm2', 'ECL', 'Traffic', 'Weather', 'WTH']:
            hyperparam['seq_len'] = 512

        if data.startswith('ETTm'):
            hyperparam['stride'] = 16
        elif args.seq_len == 104:
            hyperparam['stride'] = 2

    return hyperparam


pretrain_lr_online_dict = {
     'TCN': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.001, 'Weather': 0.001, 'Traffic': 0.003},
     'TCN_RevIN': {'ECL': 0.003, 'ETTh2': 0.001, 'ETTm1': 0.0001, 'Weather': 0.001, 'Traffic': 0.003},
     'TCN_Ensemble': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.0003, 'Weather': 0.001, 'Traffic': 0.003},
     'FSNet_RevIN': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.001, 'Weather': 0.003, 'Traffic': 0.003},
    'GPT4TS': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.0001, 'ECL': 0.0001},
    'PatchTST': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'iTransformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.00001, 'ECL': 0.0005},
    'NLinear': {'ETTh2': 0.05, 'ETTm1': 0.05, 'Traffic': 0.005, 'Weather': 0.01, 'ECL': 0.01},
    'Informer_RevIN': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'Autoformer_RevIN': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'Informer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'Autoformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001}
}

pretrain_lr_dict = {
    'PatchTST': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'iTransformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.0001, 'ECL': 0.0005},
}


def drop_last_PatchTST(args):
    bs = 128 if args.dataset in ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Weather'] else 32
    test_num = args.borders[1][2] - args.borders[0][2] - args.seq_len - args.pred_len + 1
    args.borders[1][2] -= test_num % bs
