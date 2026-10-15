from pickle import FALSE
from datasets.dataset import *
from utils import *

from datetime import datetime

class setting_config:
    """
    高血压数据集VM-UNet训练配置 (编码器纯VSS + 解码器EADB)
    - 使用NPZ训练数据 + H5测试数据
    - Encoder所有层: VSSBlock (Mamba全局建模)
    - Decoder所有层: EADB-Block (边缘增强融合)
    - 使用ImageNet预训练权重
    """
    network = 'original_vmunet'
    model_tag = 'baseline_original_vmunet'
    model_config = {
        'num_classes': 5,  # 5类：背景(0) + 肌肉(1) + 肌间脂肪(2) + 皮下脂肪(3) + 肺(4)
        'input_channels': 3,  # RandomGenerator会将单通道转换为3通道 
        # ----- VM-UNet 分层架构配置 ----- #
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.15,  # 降低DropPath，适合高血压数据集
        'load_ckpt_path': './pre_trained_weights/vmamba_tiny_e292.pth',  # 使用ImageNet预训练权重
        # 混合架构配置
        # Encoder所有层: VSSBlock (Mamba-only)  
        # Decoder所有层: EADB-Block (边缘增强融合)
        'use_wgd': True,         # 启用WGD下采样
        'use_phr': False,            # 禁用PHR-Block，编码器使用纯VSS
        'phr_ratio': 0.4,            # PHR通道分割比例（已禁用）
        'phr_gate_hidden_ratio': 0.3, # PHR门控MLP隐藏维度比例（已禁用）
        'use_eadb': True,             # 启用EADB-Block，解码器使用边缘增强
        'eadb_edge_enhance_factor': 4, # EADB边缘增强系数：针对小器官
        'use_fgstem_embed': True,        # 启用FGStem增强的PatchEmbedding
    }
    
    datasets_name = 'hypertension' 
    input_size_h = 224  # 与Synapse保持一致，避免维度问题
    input_size_w = 224
    if datasets_name == 'hypertension':
        data_path = './data/hypertension_synapse_format/train_npz/'  # 训练数据路径
        test_vol_path = './data/hypertension_synapse_format/test_vol_h5/'  # 测试数据路径
        datasets = Synapse_dataset  # 使用Synapse数据集类
        list_dir = './data/hypertension_synapse_format/lists/lists_hypertension/'  # 列表文件路径
        
    # 多分类损失函数 - 使用与Synapse相同的损失函数
    criterion = CeDiceLoss(num_classes=5, loss_weight=[0.4, 0.6])  # [CE权重, Dice权重]
    
    pretrained_path = './pre_trained/'
    num_classes = 5  # 5类分割
    input_channels = 3  # RandomGenerator会将单通道转换为3通道
    z_spacing = 1  # 3D间距，用于HD95计算
    distributed = False
    local_rank = -1
    num_workers = 8
    seed = 42
    world_size = None
    rank = None
    amp = True  # 启用混合精度训练
    gpu_id = '0'
    batch_size = 12  # 增加batch size以提高训练稳定性 (原为4)
    epochs = 200   # 增加训练轮数 (原为200)
    early_stopping_patience = 50  # 更多耐心

    # 训练结果文件夹命名
    work_dir = f'results/{datasets_name}/{network}_{model_tag}_{datasets_name}_bs{batch_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'

    print_interval = 20
    val_interval = 1   # 每轮都验证
    save_interval = 50
    threshold = 0.5
    only_test_and_save_figs = False
    best_ckpt_path = 'PATH_TO_YOUR_BEST_CKPT'
    img_save_path = 'PATH_TO_SAVE_IMAGES'

    # 优化器配置 - 与Synapse保持一致
    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01
        rho = 0.9
        eps = 1e-6
        weight_decay = 0.05
    elif opt == 'Adagrad':
        lr = 0.01
        lr_decay = 0
        eps = 1e-10
        weight_decay = 0.05
    elif opt == 'Adam':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0001
        amsgrad = False
    elif opt == 'AdamW':
        lr = 0.0001  # 高血压数据集学习率
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.05
        amsgrad = False
    elif opt == 'Adamax':
        lr = 2e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
    elif opt == 'ASGD':
        lr = 0.01
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
    elif opt == 'RMSprop':
        lr = 1e-2
        momentum = 0
        alpha = 0.99
        eps = 1e-8
        centered = False
        weight_decay = 0
    elif opt == 'Rprop':
        lr = 1e-2
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
        dampening = 0
        nesterov = False
    
    # 学习率调度器 - 与Synapse保持一致
    sch = 'WP_CosineLR'
    if sch == 'StepLR':
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]
        gamma = 0.1
        last_epoch = -1
    elif sch == 'ExponentialLR':
        gamma = 0.99
        last_epoch = -1
    elif sch == 'CosineAnnealingLR':
        T_max = epochs
        eta_min = 0.00001
        last_epoch = -1
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'
        factor = 0.1
        patience = 10
        threshold = 0.0001
        threshold_mode = 'rel'
        cooldown = 0
        min_lr = 0
        eps = 1e-08
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50
        T_mult = 2
        eta_min = 1e-6
        last_epoch = -1
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 5  # 高血压数据集warmup轮数
        eta_min = 1e-6
        last_epoch = -1

    # ----- EMA (指数滑动平均) 配置 ----- #
    use_ema = True                    # 是否使用EMA
    ema_decay = 0.9999                 # EMA衰减系数，推荐0.999或0.9999
    ema_warmup_epochs = 0             # EMA预热轮数，立即开始使用EMA进行验证
    use_ema_for_test = True           # 最终测试时是否优先使用EMA权重
