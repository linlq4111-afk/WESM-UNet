from pickle import FALSE
from datasets.dataset import *
from utils import *

from datetime import datetime



class setting_config:
    """
    Synapse多器官数据集VM-UNet训练配置 (编码器纯VSS + 解码器EADB)
    - Encoder所有层: VSSBlock (Mamba全局建模)
    - Decoder所有层: EADB-Block (边缘增强融合)
    - 使用ImageNet预训练权重
    """
    network = 'vmunet'
    model_tag = 'encoder_vss_decoder_eap_pretrained'  # 编码器VSS + 解码器EADB + 预训练
    model_config = {
        'num_classes': 9,  # 9个器官类别
        'input_channels': 3, 
        # ----- VM-UNet 分层架构配置 ----- #
        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',  # 使用Small预训练权重
        # 混合架构配置
        # Encoder所有层: VSSBlock (Mamba-only)  
        # Decoder所有层: EADB-Block (边缘增强融合)
        'use_phr': False,            # 禁用PHR-Block，编码器使用纯VSS
        'phr_ratio': 0.4,            # PHR通道分割比例（已禁用）
        'phr_gate_hidden_ratio': 0.3, # PHR门控MLP隐藏维度比例（已禁用）
        'use_eadb': True,             # 启用EADB-Block，解码器使用边缘增强
        'eadb_edge_enhance_factor': 3, # EADB边缘增强系数：针对Class 2&6小器官
        'use_fgstem_embed': True,        # 启用FGStem增强的PatchEmbedding
        'use_wgd': True,             # 启用WGD下采样
        'use_sasf': True,      # 启用SASF
    }
    
    datasets_name = 'synapse' 
    input_size_h = 224
    input_size_w = 224
    if datasets_name == 'synapse':
        data_path = './data/Synapse/train_npz/'
        datasets = Synapse_dataset
        list_dir = './data/Synapse/lists/lists_Synapse/'
        volume_path = './data/Synapse/test_vol_h5/'
    else:
        raise Exception('datasets in not right!')
    
    pretrained_path = '' # if using pretrained, please enter the path of weights
    num_classes = 9
    input_channels = 3

    # 损失函数 - 平衡CE和Dice损失，避免过拟合
    loss_weight = [1, 1]  # 平衡CE和Dice损失权重
    criterion = CeDiceLoss(num_classes, loss_weight)
    
    z_spacing = 1

    distributed = False
    local_rank = -1
    num_workers = 16 #16
    seed = 42
    world_size = None
    rank = None
    amp = True  # 启用混合精度训练，节省显存
    gpu_id = '0'

    batch_size = 12  # 尝试12，最大化显存利用
    epochs = 300
    early_stopping_patience = 50

    # ----- EMA (指数滑动平均) 配置 ----- #
    use_ema = True                    # 是否使用EMA
    ema_decay = 0.9999                 # EMA衰减系数，推荐0.999或0.9999
    ema_warmup_epochs = 0             # EMA预热轮数，立即开始使用EMA进行验证
    use_ema_for_test = True           # 最终测试时是否优先使用EMA权重

    # 训练结果文件夹命名
    work_dir = f'results/synapse/{network}_{model_tag}_{datasets_name}_bs{batch_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    # 强制指定为上次中断的目录以进行断点续训
    # work_dir = 'results/synapse/vmunet_encoder_vss_decoder_eap_pretrained_synapse_bs12_20251212_102043/'
    
    print_interval = 10
    val_interval = 10  # 300轮训练，适当减少验证频率节省时间
    save_interval = 50   # 更频繁保存，防止长训练意外中断
    threshold = 0.5
    
    # ----- 快速验证配置 ----- #
    fast_validation = True
    only_test_and_save_figs = False
    best_ckpt_path = 'PATH_TO_YOUR_BEST_CKPT'
    img_save_path = 'PATH_TO_SAVE_IMAGES'
    
    # 优化器配置
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
        lr = 0.0001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.05
        amsgrad = False
    elif opt == 'AdamW':
        lr = 4e-4    # Batch Size 12，学习率调整为4e-4
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.04 # 适中的正则化强度
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
        lr = 0.003
        momentum = 0.9
        weight_decay = 0.0001
        dampening = 0
        nesterov = False
    
    # 学习率调度器
    sch = 'WP_CosineLR'  # 使用带warmup的余弦调度，适合预训练权重
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
        T_max = 300  # 匹配新的epochs
        eta_min = 0.000001  # 300轮训练，最小学习率可以更低
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
        warm_up_epochs = 20  # 300轮训练，适当延长warmup
        eta_min = 1e-6       # 更低的最小学习率，充分收敛
        last_epoch = -1

    # 数据增强已更新为从头训练版本，包含：
    # - 几何增强：随机旋转/翻转/缩放([0.5,2.0])/平移(±20%)
    # - 强度增强：对比度([0.5,1.5])/高斯噪声(0.005)/高斯模糊(σ=1.0)
    # - 标准化：[0,1] → [-1,1] 简单映射，不使用ImageNet预处理
    # - 与msvm-unet-main的数据增强策略保持一致
