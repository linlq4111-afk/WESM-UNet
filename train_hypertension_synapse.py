import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import Synapse_dataset
from tensorboardX import SummaryWriter
from models.vmunet.wesmunet import WESMUNet

from engine_synapse import *  # 使用Synapse的训练引擎
import os
import sys

from utils import *
from configs.config_setting_hypertension_synapse import setting_config
from datasets.dataset import RandomGenerator

# 直接导入EMA类
from timm.utils import ModelEmaV2

import warnings
warnings.filterwarnings("ignore")

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    # 为单通道数据创建transform
    train_transform = RandomGenerator(output_size=[config.input_size_h, config.input_size_w])
    
    train_dataset = config.datasets(config.data_path, config.list_dir, split="train", transform=train_transform)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)
    val_dataset = config.datasets(config.test_vol_path, config.list_dir, split="test_vol", transform=None)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet_FGStemBlock(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
            # 小波变换配置
            use_wgd=model_cfg.get('use_wgd', False),
            # PHR-Block 配置
            use_phr=model_cfg.get('use_phr', False),
            phr_ratio=model_cfg.get('phr_ratio', 0.4),
            phr_gate_hidden_ratio=model_cfg.get('phr_gate_hidden_ratio', 0.3),
            # EADB-Block 配置
            use_eadb=model_cfg.get('use_eadb', False),
            eadb_edge_enhance_factor=model_cfg.get('eadb_edge_enhance_factor', 3),
            # FGStem Embedding 配置
            use_fgstem_embed=model_cfg.get('use_fgstem_embed', True),
        )
        model.load_from()
        
    elif config.network == 'unet':
        from models.baselines.unet import UNet
        model = UNet(in_channels=model_cfg['input_channels'], num_classes=model_cfg['num_classes'])
        
    elif config.network == 'original_vmunet':
        from models.baselines.vmunet import VMUNet
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path']
        )
        model.load_from()

    else: 
        raise Exception(f'network {config.network} is not supported!')
    model = model.cuda()

    # EMA模型初始化
    if config.network == 'original_vmunet':
        model_ema = None
    else:
        model_ema = ModelEmaV2(model, decay=0.999)

    cal_params_flops(model, 224, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # 混合精度训练scaler
    if config.amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        loss = val_one_epoch(
                val_dataset,
                val_loader,
                model,
                0,  # epoch
                logger,
                config,
                test_save_path=config.work_dir + 'outputs/',
                val_or_test=True,  # 测试模式
                model_ema=None,
                use_ema_for_val=False,
                fast_val=False  # 测试时使用完整验证
            )
        return

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        
        # 恢复EMA模型
        if 'model_ema_state_dict' in checkpoint and model_ema is not None:
            model_ema.module.load_state_dict(checkpoint['model_ema_state_dict'])

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler,
            model_ema
        )

        performance, mean_hd95 = val_one_epoch(
            val_dataset,
            val_loader,
            model,
            epoch,
            logger,
            config,
            test_save_path=None,
            val_or_test=False,
            model_ema=model_ema,
            use_ema_for_val=False,
            fast_val=True  # 训练时使用快速验证
        )

        # 使用负的Dice作为loss（Dice越高越好，所以用负值作为loss）
        loss = -performance
        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            if model_ema is not None:
                torch.save(model_ema.module.state_dict(), os.path.join(checkpoint_dir, 'best_ema.pth'))
            min_loss = loss
            min_epoch = epoch

        save_dict = {
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        if model_ema is not None:
            save_dict['model_ema_state_dict'] = model_ema.module.state_dict()
            
        torch.save(save_dict, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = val_one_epoch(
                val_dataset,
                val_loader,
                model,
                0,  # epoch
                logger,
                config,
                test_save_path=config.work_dir + 'outputs/',
                val_or_test=True,  # 测试模式
                model_ema=None,
                use_ema_for_val=False,
                fast_val=False  # 测试时使用完整验证
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best_ema.pth'),
            os.path.join(checkpoint_dir, f'best_ema-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )

if __name__ == '__main__':
    config = setting_config()
    main(config)
