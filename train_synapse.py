import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from datasets.dataset import Synapse_dataset, RandomGenerator
from tensorboardX import SummaryWriter
from models.vmunet.wesmunet import WESMUNet
from engine_synapse import *
import os
import sys
from utils import *
from configs.config_setting_synapse import setting_config

# 直接导入EMA类
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入EMA功能
try:
    # 尝试从utils.ema导入，如果失败则使用内联实现
    # 注意：由于utils.py存在，直接import utils.ema可能会失败
    # 这里使用更健壮的导入方式
    import importlib.util
    spec = importlib.util.spec_from_file_location("utils.ema", os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils/ema.py"))
    ema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ema_module)
    ModelEMA = ema_module.ModelEMA
    EMA_AVAILABLE = True
    print("✅ EMA模块导入成功")
except Exception as e:
    print(f"⚠️  EMA模块导入失败: {e}")
    print("🔄 使用内联EMA实现")
    EMA_AVAILABLE = False
    
    # 内联EMA实现
    import torch
    import torch.nn as nn
    import copy
    from typing import Optional

    class ModelEMA:
        def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
            self.ema = copy.deepcopy(model).eval()
            self.decay = decay
            self.device = device
            self.updates = 0
            
            for param in self.ema.parameters():
                param.requires_grad_(False)
                
            if device is not None:
                self.ema.to(device)
        
        def update(self, model: nn.Module):
            with torch.no_grad():
                self.updates += 1
                d = self.decay
                msd = model.state_dict()
                
                for k, v in self.ema.state_dict().items():
                    if v.dtype.is_floating_point:
                        v *= d
                        v += (1.0 - d) * msd[k].detach()
        
        def state_dict(self):
            return {
                'ema': self.ema.state_dict(),
                'updates': self.updates,
                'decay': self.decay
            }
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
    train_dataset = Synapse_dataset(base_dir=config.data_path, list_dir=config.list_dir, split="train",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]))
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = Synapse_dataset(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
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
            # PFR 相关配置向下传入 WESMBackbone -> VSSLayer(_up)
            use_pfr=model_cfg.get('use_pfr', False),
            pfr_ratio=model_cfg.get('pfr_ratio', 0.7),
            pfr_use_gating=model_cfg.get('pfr_use_gating', False),
            # PDR-Soft 配置
            use_pdr=model_cfg.get('use_pdr', False),
            pdr_init_ratio=model_cfg.get('pdr_init_ratio', 0.7),
            pdr_gate_hidden_ratio=model_cfg.get('pdr_gate_hidden_ratio', 0.25),
    # PHR-Block 配置
    use_phr=model_cfg.get('use_phr', False),
    phr_ratio=model_cfg.get('phr_ratio', 0.6),
    phr_gate_hidden_ratio=model_cfg.get('phr_gate_hidden_ratio', 0.25),
    # EADB-Block 配置 (边缘增强版PHR)
    use_eadb=model_cfg.get('use_eadb', False),
    eadb_edge_enhance_factor=model_cfg.get('eadb_edge_enhance_factor', 2.0),
    # FGStem Embedding 配置
    use_fgstem_embed=model_cfg.get('use_fgstem_embed', True),
    # WGD Downsampling 配置
    use_wgd=model_cfg.get('use_wgd', False),
    # SASF 配置
    use_sasf=model_cfg.get('use_sasf', False),
        )
        model.load_from()
    elif config.network == 'unet':
        from models.baselines.unet import UNet
        model_cfg = config.model_config
        model = UNet(in_channels=model_cfg['input_channels'], num_classes=model_cfg['num_classes'])
    elif config.network == 'original_vmunet':
        from models.baselines.vmunet import VMUNet
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
    else: 
        raise Exception(f'network {config.network} is not supported!')
    model = model.cuda()

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # 初始化EMA
    model_ema = None
    if hasattr(config, 'use_ema') and config.use_ema:
        print(f'🔄 初始化EMA，衰减系数: {config.ema_decay}')
        model_ema = ModelEMA(model, decay=config.ema_decay, device=torch.device('cuda'))
        print('✅ EMA初始化完成')

    print('#----------Set other params----------#')
    best_dice = 0  # 最佳Dice分数，初始化为0，越大越好
    start_epoch = 1
    best_epoch = 1
    early_stopping_counter = 0
    early_stopping_patience = config.early_stopping_patience
    
    # 混合精度训练设置
    if config.amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        # 只测试模式使用完整验证（计算HD95）
        mean_dice, mean_hd95 = val_one_epoch(
                val_dataset,
                val_loader,
                model,
                epoch,
                logger,
                config,
                test_save_path=outputs,
                val_or_test=True,
                fast_val=False  # 测试时强制使用完整验证
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
        best_dice, best_epoch, loss = checkpoint['best_dice'], checkpoint['best_epoch'], checkpoint['loss']
        early_stopping_counter = checkpoint.get('early_stopping_counter', 0)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, best_dice: {best_dice:.4f}, best_epoch: {best_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        loss = train_one_epoch(
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
        
        # 根据val_interval决定是否进行验证
        if epoch % config.val_interval == 0:
            # 判断是否使用EMA进行验证
            use_ema_for_val = (model_ema is not None and 
                             hasattr(config, 'ema_warmup_epochs') and 
                             epoch >= config.ema_warmup_epochs)
            
            # 使用配置文件中的快速验证设置
            fast_val = getattr(config, 'fast_validation', True)  # 默认启用快速验证
            mean_dice, mean_hd95 = val_one_epoch(
                    val_dataset,
                    val_loader,
                    model,
                    epoch,
                    logger,
                    config,
                    test_save_path=outputs,
                    val_or_test=False,
                    model_ema=model_ema,
                    use_ema_for_val=use_ema_for_val,
                    fast_val=fast_val
                )
            if mean_dice > best_dice:  # 对于Dice，越大越好
                # 保存最佳模型的完整检查点（包含EMA权重）
                best_checkpoint_dict = {
                    'epoch': epoch,
                    'best_dice': mean_dice,
                    'best_epoch': epoch,
                    'loss': loss,
                    'mean_dice': mean_dice,
                    'mean_hd95': mean_hd95,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'early_stopping_counter': 0,
                }
                # 如果使用EMA，也保存EMA权重到best.pth
                if model_ema is not None:
                    best_checkpoint_dict['ema_state_dict'] = model_ema.state_dict()
                    logger.info(f"🔄 保存最佳模型 (Epoch {epoch}, Dice: {mean_dice:.4f}) - 包含EMA权重")
                else:
                    logger.info(f"📊 保存最佳模型 (Epoch {epoch}, Dice: {mean_dice:.4f}) - 仅原始权重")
                
                torch.save(best_checkpoint_dict, os.path.join(checkpoint_dir, 'best.pth'))
                best_dice = mean_dice
                best_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f'Early stopping triggered at epoch {epoch}. Best Dice: {best_dice:.4f} at epoch {best_epoch}')
                break
        else:
            # 不验证时，使用训练loss作为参考（简单处理）
            mean_dice = 0  # 占位符
            mean_hd95 = 0  # 占位符
        # 保存检查点（包含EMA权重）
        checkpoint_dict = {
            'epoch': epoch,
            'best_dice': best_dice,
            'best_epoch': best_epoch,
            'loss': loss,
            'mean_dice': mean_dice,
            'mean_hd95': mean_hd95,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_counter': early_stopping_counter,
        }
        
        # 如果使用EMA，也保存EMA权重
        if model_ema is not None:
            checkpoint_dict['ema_state_dict'] = model_ema.state_dict()
            
        torch.save(checkpoint_dict, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        
        # 加载最佳检查点
        best_checkpoint = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        
        # 检查是否有EMA权重并优先使用
        if model_ema is not None and 'ema_state_dict' in best_checkpoint:
            # 使用EMA权重进行最终测试
            ema_state_dict = best_checkpoint['ema_state_dict']['ema']
            model.load_state_dict(ema_state_dict)
            logger.info("🔄 最终测试使用EMA权重")
            test_weight_type = "EMA"
        else:
            # 使用普通权重
            if 'model_state_dict' in best_checkpoint:
                model.load_state_dict(best_checkpoint['model_state_dict'])
            else:
                model.load_state_dict(best_checkpoint)
            logger.info("📊 最终测试使用普通权重")
            test_weight_type = "Normal"
        
        # 最终测试使用完整验证模式（计算HD95）并生成可视化
        logger.info(f"🎯 开始最终测试 ({test_weight_type}权重) - 使用reproduce_val_vis逻辑")
        
        model.eval()
        metric_list = 0.0
        
        # 确保输出目录存在
        test_save_path = os.path.join(outputs, 'final_test_vis')
        os.makedirs(test_save_path, exist_ok=True)
        
        for i_batch, data in tqdm(enumerate(val_loader)):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            
            metric_i = test_single_volume(
                img, msk, model, 
                classes=config.num_classes, 
                patch_size=[config.input_size_h, config.input_size_w],
                test_save_path=test_save_path, 
                case=case_name, 
                z_spacing=config.z_spacing, 
                val_or_test=True,  # 关键：设为True以触发可视化保存
                dataset_type='synapse', 
                fast_val=False
            )
            
            metric_list += np.array(metric_i)
            # 使用print确保输出可见，tqdm可能会吞掉logger输出
            print(f"Case {case_name}: Mean Dice: {np.mean(metric_i, axis=0)[0]:.4f}")
            logger.info(f"Case {case_name}: Mean Dice: {np.mean(metric_i, axis=0)[0]:.4f}")
            
        # 计算平均指标
        metric_list = metric_list / len(val_dataset)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        
        result_msg = (
            "="*50 + "\n" +
            f"Final Test Result ({test_weight_type}):\n" +
            f"Mean Dice: {performance:.4f}\n" +
            f"Mean HD95: {mean_hd95:.4f}\n" +
            f"Visualization saved to: {test_save_path}\n" +
            "="*50
        )
        print(result_msg)
        logger.info(result_msg)
        
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-dice{best_dice:.4f}.pth')
        )

if __name__ == '__main__':
    config = setting_config()
    main(config)