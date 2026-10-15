import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from utils import test_single_volume, calculate_metric_percase

import time


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config,
                    scaler=None,
                    model_ema=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
        
        # 更新EMA权重
        if model_ema is not None:
            model_ema.update(model)
            
    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss





def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False,
                    model_ema=None,
                    use_ema_for_val=False,
                    fast_val=None):
    dataset_type = 'acdc'
    # 选择使用原始模型还是EMA模型进行验证
    eval_model = model
    if use_ema_for_val and model_ema is not None:
        eval_model = model_ema.ema
        print(f"🔄 Epoch {epoch}: 使用EMA权重进行验证")
    else:
        print(f"🔄 Epoch {epoch}: 使用原始权重进行验证")
    
    # 确定是否使用快速验证模式
    if fast_val is None:
        fast_val = not val_or_test  # 默认：验证时快速模式，测试时完整模式
    
    mode_str = "快速验证(仅Dice)" if fast_val else "完整验证(Dice+HD95)"
    print(f"🚀 验证模式: {mode_str}")
    
    # switch to evaluate mode
    stime = time.time()
    eval_model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            
            # ACDC Validation (Always 3D Volume)
            if dataset_type == 'acdc':
                 metric_i = test_single_volume(img, msk, eval_model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                        test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test, dataset_type='acdc', fast_val=fast_val)
            
            # Legacy 2D Slice Validation (if needed for other datasets)
            elif len(img.shape) == 4 and img.shape[1] == 3:
                img = img.cuda().float()
                out = eval_model(img)
                pred = torch.argmax(torch.softmax(out, dim=1), dim=1).cpu().detach().numpy()
                msk = msk.numpy()
                
                # Calculate metrics for this slice
                # metric_i shape: [num_classes-1, 2] (dice, hd95)
                metric_i = []
                for i in range(1, config.num_classes):
                    dice, hd95 = calculate_metric_percase(pred[0] == i, msk[0] == i, compute_hd95=not fast_val)
                    metric_i.append([dice, hd95])
                metric_i = np.array(metric_i)
            else:
                # Fallback to 3D volume testing (if data is 3D)
                metric_i = test_single_volume(img, msk, eval_model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                        test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test, dataset_type='acdc', fast_val=fast_val)
            
            metric_list += np.array(metric_i)

            if fast_val:
                logger.info('idx %d case %s mean_dice %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0]))
            else:
                logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                            np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            if fast_val:
                logger.info('Mean class %d mean_dice %f' % (i, metric_list[i-1][0]))
            else:
                logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        etime = time.time()
        if fast_val:
            log_info = f'val epoch: {epoch}, mean_dice: {performance}, time(s): {etime-stime:.2f} [快速验证模式]'
            print(log_info)
            logger.info(log_info)
        else:
            log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
            print(log_info)
            logger.info(log_info)
    
    return performance, mean_hd95
