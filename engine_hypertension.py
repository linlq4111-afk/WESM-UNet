import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs



def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    多分类训练一个epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).long()  # 多分类使用long类型
        targets = targets.squeeze(1)  # 修正：去掉多余的通道，shape变为[N, H, W]

        out = model(images)
        
        # 模型输出应该与标签尺寸匹配（已修复Final_PatchExpand2D为4倍上采样）
        
        # 使用原始损失函数
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).long()  # 多分类使用long类型
            msk = msk.squeeze(1)  # 修正：去掉多余的通道，shape变为[N, H, W]

            out = model(img)
            
            # 模型输出应该与标签尺寸匹配（已修复Final_PatchExpand2D为4倍上采样）
            
            # 使用原始损失函数
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            
            # 多分类预测
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            gts.append(msk.cpu().detach().numpy())
            preds.append(pred.cpu().detach().numpy())

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # 计算每个类别的指标
        class_names = ['背景', 'R1(肌肉)', 'R2(肌间脂肪)', 'R3(皮下脂肪)', 'R4(肺)']
        metrics = {}
        
        for i in range(5):  # 5个类别
            y_true = (gts == i).astype(int)
            y_pred = (preds == i).astype(int)
            
            if np.sum(y_true) > 0 or np.sum(y_pred) > 0:
                confusion = confusion_matrix(y_true, y_pred)
                if confusion.shape == (2, 2):
                    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
                    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
                    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
                    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
                    dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
                    iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
                else:
                    accuracy = sensitivity = specificity = dice = iou = 0
            else:
                accuracy = sensitivity = specificity = dice = iou = 0
                
            metrics[class_names[i]] = {
                'dice': dice,
                'iou': iou,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity
            }

        # 计算平均指标
        avg_dice = np.mean([metrics[name]['dice'] for name in class_names[1:]])  # 排除背景
        avg_iou = np.mean([metrics[name]['iou'] for name in class_names[1:]])    # 排除背景
        
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, avg_dice: {avg_dice:.4f}, avg_iou: {avg_iou:.4f}'
        print(log_info)
        logger.info(log_info)
        
        # 打印每个类别的指标
        for name in class_names:
            if name in metrics:
                log_info = f'  {name}: Dice={metrics[name]["dice"]:.4f}, IoU={metrics[name]["iou"]:.4f}'
                print(log_info)
                logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).long()  # 多分类使用long类型

            out = model(img)
            
            # 模型输出应该与标签尺寸匹配（已修复Final_PatchExpand2D为4倍上采样）
            
            # 使用原始损失函数
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            
            # 多分类预测
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            msk_np = msk.cpu().detach().numpy()
            pred_np = pred.cpu().detach().numpy()
            
            gts.append(msk_np)
            preds.append(pred_np)
            
            if i % config.save_interval == 0:
                # 保存预测结果图像
                save_imgs(img, msk_np, pred_np, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # 计算每个类别的指标
        class_names = ['背景', 'R1(肌肉)', 'R2(肌间脂肪)', 'R3(皮下脂肪)', 'R4(肺)']
        metrics = {}
        
        for i in range(5):  # 5个类别
            y_true = (gts == i).astype(int)
            y_pred = (preds == i).astype(int)
            
            if np.sum(y_true) > 0 or np.sum(y_pred) > 0:
                confusion = confusion_matrix(y_true, y_pred)
                if confusion.shape == (2, 2):
                    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
                    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
                    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
                    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
                    dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
                    iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
                else:
                    accuracy = sensitivity = specificity = dice = iou = 0
            else:
                accuracy = sensitivity = specificity = dice = iou = 0
                
            metrics[class_names[i]] = {
                'dice': dice,
                'iou': iou,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity
            }

        # 计算平均指标
        avg_dice = np.mean([metrics[name]['dice'] for name in class_names[1:]])  # 排除背景
        avg_iou = np.mean([metrics[name]['iou'] for name in class_names[1:]])    # 排除背景

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
            
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, avg_dice: {avg_dice:.4f}, avg_iou: {avg_iou:.4f}'
        print(log_info)
        logger.info(log_info)
        
        # 打印每个类别的指标
        for name in class_names:
            if name in metrics:
                log_info = f'  {name}: Dice={metrics[name]["dice"]:.4f}, IoU={metrics[name]["iou"]:.4f}'
                print(log_info)
                logger.info(log_info)

    return np.mean(loss_list) 