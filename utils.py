import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric
from PIL import Image


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    
    # 获取所有属性（包括类属性和实例属性）
    # 优先使用 dir() 获取所有属性名称，因为config可能是类实例但属性定义在类中
    all_attrs = dir(config)
    
    for k in all_attrs:
        # 过滤掉私有属性和内置属性
        if k.startswith('_'):
            continue
            
        v = getattr(config, k)
        
        # 过滤掉模块、函数等不需要打印的类型
        # 保留 int, float, str, list, dict, tuple, bool, type(类), torch.nn.Module
        # 也可以简单地打印所有非函数/模块的东西
        import types
        if isinstance(v, (types.FunctionType, types.MethodType, types.ModuleType)):
            continue
            
        log_info = f'{k}: {v},'
        logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        # 修复WP_CosineLR支持eta_min参数
        eta_min_ratio = config.eta_min / config.lr if hasattr(config, 'eta_min') else 0.0
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    # 处理图像数据
    # 修正：反归一化ImageNet标准化，确保原图不是全黑
    if hasattr(img, 'cpu'):
        img = img.squeeze(0).detach().cpu().numpy()  # [3, H, W] 或 [H, W]
    if img.ndim == 3 and img.shape[0] == 3:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1,2,0))  # [H, W, 3]
    elif img.ndim == 2 or (img.ndim == 3 and img.shape[0] == 1):
        img = np.squeeze(img)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
    
    # 处理掩码
    msk = np.array(msk)
    msk_pred = np.array(msk_pred)
    if msk.ndim == 3:
        msk = np.squeeze(msk, axis=0)
    if msk_pred.ndim == 3:
        msk_pred = np.squeeze(msk_pred, axis=0)

    # 设置画布大小
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)

    # 原图
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')

    # 预测掩码（伪彩色）
    plt.subplot(1, 2, 2)
    plt.imshow(msk_pred, cmap='jet', vmin=0, vmax=4)
    plt.axis('off')
    plt.title('Predicted Mask (Color)')

    # 保存图片
    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, softmax=True):
        if softmax:
            pred = torch.softmax(pred, dim=1)
        
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
    

class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight
    
    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
    

class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss



class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        image = TF.to_tensor(image)  # (C, H, W), float32
        mask = torch.from_numpy(np.array(mask)).long()  # (H, W), int64
        return image, mask
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        # image
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
                image = np.transpose(image, (1, 2, 0))  # (H, W, C)
            image = image.astype(np.uint8)
        elif isinstance(image, np.ndarray):
            image = image.astype(np.uint8)
        else:
            image = np.array(image).astype(np.uint8)
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize((self.size_w, self.size_h), resample=Image.BILINEAR)
        # mask
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        elif not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        mask = mask.astype(np.uint8)
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((self.size_w, self.size_h), resample=Image.NEAREST)
        return image_pil, mask_pil
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.p = p
        self.degree = degree
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            image = image.rotate(angle)
            mask = mask.rotate(angle)
        return image, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
        elif data_name == 'hypertension':
            # 高血压数据集：使用ImageNet标准化
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            
    def __call__(self, data):
        img, msk = data
        if hasattr(self, 'mean') and isinstance(self.mean, list):
            # 高血压数据集：ImageNet标准化
            img = img.astype(np.float32) / 255.0
            img_normalized = (img - np.array(self.mean)) / np.array(self.std)
        else:
            # 其他数据集：原有标准化方法
            img_normalized = (img-self.mean)/self.std
            img_normalized = ((img_normalized - np.min(img_normalized)) 
                                / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk
    


from thop import profile		 ## 导入thop模块
# 动态导入Fusion类
import importlib.util
import os
fusion_path = os.path.join(os.path.dirname(__file__), 'models/selectedmoudel/1/5、 多尺度小波融合(CVPR 2024).py')
fusion_spec = importlib.util.spec_from_file_location("Fusion_module", fusion_path)
fusion_module = importlib.util.module_from_spec(fusion_spec)
fusion_spec.loader.exec_module(fusion_module)
Fusion = fusion_module.Fusion

# 注册空forward_hook，彻底跳过Fusion的FLOPs统计

def fusion_flops_counter_hook(module, input, output):
    module.total_ops = torch.zeros(1)

def cal_params_flops(model, input_size, logger):
    input = torch.randn(1, 3, input_size, input_size).to(next(model.parameters()).device)
    # 删除递归赋值total_ops，恢复thop原生行为
    flops, params = profile(model, inputs=(input,))
    logger.info(f"FLOPs: {flops}, Params: {params}")
    return flops, params






def calculate_metric_percase(pred, gt, compute_hd95=True):
    """
    计算单个案例的指标
    Args:
        pred: 预测结果
        gt: 真值标签
        compute_hd95: 是否计算HD95，验证时设为False节省时间
    Returns:
        dice, hd95 (如果compute_hd95=False，hd95返回0)
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        if compute_hd95:
            hd95 = metric.binary.hd95(pred, gt)
        else:
            hd95 = 0.0  # 验证时不计算HD95，节省时间
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0



def test_single_volume(image, label, net, classes, patch_size=[256, 256], 
                    test_save_path=None, case=None, z_spacing=1, val_or_test=False, dataset_type='default', fast_val=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # 暂时禁用TTA，回到原始单次推理（避免复杂bug）
            # Synapse数据集已经预处理为[0,1]范围，无需额外的HU值标准化
            # 保持与训练一致：直接使用[0,1]数据
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  # [1, 1, H, W]
            # 根据数据集类型处理通道数
            if dataset_type in ['synapse', 'acdc']:
                # Synapse和ACDC数据集：将单通道图像重复为3通道以匹配模型输入
                input = input.repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W]
                # 从头训练：使用简单的标准化，不使用ImageNet预处理
                # 将[0,1]范围映射到[-1,1]范围，与训练时保持一致
                input = input * 2.0 - 1.0
            # 其他数据集（如hypertension）保持原有格式
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
            # 根据数据集类型处理通道数
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            prediction[ind] = pred
    else:
        # Synapse数据集已经预处理为[0,1]范围，无需额外的HU值标准化
        # 保持与训练一致：直接使用[0,1]数据
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        # 根据数据集类型处理通道数
        if dataset_type in ['synapse', 'acdc']:
            # Synapse和ACDC数据集：将单通道图像重复为3通道以匹配模型输入
            input = input.repeat(1, 3, 1, 1)
            # 从头训练：使用简单的标准化，不使用ImageNet预处理
            # 将[0,1]范围映射到[-1,1]范围，与训练时保持一致
            input = input * 2.0 - 1.0
        # 其他数据集（如hypertension）保持原有格式
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    # 根据是否为快速验证模式决定是否计算HD95
    compute_hd95 = not fast_val  # 快速验证时不计算HD95
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i, compute_hd95=compute_hd95))

    if test_save_path is not None and val_or_test is True:
        case_num = 999  # default fallback to avoid unbound variable on visualization import failure
        # 导入可视化工具
        try:
            import importlib.util
            utils_vis_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils_visualization.py")
            spec = importlib.util.spec_from_file_location("project_utils_visualization", utils_vis_path)
            vis_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vis_module)
            save_synapse_visualization = vis_module.save_synapse_visualization
            
            # 解析case编号（更鲁棒的方法）
            import re
            case_match = re.search(r'case(\d+)', case)
            if case_match:
                case_num = int(case_match.group(1))
            else:
                case_num = 999  # 如果解析失败，设为大数字跳过
            
            print(f"处理案例: {case}, 编号: {case_num}")
            
            # 保存所有案例的可视化
            if True:  # 强制保存所有案例
                print(f"开始生成可视化: {case}")
                # 使用新的全切片保存函数
                try:
                    save_all_slices_visualization = vis_module.save_all_slices_visualization
                    save_all_slices_visualization(
                        image=image, 
                        gt_mask=label, 
                        pred_mask=prediction,
                        case_name=case,
                        save_dir=test_save_path
                    )
                except ImportError:
                    # 回退到旧的单切片保存
                    save_synapse_visualization(
                        image=image, 
                        gt_mask=label, 
                        pred_mask=prediction,
                        case_name=case,
                        save_dir=test_save_path
                    )
            
        except Exception as e:
            print(f"可视化保存失败 {case}: {e}")
            import traceback
            traceback.print_exc()
        
        # 保留原始nii.gz格式（但只保存前3个案例）
        try:
            if case_num <= 3:
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
                lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
                img_itk.SetSpacing((1, 1, z_spacing))
                prd_itk.SetSpacing((1, 1, z_spacing))
                lab_itk.SetSpacing((1, 1, z_spacing))
                sitk.WriteImage(prd_itk, test_save_path + '/'+ case + "_pred.nii.gz")
                sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
                sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
        except Exception as e:
            print(f"NII保存失败 {case}: {e}")
    return metric_list


class MultiClassCrossEntropyDiceLoss(nn.Module):
    """多分类交叉熵 + Dice损失函数"""
    def __init__(self, num_classes=5, ce_weight=1.0, dice_weight=1.0, class_weights=None):
        super(MultiClassCrossEntropyDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target):
        """计算Dice损失"""
        smooth = 1e-6
        pred_softmax = F.softmax(pred, dim=1)
        # 将target转换为one-hot编码
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

    def forward(self, pred, target):
        # 保证权重和pred在同一设备
        if hasattr(self.ce_loss, 'weight') and self.ce_loss.weight is not None:
            self.ce_loss.weight = self.ce_loss.weight.to(pred.device)
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total_loss
