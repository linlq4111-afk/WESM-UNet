"""
EMA (Exponential Moving Average) for model parameters
指数滑动平均工具类，用于稳定训练和提升模型性能
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class ModelEMA:
    """
    模型指数滑动平均 (Exponential Moving Average)
    
    Args:
        model: 要应用EMA的模型
        decay: EMA衰减系数，通常为0.999或0.9999
        device: 设备类型
        resume: 恢复训练时的EMA权重路径
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        resume: Optional[str] = None
    ):
        # 创建EMA模型的深拷贝
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        self.updates = 0  # 更新次数
        
        # 冻结EMA模型参数，不参与梯度计算
        for param in self.ema.parameters():
            param.requires_grad_(False)
            
        # 如果指定设备，将EMA模型移动到该设备
        if device is not None:
            self.ema.to(device)
            
        # 如果有恢复路径，加载EMA权重
        if resume is not None:
            self.load_state_dict(resume)
    
    def update(self, model: nn.Module):
        """
        更新EMA权重
        
        Args:
            model: 当前训练的模型
        """
        with torch.no_grad():
            self.updates += 1
            
            # 动态调整衰减系数（可选）
            # 训练初期使用较小的衰减系数，后期使用较大的衰减系数
            d = self.decay
            # d = min(self.decay, (1 + self.updates) / (10 + self.updates))
            
            # 获取模型状态字典
            msd = model.state_dict()  # model state dict
            
            # 更新EMA模型的每个参数
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    # EMA更新公式: ema_param = decay * ema_param + (1 - decay) * current_param
                    v *= d
                    v += (1.0 - d) * msd[k].detach()
    
    def update_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ('process_group', 'reducer')):
        """
        更新EMA模型的属性（非参数属性）
        
        Args:
            model: 当前训练的模型
            include: 需要包含的属性
            exclude: 需要排除的属性
        """
        # 复制模型的非参数属性到EMA模型
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)
    
    def clone_ema_to(self, model: nn.Module):
        """
        将EMA权重克隆到指定模型
        
        Args:
            model: 目标模型
        """
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.shape == msd[k].shape:
                msd[k].copy_(v)
    
    def state_dict(self):
        """返回EMA模型的状态字典"""
        return {
            'ema': self.ema.state_dict(),
            'updates': self.updates,
            'decay': self.decay
        }
    
    def load_state_dict(self, checkpoint_path: str):
        """
        加载EMA状态字典
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
            self.updates = checkpoint.get('updates', 0)
            self.decay = checkpoint.get('decay', self.decay)
            print(f"✅ EMA权重已从 {checkpoint_path} 加载")
        else:
            print(f"⚠️  检查点中未找到EMA权重: {checkpoint_path}")
    
    def save(self, filepath: str):
        """
        保存EMA状态
        
        Args:
            filepath: 保存路径
        """
        torch.save(self.state_dict(), filepath)
        print(f"✅ EMA权重已保存到 {filepath}")
    
    @property
    def module(self):
        """返回EMA模型（兼容性接口）"""
        return self.ema
    
    def __call__(self, *args, **kwargs):
        """使EMA对象可以像模型一样调用"""
        return self.ema(*args, **kwargs)


def is_parallel(model: nn.Module) -> bool:
    """检查模型是否为并行模型"""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model: nn.Module) -> nn.Module:
    """获取并行模型的原始模型"""
    return model.module if is_parallel(model) else model


class EMAHook:
    """
    EMA钩子类，用于在训练过程中自动更新EMA权重
    """
    
    def __init__(self, model_ema: ModelEMA):
        self.model_ema = model_ema
    
    def __call__(self, model: nn.Module):
        """在每次训练步骤后调用"""
        self.model_ema.update(de_parallel(model))


# 使用示例
if __name__ == "__main__":
    # 创建一个简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # 测试EMA功能
    model = SimpleModel()
    ema = ModelEMA(model, decay=0.999)
    
    print("✅ EMA工具类创建成功！")
    print(f"EMA衰减系数: {ema.decay}")
    print(f"更新次数: {ema.updates}")



