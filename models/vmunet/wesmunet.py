from models.vmunet.wesm_backbone import WESMBackbone
import torch
from torch import nn


class VMUNet_FGStemBlock(nn.Module):
    """
    VM-UNet with FGStem Enhancement
    Uses FGStemEmbed2D instead of standard PatchEmbedding for improved feature extraction
    """
    def __init__(self, input_channels=3, num_classes=5, **kwargs):
        super().__init__()
        # 提取load_ckpt_path参数
        self.load_ckpt_path = kwargs.pop('load_ckpt_path', None)
        self.vmunet = VSSM_FGStemBlock(in_chans=input_channels, num_classes=num_classes, **kwargs)
        # 自动统计参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f'当前模型总参数量: {total_params} ({total_params/1e6:.2f}M)')

    def forward(self, x):
        return self.vmunet(x)
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 只加载shape完全匹配的参数
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            # 如果启用了小波下采样，必须过滤掉预训练的downsample层权重
            # 因为预训练权重是针对空间PatchMerging的，而现在是针对频率域的小波系数
            # 形状虽然匹配(4C->2C)，但物理含义完全不同，强行加载会破坏特征
            if getattr(self.vmunet, 'use_wgd', False):
                print("🌊 检测到小波下采样：正在过滤预训练的downsample层权重...")
                keys_to_remove = [k for k in filtered_dict.keys() if 'downsample' in k]
                for k in keys_to_remove:
                    del filtered_dict[k]
                print(f"已移除 {len(keys_to_remove)} 个不匹配的downsample权重")

            model_dict.update(filtered_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(filtered_dict)))
            self.vmunet.load_state_dict(model_dict, strict=False)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in filtered_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            # decoder部分同理
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 只加载shape完全匹配的参数
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(filtered_dict)))
            self.vmunet.load_state_dict(model_dict, strict=False)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in filtered_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")