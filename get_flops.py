
import torch
from thop import profile
from models.vmunet.wesmunet import WESMUNet
from models.vmunet.wesm_backbone import SS2D, flops_selective_scan_ref

def ss2d_flops_handler(module, input, output):
    x = input[0] # Input to SS2D forward is (B, H, W, C)
    # VMamba SS2D input is (B, H, W, C) as per wesm_backbone.py
    B, H, W, C = x.shape
    L = H * W
    K = 4 
    
    flops = 0
    
    # 1. in_proj: Linear(d_model, d_inner*2)
    # Input is (B, H, W, d_model) -> (B, H, W, d_inner*2) ? 
    # Or (B, d_model, H, W)?
    # module.in_proj is Linear. Linear expects (..., in_features).
    # If x is (B, C, H, W), it might need permute.
    # But usually FLOPs is just InputElements * OutputDim.
    # Input elements = B * L. Output dim = d_inner * 2.
    flops += B * L * module.d_model * (module.d_inner * 2)
    
    # 2. conv2d (Depthwise, groups=d_inner)
    # [B, d_inner, H, W]
    # k = d_conv (3)
    flops += B * module.d_inner * L * module.d_conv * module.d_conv
    
    # 3. act (SiLU)
    flops += B * module.d_inner * L
    
    # 4. x_proj (Einsum)
    # N = dt_rank + d_state*2
    N = module.dt_rank + module.d_state * 2
    flops += B * K * L * N * module.d_inner
    
    # 5. dt_proj (Einsum)
    flops += B * K * L * module.d_inner * module.dt_rank
    
    # 6. Selective Scan
    # Use reference function
    # flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False)
    # We invoke it with our dims
    # B_scan = B * K, L=L, D=d_inner, N=d_state
    
    # Handle D/Z args based on module attributes?
    # SS2D usually has D (skip), Z (gating).
    # forward_corev0 doesn't seem to use Z explicitly in the scan call provided in file?
    # forward_corev0:
    # out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, ...)
    # So z=None. with_Z=False.
    # Ds is passed. with_D=True.
    scan_flops = flops_selective_scan_ref(B=B*K, L=L, D=module.d_inner, N=module.d_state, with_D=True, with_Z=False, with_Group=True)
    if scan_flops is not None:
        flops += scan_flops
    else:
        # Fallback approximation
        flops += 9 * B * K * L * module.d_inner * module.d_state

    # 7. out_norm (LayerNorm)
    flops += B * L * module.d_inner
    
    # 8. out_proj
    flops += B * L * module.d_inner * module.d_model
    
    # Debug print for all layers
    print(f"DEBUG: SS2D Call: L={L}, dim={module.d_model}, FLOPs={flops/1e9:.4f} G")
    
    # Store directly into module attributes as expected by thop if handler returns None?
    # Or rely on return value?
    # If we return None, output is preserved (if hook logic allows).
    # If we return a tuple, it replaces output and crashes downstream.
    # Standard thop handlers: count_conv2d returns only FLOPs (int).
    # If thop uses strict handler(m, x, y) returns pops.
    # It must be that thop catches the return value and sums it to m.total_ops.
    # But if it does that, why does VSSBlock see the tuple?
    # This implies the hook returns the tuple and PyTorch replaces output.
    
    # Workaround:
    # 1. Assign to module.total_ops manually.
    # 2. Return None.
    
    module.total_ops += torch.DoubleTensor([flops])
    module.total_params += torch.DoubleTensor([sum(p.numel() for p in module.parameters())])
    
    # IMPORTANT: Return None to avoid modifying the module output!
    pass

def get_flops_params():
    # 按照 config_setting_synapse.py 的配置
    model_config = {
        'num_classes': 9,
        'input_channels': 3,
        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'use_phr': False,
        'use_eadb': True,  # 你的解码器用了EADB
        'use_fgstem_embed': True, # frontend用了FGStem
        'use_wgd': True, # 下采样用了WGD
        'use_sasf': True # Skip connection用了Semantic Align
    }

    model = VMUNet_FGStemBlock(
        input_channels=model_config['input_channels'],
        num_classes=model_config['num_classes'],
        depths=model_config['depths'],
        depths_decoder=model_config['depths_decoder'],
        drop_path_rate=model_config['drop_path_rate'],
        load_ckpt_path='',
        use_phr=model_config['use_phr'],
        use_eadb=model_config['use_eadb'],
        use_fgstem_embed=model_config['use_fgstem_embed'],
        use_wgd=model_config['use_wgd'],
        use_sasf=model_config['use_sasf'] if 'use_sasf' in model_config else True
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to run get_flops.py because mamba_ssm selective_scan uses CUDA kernels. "
            "Please run this script on a machine with an available NVIDIA GPU and CUDA-enabled PyTorch."
        )

    device = torch.device('cuda')
    input = torch.randn(1, 3, 224, 224, device=device)
    model = model.to(device)
    
    # 获取正确的参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 注册自定义handler处理SS2D模块，避免漏算scan FLOPs
    flops, params = profile(model, inputs=(input, ), custom_ops={SS2D: ss2d_flops_handler}, verbose=False)
    
    print(f"==========================================")
    print(f"VM-UNet (Synapse Config) Model Analysis")
    print(f"Device: {device}")
    print(f"Input Size: (1, 3, 224, 224)")
    print(f"------------------------------------------")
    print(f"Params (PyTorch): {total_params / 1e6:.2f} M")
    print(f"Params (Thop):    {params / 1e6:.2f} M")
    print(f"FLOPs:  {flops / 1e9:.2f} G")
    print(f"==========================================")

if __name__ == '__main__':
    get_flops_params()
