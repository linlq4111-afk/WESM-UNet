import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

# 导入FGStem
from models.vmunet.stem import FGStemBlock
from .wavelet import DWT
# from .ssp_bridge import StateSharingBridge  # 模块不存在，已禁用
# from .anatomy_conditioner import AnatomyConditioner  # 模块不存在，已禁用
import torch.nn.functional as F

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class StandardPatchEmbed2D(nn.Module):
    r""" Standard Image to Patch Embedding (without FGStem)
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        # 标准的patch embedding - 直接用卷积进行patch化和通道转换
        # 使用4x4卷积，stride=4实现标准的patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 标准patch embedding: 一步完成patch化和特征提取
        x = self.proj(x)  # [B, embed_dim, H/4, W/4] - 标准4x降采样
        
        # 转换为序列格式
        x = x.permute(0, 2, 3, 1)  # [B, H/4, W/4, embed_dim]
        
        if self.norm is not None:
            x = self.norm(x)
        return x


class FGStemEmbed2D(nn.Module):
    r""" FGStem-based Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        # 不降采样的通道转换卷积
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0)
        
        # FGStem处理 - 使用原始配置以保持精度
        self.fgstem = FGStemBlock(c=embed_dim, DW_Expand=2, FFN_Expand=2, dilations=[1,2])
        
        # 适度的空间降采样 - 2x2而不是4x4，保留更多空间信息
        self.downsample = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, padding=0)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 通道转换，不降采样
        x = self.proj(x)  # [B, embed_dim, H, W] - 保持原始空间分辨率
        
        # FGStem处理 - 在原始分辨率上进行特征增强
        x = self.fgstem(x)  # [B, embed_dim, H, W]
        
        # 适度的空间降采样 - 2x2降采样，保留更多空间信息
        x = self.downsample(x)  # [B, embed_dim, H/2, W/2]
        
        # 转换为序列格式
        x = x.permute(0, 2, 3, 1)  # [B, H/2, W/2, embed_dim]
        
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class WaveletPatchMerging(nn.Module):
    r""" Wavelet Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        print(f"🌊 Initializing WaveletPatchMerging for dim={dim}")
        self.dim = dim
        self.dwt = DWT()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        
        # DWT expects B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()
        x_dwt = self.dwt(x) # B, 4*C, H/2, W/2
        
        # Back to B, H/2, W/2, 4*C
        x = x_dwt.permute(0, 2, 3, 1).contiguous()
        
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        # print(f"SS2D internal return type: {type(out)}")
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        att = self.self_attention(x)
        if isinstance(att, tuple):
            print(f"DEBUG: SS2D returned tuple of length {len(att)}")
            # Attempt to fix if it's returning extra stuff
            att = att[0]
        x = input + self.drop_path(att)
        return x






class HLABlock(nn.Module):
    """
    Hybrid Local Attention Block (HLA-Block)
    - 轻量大核卷积建模中尺度上下文，比FGStem更锐利
    - 专为PHR的部分通道场景优化，避免"两边都模糊"问题
    - 结合深度卷积(5x5) + 扩张卷积(7x7) + 点卷积的混合设计
    - 输入/输出: [B, C, H, W] (BCHW格式)
    """
    def __init__(self, c: int, kernel_sizes: list = [5, 7], dilation: int = 3):
        super().__init__()
        self.c = c
        
        # 多尺度深度卷积 - 捕获不同尺度的局部特征
        self.conv1 = nn.Conv2d(c, c, kernel_sizes[0], padding=kernel_sizes[0]//2, groups=c, bias=False)
        self.conv2 = nn.Conv2d(c, c, kernel_sizes[1], padding=(kernel_sizes[1]//2)*dilation, 
                              groups=c, dilation=dilation, bias=False)
        
        # 点卷积融合 + 门控机制
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.gate = nn.Conv2d(c, c, 1, bias=True)  # 门控权重
        
        # 归一化和激活
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        
        # 初始化门控偏向开放
        with torch.no_grad():
            self.gate.bias.fill_(0.0)  # sigmoid(0) = 0.5

    def forward(self, x):
        identity = x  # 残差连接
        
        # 多尺度特征提取
        out1 = self.act(self.bn1(self.conv1(x)))      # 5x5局部特征
        out2 = self.act(self.bn2(self.conv2(out1)))   # 7x7扩张特征
        
        # 点卷积融合
        out = self.pw(out2)
        
        # 门控机制 - 自适应特征选择
        gate_weight = torch.sigmoid(self.gate(x))
        out = out * gate_weight
        
        # 残差连接
        return identity + out


class EADBBlock(nn.Module):
    """
    Edge-Aware Dual-Branch Block (EADB-Block)
    - 边缘引导的PHR增强版，专为医学图像边界分割优化
    - 结合Sobel边缘检测 + 自适应边界增强 + 动态路由
    - 解决FGStem"全局模糊"问题，提升器官边界锐度
    - 输入/输出: [B, H, W, C] (BHWC格式，与PHR一致)
    """
    def __init__(
        self,
        hidden_dim: int = 0,
        phr_ratio: float = 0.6,
        gate_hidden_ratio: float = 0.25,
        edge_enhance_factor: float = 2.0,  # 边缘增强系数
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_enhance_factor = edge_enhance_factor
        
        # 第一层：Partial Channel Split (PFR 思想)
        self.c_cnn = max(1, int(hidden_dim * phr_ratio))
        self.c_mamba = max(1, hidden_dim - self.c_cnn)
        
        self.norm = norm_layer(hidden_dim)
        
        # 边缘检测模块 - Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # CNN 分支 - 使用轻量卷积，专注边缘特征
        self.cnn_dw1 = nn.Conv2d(self.c_cnn, self.c_cnn, 3, padding=1, groups=self.c_cnn, bias=False)
        self.cnn_dw2 = nn.Conv2d(self.c_cnn, self.c_cnn, 3, padding=2, dilation=2, groups=self.c_cnn, bias=False)
        self.cnn_pw = nn.Conv2d(self.c_cnn, self.c_cnn, 1, bias=False)
        self.cnn_bn = nn.BatchNorm2d(self.c_cnn)
        self.cnn_act = nn.SiLU()
        
        # Mamba 分支 (只处理 c_mamba 个通道)
        self.self_attention = SS2D(d_model=self.c_mamba, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        
        # 边缘自适应权重生成
        self.edge_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        # 第二层：Dynamic Routing Gate (PDR 思想)
        gate_hidden = max(1, int(hidden_dim * gate_hidden_ratio))
        self.gating_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, gate_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1, bias=True),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fuse = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drop_path = DropPath(drop_path)
        
        # 初始化门控偏向平衡
        with torch.no_grad():
            self.gating_module[-2].bias.fill_(0.0)

    def extract_edges(self, x):
        """提取边缘特征图"""
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 转为灰度图 (简单平均)
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Sobel边缘检测
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # 边缘强度
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)  # [B, 1, H, W]
        
        # 自适应边缘权重
        edge_weight = self.edge_conv(edge_magnitude)
        edge_weight = torch.sigmoid(edge_weight)
        
        return edge_weight

    def forward(self, input: torch.Tensor):
        B, H, W, C = input.shape
        x = self.norm(input)  # [B, H, W, C]
        
        # 第一层：Partial Channel Split
        x_cnn = x[..., :self.c_cnn]      # [B, H, W, c_cnn]
        x_mamba = x[..., self.c_cnn:]    # [B, H, W, c_mamba]
        
        # 边缘引导的CNN分支处理
        x_cnn_bchw = x_cnn.permute(0, 3, 1, 2).contiguous()  # [B, c_cnn, H, W]
        
        # 提取边缘权重
        edge_weight = self.extract_edges(x_cnn_bchw)  # [B, 1, H, W]
        
        # 边缘增强输入
        x_cnn_enhanced = x_cnn_bchw * (1 + self.edge_enhance_factor * edge_weight)
        
        # 轻量CNN处理 - 专注边缘特征
        y_cnn = self.cnn_dw1(x_cnn_enhanced)
        y_cnn = self.cnn_dw2(y_cnn)
        y_cnn = self.cnn_pw(y_cnn)
        y_cnn = self.cnn_act(self.cnn_bn(y_cnn))  # [B, c_cnn, H, W]
        y_cnn = y_cnn.permute(0, 2, 3, 1).contiguous()  # [B, H, W, c_cnn]
        
        # Mamba 分支处理
        y_mamba = self.self_attention(x_mamba)  # [B, H, W, c_mamba]
        if isinstance(y_mamba, tuple): y_mamba = y_mamba[0]
        
        # 第二层：Dynamic Routing Gate
        x_bchw = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        alpha = self.gating_module(x_bchw)  # [B, 1]
        alpha = alpha.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        
        # 分支间软路由融合
        y_cnn_extended = torch.cat([y_cnn, torch.zeros_like(y_mamba)], dim=-1)  # [B, H, W, C]
        y_mamba_extended = torch.cat([torch.zeros_like(y_cnn), y_mamba], dim=-1)  # [B, H, W, C]
        
        # 软路由：α 控制 CNN vs Mamba 的权重
        y_routed = alpha * y_cnn_extended + (1 - alpha) * y_mamba_extended
        
        # 融合与残差
        y = self.fuse(y_routed)
        out = input + self.drop_path(y)
        
        return out


class PHRBlock(nn.Module):
    """
    Partial Hybrid Routing Block (PHR-Block)
    - 结合 PFR 的通道分割思想和 PDR-Soft 的动态路由思想
    - 第一层：Partial Channel Split - 按比例将通道分给 CNN 和 Mamba 分支
    - 第二层：Dynamic Routing Gate - 用样本自适应权重融合两个分支
    - 相比 PDR-Soft 节省约 40-50% 计算量，同时保持动态适应能力
    - 输入/输出: [B, H, W, C]
    """
    def __init__(
        self,
        hidden_dim: int = 0,
        phr_ratio: float = 0.6,  # CNN 分支通道占比
        gate_hidden_ratio: float = 0.25,  # 门控 MLP 隐藏层比例
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 第一层：Partial Channel Split (PFR 思想)
        self.c_cnn = max(1, int(hidden_dim * phr_ratio))
        self.c_mamba = max(1, hidden_dim - self.c_cnn)
        
        self.norm = norm_layer(hidden_dim)
        
        # CNN 分支 (只处理 c_cnn 个通道) - 使用 HLA 增强局部特征，避免FGStem的"全局模糊"问题
        self.cnn_branch = HLABlock(c=self.c_cnn)
        
        # Mamba 分支 (只处理 c_mamba 个通道)
        self.self_attention = SS2D(d_model=self.c_mamba, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        
        # 第二层：Dynamic Routing Gate (PDR 思想)
        gate_hidden = max(1, int(hidden_dim * gate_hidden_ratio))
        self.gating_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context
            nn.Flatten(),
            nn.Linear(hidden_dim, gate_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1, bias=True),  # 输出标量权重
            nn.Sigmoid()
        )
        
        # 融合层
        self.fuse = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drop_path = DropPath(drop_path)
        
        # 初始化门控偏向平衡 (α ≈ 0.5)
        with torch.no_grad():
            self.gating_module[-2].bias.fill_(0.0)  # sigmoid(0) = 0.5

    def forward(self, input: torch.Tensor):
        B, H, W, C = input.shape
        x = self.norm(input)  # [B, H, W, C]
        
        # 第一层：Partial Channel Split
        x_cnn = x[..., :self.c_cnn]      # [B, H, W, c_cnn]
        x_mamba = x[..., self.c_cnn:]    # [B, H, W, c_mamba]
        
        # CNN 分支处理 - 使用 HLA
        x_cnn_bchw = x_cnn.permute(0, 3, 1, 2).contiguous()  # [B, c_cnn, H, W]
        y_cnn = self.cnn_branch(x_cnn_bchw)  # HLA 处理 - 更锐利的局部特征
        y_cnn = y_cnn.permute(0, 2, 3, 1).contiguous()  # [B, H, W, c_cnn]
        
        # Mamba 分支处理
        y_mamba = self.self_attention(x_mamba)  # [B, H, W, c_mamba]
        if isinstance(y_mamba, tuple): y_mamba = y_mamba[0]
        
        # 第二层：Dynamic Routing Gate
        x_bchw = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        alpha = self.gating_module(x_bchw)  # [B, 1]
        alpha = alpha.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        
        # 分支间软路由融合
        # 将两个分支的输出扩展到全通道维度
        y_cnn_extended = torch.cat([y_cnn, torch.zeros_like(y_mamba)], dim=-1)  # [B, H, W, C]
        y_mamba_extended = torch.cat([torch.zeros_like(y_cnn), y_mamba], dim=-1)  # [B, H, W, C]
        
        # 软路由：α 控制 CNN vs Mamba 的权重
        y_routed = alpha * y_cnn_extended + (1 - alpha) * y_mamba_extended
        
        # 融合与残差
        y = self.fuse(y_routed)
        out = input + self.drop_path(y)
        
        return out


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # PHR-Block 控制参数
        use_phr = kwargs.pop('use_phr', False)
        phr_ratio = kwargs.pop('phr_ratio', 0.6)
        phr_gate_hidden_ratio = kwargs.pop('phr_gate_hidden_ratio', 0.25)
        # EADB-Block 控制参数 (边缘增强版PHR)
        use_eadb = kwargs.pop('use_eadb', kwargs.pop('use_eap', False))
        eadb_edge_enhance_factor = kwargs.pop('eadb_edge_enhance_factor', kwargs.pop('eap_edge_enhance_factor', 2.0))

        blocks = []
        for i in range(depth):
            dp = drop_path[i] if isinstance(drop_path, list) else drop_path
            if use_phr:  # 优先级最高：PHR-Block
                blk = PHRBlock(
                    hidden_dim=dim,
                    phr_ratio=phr_ratio,
                    gate_hidden_ratio=phr_gate_hidden_ratio,
                    drop_path=dp,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    **kwargs,
                )
            else:
                blk = VSSBlock(
                    hidden_dim=dim,
                    drop_path=dp,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # PHR-Block 控制参数
        use_phr = kwargs.pop('use_phr', False)
        phr_ratio = kwargs.pop('phr_ratio', 0.6)
        phr_gate_hidden_ratio = kwargs.pop('phr_gate_hidden_ratio', 0.25)
        # EADB-Block 控制参数 (边缘增强版PHR)
        use_eadb = kwargs.pop('use_eadb', kwargs.pop('use_eap', False))
        eadb_edge_enhance_factor = kwargs.pop('eadb_edge_enhance_factor', kwargs.pop('eap_edge_enhance_factor', 2.0))

        blocks = []
        for i in range(depth):
            dp = drop_path[i] if isinstance(drop_path, list) else drop_path
            if use_eadb:  # 最高优先级：EADB-Block (边缘增强版PHR)
                blk = EADBBlock(
                    hidden_dim=dim,
                    phr_ratio=phr_ratio,
                    gate_hidden_ratio=phr_gate_hidden_ratio,
                    edge_enhance_factor=eadb_edge_enhance_factor,
                    drop_path=dp,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    **kwargs,
                )
            elif use_phr:  # 次优先级：PHR-Block (HLA增强版)
                blk = PHRBlock(
                    hidden_dim=dim,
                    phr_ratio=phr_ratio,
                    gate_hidden_ratio=phr_gate_hidden_ratio,
                    drop_path=dp,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    **kwargs,
                )
            else:
                blk = VSSBlock(
                    hidden_dim=dim,
                    drop_path=dp,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    


class VSSM_FGStemBlock(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # selective fusion & conditioning controls
        self.use_ssp_bridge = kwargs.pop('use_ssp_bridge', False)
        self.ssp_state_ratio = kwargs.pop('ssp_state_ratio', 0.25)
        self.use_anatomy_tokens = kwargs.pop('use_anatomy_tokens', False)
        self.anatomy_token_dim = kwargs.pop('anatomy_token_dim', 128)
        self.anatomy_token_dropout = kwargs.pop('anatomy_token_dropout', 0.0)
        
        # WGD 配置
        # SASF 配置
        self.use_sasf = kwargs.pop('use_sasf', kwargs.pop('use_semantic_align', False))
        if self.use_sasf:
            print("🌉 SASF Enabled!")
            bridge_dims = [dims_decoder[0], dims_decoder[1], dims_decoder[2]]
            self.sasf_convs = nn.ModuleList([
                nn.Conv2d(dim, dim, 1) for dim in bridge_dims
            ])
        else:
            self.sasf_convs = None
        self.use_wgd = kwargs.pop('use_wgd', kwargs.pop('use_wavelet', False))
        if self.use_wgd:
            print("🌊 WGD Enabled!")

        # ...existing code...

        # 可选择使用标准PatchEmbedding或FGStem增强版
        use_fgstem_embed = kwargs.get('use_fgstem_embed', True)  # 默认使用FGStem
        if use_fgstem_embed:
            self.patch_embed = FGStemEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                norm_layer=norm_layer if patch_norm else None)
        else:
            self.patch_embed = StandardPatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # build layers - 实现分层架构
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 尊重配置文件中的设置，不再强制覆盖
            layer_kwargs = kwargs.copy()
            # 如果配置文件中启用了use_phr，则应用分层策略（前两层PHR，后两层Mamba）
            # 如果配置文件use_phr=False，则全层使用VSSBlock（这样可以加载预训练权重）
            if kwargs.get('use_phr', False):
                if i_layer < 2:
                    layer_kwargs['use_phr'] = True
                else:
                    layer_kwargs['use_phr'] = False
            
            # 确保Encoder不使用EADB
            layer_kwargs['use_eadb'] = False
            
            # 选择下采样方式：小波变换 or PatchMerging
            downsample_layer = None
            if i_layer < self.num_layers - 1:
                downsample_layer = WaveletPatchMerging if self.use_wgd else PatchMerging2D

            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample_layer,
                use_checkpoint=use_checkpoint,
                **layer_kwargs,
            )
            self.layers.append(layer)

        # build decoder layers - Decoder使用EADB-Block (边缘引导融合)
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Decoder分层架构：使用EADB-Block进行边缘引导的CNN+Mamba融合
            decoder_kwargs = kwargs.copy()
            decoder_kwargs['use_eadb'] = True  # Decoder使用EADB-Block
            decoder_kwargs['use_phr'] = False
            
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                **decoder_kwargs,
            )
            self.layers_up.append(layer)

        # 根据是否使用FGStem调整最终上采样倍数
        final_scale = 2 if use_fgstem_embed else 4
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=final_scale, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//final_scale, num_classes, 1)

        # SSP Bridge和Anatomy Conditioner功能已禁用（模块不存在）
        self.use_ssp_bridge = False  # 强制禁用
        self.use_anatomy_tokens = False  # 强制禁用
        self.ssp_bridges = None
        
        # if self.use_ssp_bridge:
        #     bridge_dims = [dims_decoder[i - 1] for i in range(1, self.num_layers)]
        #     skip_dims = list(reversed(dims))[:len(bridge_dims)]
        #     self.ssp_bridges = nn.ModuleList([
        #         StateSharingBridge(dec_dim=bridge_dims[idx], skip_dim=skip_dims[idx], state_ratio=self.ssp_state_ratio)
        #         for idx in range(len(bridge_dims))
        #     ])
        # else:
        #     self.ssp_bridges = None

        # if self.use_anatomy_tokens:
        #     self.anatomy_conditioner = AnatomyConditioner(
        #         num_classes=self.num_classes,
        #         stage_dims=dims_decoder,
        #         token_dim=self.anatomy_token_dim,
        #         dropout=self.anatomy_token_dropout,
        #     )
        # else:
        #     self.anatomy_conditioner = None

        # Backward-compatible attribute aliases
        self.use_wavelet = self.use_wgd
        self.use_semantic_align = self.use_sasf
        self.semantic_align_convs = self.sasf_convs
        self.anatomy_conditioner = None

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list
    
    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                # Up Layer 0: 无跳跃连接
                x = layer_up(x)
            else:
                skip_feature = skip_list[-inx]
                
                if self.use_ssp_bridge:
                    fused = self.ssp_bridges[inx - 1](x, skip_feature)
                    x = layer_up(fused)
                elif self.use_sasf:
                    # Semantic-Aligned Skip
                    alpha = torch.sigmoid(self.sasf_convs[inx - 1](x.permute(0, 3, 1, 2)))
                    skip_aligned = alpha * skip_feature.permute(0, 3, 1, 2)
                    x = layer_up(x + skip_aligned.permute(0, 2, 3, 1))
                else:
                    x = layer_up(x + skip_feature)

            if self.use_anatomy_tokens:
                x = self.anatomy_conditioner(inx, x)

        return x
    
    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        
        return x




    


