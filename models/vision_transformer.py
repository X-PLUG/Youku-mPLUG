# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from torch.nn.parameter import Parameter
from timm.models.registry import register_model

from torch import Tensor, Size
from typing import Union, List
import numbers

from einops import rearrange


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


_shape_t = Union[int, List[int], Size]


class LayerNormWithForceFP32(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: _shape_t
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(input)

    def extra_repr(self) -> Tensor:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rel_pos_bias = None
        self.qk_float = True

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3    # (2*14-1) * (2*14-1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        if self.qk_float:
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias().type_as(attn)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias.type_as(attn)
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, postnorm=False, add_temporal_module=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, 
            attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


        self.temporal_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, 
            attn_head_dim=attn_head_dim)
        self.temporal_ln = norm_layer(dim)
        self.temporal_fc = nn.Linear(dim, dim)

        # if init_values is not None and init_values > 0:
        #     self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        #     self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        # else:
        #     self.gamma_1, self.gamma_2 = None, None
        # self.postnorm = postnorm

    def forward(self, x, cls_token, rel_pos_bias=None, attn_mask=None):
        B, T, N = x.shape[:3]

        ## Temporal
        xt = rearrange(x, 'b t n m -> (b n) t m')
        xt = self.temporal_attn(self.temporal_ln(xt))
        xt = rearrange(xt, '(b n) t m -> b (n t) m', b=B)
        xt = self.temporal_fc(xt)
        xt = rearrange(x, 'b t n m -> b (n t) m') + xt

        ## Spatial
        init_cls_token = cls_token.unsqueeze(1)
        cls_token = init_cls_token.repeat(1, T, 1)
        cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, 'b (n t) m ->  (b t) n m', n=N, t=T)
        xs = torch.cat((cls_token, xs), 1)
        xs = self.attn(self.norm1(xs))
        
        ### Taking care of CLS token
        cls_token = xs[:, 0, :]
        cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
        cls_token = torch.mean(cls_token, 1, True) ## averaging for every frame
        xs = xs[:, 1:, :]
        xs = rearrange(xs, '(b t) n m -> b (n t) m',b=B,n=N,t=T)

        # x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = torch.cat((init_cls_token, xt), 1) + torch.cat((cls_token, xs), 1)
        x = x + self.mlp(self.norm2(x))

        cls_token, x = x[:, 0, :], x[:, 1:, ]
        x = rearrange(x, 'b (n t) m -> b t n m', t=T)
        return x, cls_token

        # if self.gamma_1 is None:
        #     if self.postnorm:
        #         x = x + self.drop_path(
        #             self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
        #         x = x + self.drop_path(self.norm2(self.mlp(x)))
        #     else:
        #         x = x + self.drop_path(
        #             self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
        #         x = x + self.drop_path(self.mlp(self.norm2(x)))
        # else:
        #     if self.postnorm:
        #         x = x + self.drop_path(
        #             self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
        #         x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        #     else:
        #         x = x + self.drop_path(
        #             self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
        #         x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PlainBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, postnorm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, 
            attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        # if self.gamma_1 is None:
        #     if self.postnorm:
        #         x = x + self.drop_path(
        #             self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
        #         x = x + self.drop_path(self.norm2(self.mlp(x)))
        #     else:
        #         x = x + self.drop_path(
        #             self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
        #         x = x + self.drop_path(self.mlp(self.norm2(x)))
        # else:
        #     if self.postnorm:
        #         x = x + self.drop_path(
        #             self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
        #         x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        #     else:
        #         x = x + self.drop_path(
        #             self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
        #         x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x + self.drop_path(
                    self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AttentionPool(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, postnorm=False, kdim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.normk = norm_layer(dim)
        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     attn_drop=attn_drop, proj_drop=drop, window_size=window_size, 
        #     attn_head_dim=attn_head_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, bias=qkv_bias, add_bias_kv=qkv_bias, kdim=kdim, vdim=kdim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, k, rel_pos_bias=None, attn_mask=None):
        x = self.norm1(x).permute(1, 0, 2)
        k = self.normk(k).permute(1, 0, 2)
        x = x + self.drop_path(self.attn(x, k, k, need_weights=False)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(1, 0, 2)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class TimeSformer(nn.Module):
    def __init__(self, img_size=224, num_frames=4, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, 
                 grad_ckpt=False, stop_grad_conv1=False, postnorm=False, clip_model=False, add_temporal_module=False, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=not clip_model)
        num_patches = self.patch_embed.num_patches
        self.num_frames = num_frames

        if clip_model:
            self.norm_pre = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
                add_temporal_module=add_temporal_module
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if 'Block' in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1
        
        if postnorm:
            self._init_respostnorm()

        self.grad_ckpt = grad_ckpt
        self.stop_grad_conv1 = stop_grad_conv1

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temporal_embed', 'pos_embed', 'cls_token'}

    def get_final_patch_size(self):
        return self.patch_embed.patch_size

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x):
        batch_size, C, T, H, W = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = rearrange(x, '(b t) n c -> b (t n) c', b = batch_size)

        if self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1 

        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.total_pos_embed
        # x = self.pos_drop(x)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patch_embed.num_patches, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
        x = x + total_pos_embed
        x = self.pos_drop(x)

        if hasattr(self, "norm_pre"):
            x = self.norm_pre(x)
        
        cls_token, x = x[:, 0, :], x[:, 1:, ]
        x = rearrange(x, 'b (t n) m -> b t n m', t=T)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        if self.grad_ckpt:
            for i in range(len(self.blocks)):
                x, cls_token = torch.utils.checkpoint.checkpoint(self.blocks[i], x, cls_token, rel_pos_bias)
        else:
            for blk in self.blocks:
                x, cls_token = blk(x, cls_token, rel_pos_bias=rel_pos_bias)

        x = rearrange(x, 'b t n c -> b (t n) c')
        cls_token = cls_token.unsqueeze(1)
        x = torch.cat([cls_token, x], dim = 1)
        x = self.norm(x)

        return x

    def forward(self, image_input):
        image_features = self.forward_features(image_input)            
        pooled_image_features = image_features[:, 0]
        return pooled_image_features, image_features


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, 
                 grad_ckpt=False, stop_grad_conv1=False, postnorm=False, clip_model=False, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=not clip_model)
        num_patches = self.patch_embed.num_patches

        if clip_model:
            self.norm_pre = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PlainBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
        if postnorm:
            self._init_respostnorm()

        self.grad_ckpt = grad_ckpt
        self.stop_grad_conv1 = stop_grad_conv1

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_final_patch_size(self):
        return self.patch_embed.patch_size

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1 

        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if hasattr(self, "norm_pre"):
            x = self.norm_pre(x)
        
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if self.grad_ckpt:
            for i in range(len(self.blocks)):
                x = torch.utils.checkpoint.checkpoint(self.blocks[i], x, rel_pos_bias)
        else:
            for blk in self.blocks:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)

        return x

    def forward(self, image_input):
        image_features = self.forward_features(image_input)            
        pooled_image_features = image_features[:, 0]
        return pooled_image_features, image_features


def _convert_pretrained_vit(vit_pretrained_weights):
    for key in list(vit_pretrained_weights.keys()):
        if 'qkv.bias' in key:
            q, k, v = vit_pretrained_weights[key].chunk(3)
            vit_pretrained_weights[key.replace('qkv.bias', 'q_bias')] = q
            vit_pretrained_weights[key.replace('qkv.bias', 'v_bias')] = v
            del vit_pretrained_weights[key]
        if 'head' in key:
            del vit_pretrained_weights[key]
    return vit_pretrained_weights


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    # _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    orig = posemb_grid.dtype
    posemb_grid = F.interpolate(posemb_grid.float(), size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.to(orig)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def resize_temporal_embed(posemb, posemb_new, mode='interpolate'):
    ntok_new = posemb_new.shape[1]
    ntok_old = posemb.shape[1]
    if mode == 'padding':
        if ntok_old <= ntok_new:
            posemb_new[:, :ntok_old] = posemb
        else:
            posemb_new = posemb[:, :ntok_new]
    else:
        posemb = posemb.permute(0, 2, 1) # [1, d, num_frames]
        posemb_new = F.interpolate(posemb, ntok_new, mode="linear") # [1, d, num_frames]
        posemb_new = posemb_new.permute(0, 2, 1) # [1, num_frames, d]
    return posemb_new


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)