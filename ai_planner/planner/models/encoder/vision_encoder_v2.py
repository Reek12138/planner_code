# /workspace/zhuy25@xiaopeng.com/planner_code/ai_planner/planner/models/encoder/vision_encoder_v2.py
# -*- coding: utf-8 -*-
"""
ViT4FrameEncoder V2 with RoPE (Rotary Position Embedding)

主要改进：
1. 支持 RoPE 旋转位置编码替代绝对位置编码
2. 支持输入尺寸灵活配置（如 128x128）

保持与 V1 的接口兼容，但内部使用 RoPE
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as _ckpt


def _to_b_s_3_h_w(images: torch.Tensor) -> torch.Tensor:
    """
    Accept:
      - [B,S,3,H,W]
      - [S,3,H,W]   -> treat as B=1
      - [B,3,H,W]   -> treat as S=1
    Return: [B,S,3,H,W]
    """
    if images.dim() == 5:
        return images
    if images.dim() == 4:
        if images.shape[1] == 3:
            return images.unsqueeze(0)
        if images.shape[0] == 3:
            return images.unsqueeze(0).unsqueeze(1)
        raise ValueError(f"ambiguous 4D images shape: {tuple(images.shape)}")
    raise ValueError(f"images must be 4D/5D, got {tuple(images.shape)}")


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    
    Args:
        q: query tensor [B, N, D] or [B, h, N, d]
        k: key tensor [B, N, D] or [B, h, N, d]  
        cos: cosine component [1, N, D] or [1, 1, N, d]
        sin: sine component [1, N, D] or [1, 1, N, d]
        unsqueeze_dim: dimension to unsqueeze for multi-head attention
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPEMultiheadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE (Rotary Position Embedding)
    """
    def __init__(self, dim: int, nhead: int, dropout: float):
        super().__init__()
        assert dim % nhead == 0
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
        # RoPE 缓存
        self._rope_cache = {}

    def _compute_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """计算 RoPE 的 cos/sin 缓存"""
        if seq_len in self._rope_cache:
            cached_cos, cached_sin = self._rope_cache[seq_len]
            if cached_cos.device == device and cached_cos.dtype == dtype:
                return cached_cos, cached_sin
        
        # 计算位置编码
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=dtype, device=device) / self.head_dim))
        
        pos = torch.arange(seq_len, dtype=dtype, device=device)
        freqs = torch.outer(pos, inv_freq)  # [seq_len, head_dim/2]
        
        # 扩展维度用于旋转
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        
        cos = emb.cos()
        sin = emb.sin()
        
        self._rope_cache[seq_len] = (cos, sin)
        return cos, sin

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        x: [B, N, D]
        attn_mask: optional, broadcastable to [B, nhead, N, N]
        """
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B,N,3D]
        qkv = qkv.view(B, N, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,h,N,dh]

        # Apply RoPE to q and k
        cos, sin = self._compute_rope(N, x.device, x.dtype)
        # Reshape for multi-head: [B, h, N, d]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, d]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, N, d]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,h,N,N]
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,h,N,dh]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class RoPETransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block with RoPE: LN -> RoPE-MHA -> residual -> LN -> MLP -> residual
    """
    def __init__(self, dim: int, nhead: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEMultiheadSelfAttention(dim, nhead, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class ViT4FrameEncoderV2Config:
    img_size: int = 128  # 默认 128x128，可配置
    patch_size: int = 14
    in_chans: int = 3
    num_frames: int = 4

    embed_dim: int = 1024
    out_dim: Optional[int] = None  # 默认 None -> 2*embed_dim

    nhead: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # depth:
    spatial_layers: int = 6
    temporal_layers: int = 2

    use_cls_token: bool = True
    use_time_embed: bool = True  # 时间维度仍使用绝对编码
    use_rope: bool = True  # 空间维度使用 RoPE
    init_std: float = 0.02


class ViT4FrameEncoderV2(nn.Module):
    """
    ViT 4-frame encoder with RoPE support
    
    输入:
      images: [B,S,3,H,W] (S=4)
    输出:
      feat: [B,S,N,2*embed_dim]
      patch_start_idx: int
    
    特点:
      - 空间维度使用 RoPE 位置编码
      - 时间维度仍使用可学习的位置编码
    """
    def __init__(self, cfg: ViT4FrameEncoderV2Config):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.num_frames = cfg.num_frames
        self.embed_dim = cfg.embed_dim
        self.out_dim = cfg.out_dim if cfg.out_dim is not None else (2 * cfg.embed_dim)
        self.use_cls = cfg.use_cls_token
        self.use_rope = cfg.use_rope

        assert cfg.img_size % cfg.patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = cfg.img_size // cfg.patch_size
        self.num_patches = self.grid * self.grid
        self.patch_start_idx = 1 if self.use_cls else 0

        # Activation checkpointing
        self.gradient_checkpointing = False
        self.gc_use_reentrant = False

        # Patch projection
        self.patch_proj = nn.Conv2d(
            cfg.in_chans, cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True
        )

        # CLS token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        else:
            self.cls_token = None

        # 空间位置编码：RoPE 不需要可学习参数，但如果需要可以保留一个小的 bias
        if not self.use_rope:
            n_tokens = self.num_patches + (1 if self.use_cls else 0)
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, cfg.embed_dim))
        else:
            # RoPE 模式下不需要 pos_embed，但为兼容保留空参数
            self.register_parameter('pos_embed', None)

        # 时间嵌入（跨帧使用绝对编码）
        if cfg.use_time_embed:
            self.time_embed = nn.Parameter(torch.zeros(1, cfg.num_frames, 1, cfg.embed_dim))
        else:
            self.time_embed = None

        self.pos_drop = nn.Dropout(cfg.dropout)

        # Spatial blocks with RoPE
        self.spatial_blocks = nn.ModuleList([
            RoPETransformerBlock(cfg.embed_dim, cfg.nhead, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.spatial_layers)
        ])

        # Temporal blocks (跨帧使用标准 attention，时间维度短不需要 RoPE)
        self.temporal_blocks = nn.ModuleList([
            RoPETransformerBlock(cfg.embed_dim, cfg.nhead, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.temporal_layers)
        ])

        # Final projection
        self.out_proj = nn.Linear(cfg.embed_dim, self.out_dim)

        self._init_weights(cfg.init_std)

    def gradient_checkpointing_enable(self, enabled: bool = True, use_reentrant: bool = False):
        self.gradient_checkpointing = bool(enabled)
        self.gc_use_reentrant = bool(use_reentrant)
        return self

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpointing_enable(False)

    def _init_weights(self, std: float):
        if self.use_cls:
            nn.init.normal_(self.cls_token, std=std)
        if not self.use_rope and hasattr(self, 'pos_embed') and self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=std)
        if self.time_embed is not None:
            nn.init.normal_(self.time_embed, std=std)

        nn.init.xavier_uniform_(self.patch_proj.weight)
        if self.patch_proj.bias is not None:
            nn.init.zeros_(self.patch_proj.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*S,3,H,W] -> tokens: [B*S,P,D]
        """
        x = self.patch_proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def _run_block(self, blk: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return _ckpt(blk, x, use_reentrant=self.gc_use_reentrant)
        return blk(x)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
        images = _to_b_s_3_h_w(images)
        B, S, C, H, W = images.shape
        if S != self.num_frames:
            raise ValueError(f"expected S={self.num_frames} frames, got S={S}")
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"expected H=W={self.img_size}, got H={H}, W={W}")

        # Patch embed per frame
        x = images.reshape(B * S, C, H, W)
        tok = self._patchify(x)  # [B*S, P, D]

        # Add CLS per frame
        if self.use_cls:
            cls = self.cls_token.expand(B * S, -1, -1)
            tok = torch.cat([cls, tok], dim=1)

        # Add spatial pos emb (only if not using RoPE)
        if not self.use_rope and self.pos_embed is not None:
            tok = tok + self.pos_embed
        tok = self.pos_drop(tok)

        # Spatial blocks with RoPE
        for blk in self.spatial_blocks:
            tok = self._run_block(blk, tok)

        # Reshape back to [B,S,N,D]
        tok = tok.view(B, S, tok.shape[1], tok.shape[2])

        # Temporal blocks
        tok_t = tok.permute(0, 2, 1, 3).contiguous()  # [B,N,S,D]

        if self.time_embed is not None:
            tok_t = tok_t + self.time_embed.permute(0, 2, 1, 3)

        tok_t = tok_t.view(B * tok_t.shape[1], S, tok_t.shape[3])  # [B*N, S, D]

        for blk in self.temporal_blocks:
            tok_t = self._run_block(blk, tok_t)

        # Reshape back
        tok_t = tok_t.view(B, -1, S, self.embed_dim)
        tok = tok_t.permute(0, 2, 1, 3).contiguous()

        # Output projection
        feat = self.out_proj(tok)

        return feat, self.patch_start_idx


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ViT4FrameEncoderV2Config(
        img_size=128,
        patch_size=14,
        num_frames=4,
        embed_dim=256,
        out_dim=None,
        nhead=8,
        spatial_layers=4,
        temporal_layers=2,
        dropout=0.0,
        use_cls_token=True,
        use_time_embed=True,
        use_rope=True,
    )
    enc = ViT4FrameEncoderV2(cfg).to(device)
    enc.gradient_checkpointing_enable(True, use_reentrant=False)
    enc.train()

    B, S = 2, 4
    x = torch.randn(B, S, 3, 128, 128, device=device, requires_grad=True)
    feat, patch_start_idx = enc(x)
    loss = feat.mean()
    loss.backward()

    print("feat:", feat.shape, "patch_start_idx:", patch_start_idx)
    print("grad ok:", enc.patch_proj.weight.grad is not None)
