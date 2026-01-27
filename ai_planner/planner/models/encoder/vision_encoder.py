# /workspace/zhuy25@xiaopeng.com/planner_code/ai_planner/planner/models/encoder/vision_encoder.py
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# NEW: activation checkpointing
from torch.utils.checkpoint import checkpoint as _ckpt


# ----------------------------
# utils
# ----------------------------
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
        # could be [S,3,H,W] or [B,3,H,W]
        # assume [S,3,H,W] if first dim small and channel==3 at dim1
        if images.shape[1] == 3:
            return images.unsqueeze(0)  # [1,S,3,H,W]
        if images.shape[0] == 3:
            return images.unsqueeze(0).unsqueeze(1)  # [1,1,3,H,W]
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


class MultiheadSelfAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        x: [B, N, D]
        attn_mask: optional, broadcastable to [B, nhead, N, N] with 0/-inf style masks
        """
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B,N,3D]
        qkv = qkv.view(B, N, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,h,N,dh]

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


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block: LN -> MHA -> residual -> LN -> MLP -> residual
    """
    def __init__(self, dim: int, nhead: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, nhead, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------
# 4-frame ViT encoder
# ----------------------------
@dataclass
class ViT4FrameEncoderConfig:
    img_size: int = 518
    patch_size: int = 14
    in_chans: int = 3
    num_frames: int = 4

    embed_dim: int = 1024      # 你想对齐 vggt_embed_dim -> 输出会是 2*embed_dim
    out_dim: Optional[int] = None  # 默认 None -> 2*embed_dim

    nhead: int = 16            # embed_dim=1024 时 16 头比较常见（每头 64d）
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # depth:
    spatial_layers: int = 6    # 帧内 attention 层数
    temporal_layers: int = 2   # 跨帧 attention 层数（每个 patch 位置沿时间做 attn）

    use_cls_token: bool = True
    use_time_embed: bool = True
    init_std: float = 0.02


class ViT4FrameEncoder(nn.Module):
    """
    输入:
      images: [B,S,3,H,W] (S=4)
    输出:
      feat: [B,S,N,2*embed_dim]，其中 N = 1+P (若 use_cls_token=True)
      patch_start_idx: int (use_cls_token -> 1 else 0)

    结构:
      1) PatchEmbed per frame -> tokens
      2) Spatial transformer blocks: within each frame (帧内 attn)
      3) Temporal transformer blocks: for each token index (including CLS if enabled),
         apply attention across frames (跨帧 attn)
    """
    def __init__(self, cfg: ViT4FrameEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.num_frames = cfg.num_frames
        self.embed_dim = cfg.embed_dim
        self.out_dim = cfg.out_dim if cfg.out_dim is not None else (2 * cfg.embed_dim)
        self.use_cls = cfg.use_cls_token

        assert cfg.img_size % cfg.patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = cfg.img_size // cfg.patch_size
        self.num_patches = self.grid * self.grid
        self.patch_start_idx = 1 if self.use_cls else 0

        # NEW: activation checkpointing switches (default OFF)
        self.gradient_checkpointing = False
        self.gc_use_reentrant = False

        # Patch projection (conv as linear patch embed)
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

        # Spatial pos embedding (shared across frames)
        # shape: [1, 1+P, D] or [1,P,D]
        n_tokens = self.num_patches + (1 if self.use_cls else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, cfg.embed_dim))

        # Time embedding (added before temporal blocks)
        if cfg.use_time_embed:
            self.time_embed = nn.Parameter(torch.zeros(1, cfg.num_frames, 1, cfg.embed_dim))  # [1,S,1,D]
        else:
            self.time_embed = None

        self.pos_drop = nn.Dropout(cfg.dropout)

        # Spatial blocks (帧内)
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.nhead, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.spatial_layers)
        ])

        # Temporal blocks (跨帧)：沿时间维度做 self-attn
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.nhead, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.temporal_layers)
        ])

        # Final projection to match "2*embed_dim" style output expected by your connector
        self.out_proj = nn.Linear(cfg.embed_dim, self.out_dim)

        self._init_weights(cfg.init_std)

    # NEW: public API (huggingface-like)
    def gradient_checkpointing_enable(self, enabled: bool = True, use_reentrant: bool = False):
        """
        Enable/disable activation checkpointing (block-level).
        This does NOT change model architecture; it only wraps blocks with torch.utils.checkpoint during training.
        """
        self.gradient_checkpointing = bool(enabled)
        self.gc_use_reentrant = bool(use_reentrant)
        return self

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpointing_enable(False)

    def _init_weights(self, std: float):
        if self.use_cls:
            nn.init.normal_(self.cls_token, std=std)
        nn.init.normal_(self.pos_embed, std=std)
        if self.time_embed is not None:
            nn.init.normal_(self.time_embed, std=std)

        # conv/linear init: match common ViT init
        nn.init.xavier_uniform_(self.patch_proj.weight)
        if self.patch_proj.bias is not None:
            nn.init.zeros_(self.patch_proj.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # blocks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # already covered by some; safe
                pass

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*S,3,H,W] -> tokens: [B*S,P,D]
        """
        x = self.patch_proj(x)                 # [B*S, D, Gh, Gw]
        x = x.flatten(2).transpose(1, 2)       # [B*S, P, D]
        return x

    def _run_block(self, blk: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Helper: run a TransformerBlock with optional activation checkpointing.
        """
        if self.gradient_checkpointing and self.training:
            # checkpoint requires that at least one input has requires_grad=True (usually true in training)
            return _ckpt(blk, x, use_reentrant=self.gc_use_reentrant)
        return blk(x)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
        images = _to_b_s_3_h_w(images)  # [B,S,3,H,W]
        B, S, C, H, W = images.shape
        if S != self.num_frames:
            raise ValueError(f"expected S={self.num_frames} frames, got S={S}")
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"expected H=W={self.img_size}, got H={H}, W={W}")

        # ---- 1) patch embed per frame ----
        x = images.reshape(B * S, C, H, W)
        tok = self._patchify(x)  # [B*S, P, D]

        # add CLS per frame
        if self.use_cls:
            cls = self.cls_token.expand(B * S, -1, -1)  # [B*S,1,D]
            tok = torch.cat([cls, tok], dim=1)          # [B*S,1+P,D]

        # add spatial pos emb (same for each frame)
        tok = tok + self.pos_embed
        tok = self.pos_drop(tok)

        # ---- 2) spatial blocks (帧内 attention) ----
        for blk in self.spatial_blocks:
            tok = self._run_block(blk, tok)  # [B*S, N, D]

        # reshape back to [B,S,N,D]
        tok = tok.view(B, S, tok.shape[1], tok.shape[2])  # [B,S,N,D]

        # ---- 3) temporal blocks (跨帧 attention) ----
        tok_t = tok.permute(0, 2, 1, 3).contiguous()  # [B,N,S,D]

        if self.time_embed is not None:
            tok_t = tok_t + self.time_embed.permute(0, 2, 1, 3)  # [1,1,S,D] broadcast

        tok_t = tok_t.view(B * tok_t.shape[1], S, tok_t.shape[3])  # [B*N, S, D]

        for blk in self.temporal_blocks:
            tok_t = self._run_block(blk, tok_t)  # [B*N, S, D]

        # reshape back to [B,N,S,D] then to [B,S,N,D]
        tok_t = tok_t.view(B, -1, S, self.embed_dim)            # [B,N,S,D]
        tok = tok_t.permute(0, 2, 1, 3).contiguous()            # [B,S,N,D]

        # ---- 4) output projection to match connector in_dim ----
        feat = self.out_proj(tok)  # [B,S,N,out_dim] where out_dim defaults to 2*embed_dim

        return feat, self.patch_start_idx


# ----------------------------
# minimal quick test
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ViT4FrameEncoderConfig(
        img_size=518,
        patch_size=14,
        num_frames=4,
        embed_dim=128,
        out_dim=None,
        nhead=8,
        spatial_layers=4,
        temporal_layers=2,
        dropout=0.0,
        use_cls_token=True,
        use_time_embed=True,
    )
    enc = ViT4FrameEncoder(cfg).to(device)

    # enable checkpointing for test
    enc.gradient_checkpointing_enable(True, use_reentrant=False)
    enc.train()

    B, S = 2, 4
    x = torch.randn(B, S, 3, 518, 518, device=device, requires_grad=True)
    feat, patch_start_idx = enc(x)
    loss = feat.mean()
    loss.backward()

    print("feat:", feat.shape, "patch_start_idx:", patch_start_idx)
    print("grad ok:", enc.patch_proj.weight.grad is not None)
