# /workspace/zhuy25@xiaopeng.com/planner_code/ai_planner/planner/models/modeling/vit_planner_v2.py
# -*- coding: utf-8 -*-
"""
VelocityPlannerModel V2

基于 V1 的改进：
1. 使用 ViT4FrameEncoderV2（支持 RoPE）
2. 使用 VGGTConnectorV2（支持 2x2 token 压缩）
3. 默认输入尺寸改为 128x128

保持接口与 V1 兼容，Decoder 和 Head 逻辑完全不变
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint as _ckpt

# Import V2 components
from ..encoder.vision_encoder_v2 import ViT4FrameEncoderV2, ViT4FrameEncoderV2Config
from ..connector.connector_v2 import VGGTConnectorV2

# Import unchanged components from V1
from ..decoder.traj_decoder import PlannerDecoder, VelocityHead, VelocityHeadTanh, VelocityHeadUnit


class StateFourierEmbedding(nn.Module):
    """
    将 state（速度+位姿等）做 Fourier features，再投影到 out_dim
    与 V1 保持一致
    """
    def __init__(self, in_dim: int, num_freq: int = 16, out_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.num_freq = num_freq
        self.out_dim = out_dim

        self.register_buffer(
            "freqs",
            (2 ** torch.arange(num_freq, dtype=torch.float32)),
            persistent=False,
        )

        fourier_dim = in_dim * 2 * num_freq
        self.proj = nn.Linear(fourier_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - [B, in_dim]
          - [B, S, in_dim]
        return:
          - [B, out_dim]
          - [B, S, out_dim]
        """
        if x.dim() not in (2, 3):
            raise ValueError(f"state must be [B,D] or [B,S,D], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"state last dim must be {self.in_dim}, got {x.shape[-1]}")

        x = x.unsqueeze(-1)
        y = x * self.freqs.to(x.device) * (2.0 * math.pi)
        emb = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        emb = emb.flatten(-2)
        return self.proj(emb)


@dataclass
class VelocityPlannerConfigV2:
    # ---------- state ----------
    state_dim: int = 7
    fourier_freqs: int = 16

    # ---------- decoder dims ----------
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_ff: int = 1024
    dropout: float = 0.1

    # ---------- vision encoder (ViT 4-frame with RoPE) ----------
    img_size: int = 128  # 默认 128x128
    patch_size: int = 14

    # 注意：字段名保留 vggt_embed_dim 用于兼容
    vggt_embed_dim: int = 256

    # ViT 超参
    vit_num_frames: int = 4
    vit_nhead: int = 8
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.0
    vit_spatial_layers: int = 6
    vit_temporal_layers: int = 2
    vit_use_cls_token: bool = True
    vit_use_time_embed: bool = True
    vit_use_rope: bool = True  # V2: 启用 RoPE

    # ---------- connector V2 ----------
    max_patches: Optional[int] = None
    
    # 2x2 压缩参数
    use_2x2_compression: bool = True  # V2: 默认启用 2x2 压缩
    compression_hidden_dim: Optional[int] = None
    compression_num_layers: int = 2

    # ---------- output ----------
    out_dim: int = 3

    # ---------- freeze ----------
    freeze_vggt: bool = False
    encoder_eval_mode: bool = True
    encoder_no_grad: bool = True

    # ---------- activation checkpointing ----------
    activation_checkpointing: bool = False
    activation_ckpt_use_reentrant: bool = False
    activation_ckpt_target: str = "encoder"


class VelocityPlannerModelV2(nn.Module):
    """
    VelocityPlannerModel V2
    
    输入：
      images: [B,S,3,H,W] 或 [S,3,H,W]
      state:  [B,state_dim] 或 [B,S,state_dim]

    输出：
      v_pred: [B,out_dim]
      
    改进：
      - Encoder: 使用 RoPE 位置编码
      - Connector: 支持 2x2 token 压缩
      - 默认输入: 128x128
    """
    def __init__(self, cfg: VelocityPlannerConfigV2):
        super().__init__()
        self.cfg = cfg

        # 1) State Fourier -> d_model
        self.state_embed = StateFourierEmbedding(
            in_dim=cfg.state_dim,
            num_freq=cfg.fourier_freqs,
            out_dim=cfg.d_model,
        )

        # 2) ViT 4-frame encoder V2 (with RoPE)
        vit_cfg = ViT4FrameEncoderV2Config(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            num_frames=cfg.vit_num_frames,
            embed_dim=cfg.vggt_embed_dim,
            out_dim=None,  # -> 2*E
            nhead=cfg.vit_nhead,
            mlp_ratio=cfg.vit_mlp_ratio,
            dropout=cfg.vit_dropout,
            spatial_layers=cfg.vit_spatial_layers,
            temporal_layers=cfg.vit_temporal_layers,
            use_cls_token=cfg.vit_use_cls_token,
            use_time_embed=cfg.vit_use_time_embed,
            use_rope=cfg.vit_use_rope,
        )
        self.vggt = ViT4FrameEncoderV2(vit_cfg)

        # encoder 输出 dim = 2*E
        vggt_out_dim = 2 * cfg.vggt_embed_dim

        # 3) Connector V2: with 2x2 compression support
        # 计算 grid 尺寸
        grid = cfg.img_size // cfg.patch_size
        # 注意：如果 img_size 不能被 patch_size 整除，需要处理
        # 这里假设 img_size 是 patch_size 的倍数或接近
        # 对于 128/14 ≈ 9.14，我们向下取整为 9
        grid_h = cfg.img_size // cfg.patch_size
        grid_w = cfg.img_size // cfg.patch_size
        
        self.connector = VGGTConnectorV2(
            in_dim=vggt_out_dim,
            d_model=cfg.d_model,
            use_last_frame=True,
            keep_time=False,
            max_patches=cfg.max_patches,
            proj_hidden_dim=None,
            proj_num_layers=2,
            dropout=0.0,
            activation="gelu",
            use_layernorm=True,
            # 2x2 compression params
            use_2x2_compression=cfg.use_2x2_compression,
            grid_h=grid_h,
            grid_w=grid_w,
            compression_hidden_dim=cfg.compression_hidden_dim,
            compression_num_layers=cfg.compression_num_layers,
        )

        # 4) Decoder + velocity head (与 V1 完全一致)
        self.decoder = PlannerDecoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_ff=cfg.dim_ff,
            dropout=cfg.dropout,
            use_self_attn=False,
            pre_norm=True,
        )
        self.head = VelocityHeadUnit(d_model=cfg.d_model, out_dim=cfg.out_dim)

        # 5) Freeze ONLY encoder
        if cfg.freeze_vggt:
            for p in self.vggt.parameters():
                p.requires_grad = False

        if cfg.freeze_vggt and cfg.encoder_eval_mode:
            self.vggt.eval()

        # 6) Activation checkpointing
        if cfg.activation_checkpointing:
            self.gradient_checkpointing_enable(
                enabled=True,
                use_reentrant=cfg.activation_ckpt_use_reentrant,
                target=cfg.activation_ckpt_target,
            )

    # -------------------------
    # public API: checkpointing
    # -------------------------
    def gradient_checkpointing_enable(
        self,
        enabled: bool = True,
        use_reentrant: bool = False,
        target: str = "encoder",
    ):
        self.cfg.activation_checkpointing = bool(enabled)
        self.cfg.activation_ckpt_use_reentrant = bool(use_reentrant)
        self.cfg.activation_ckpt_target = str(target)

        if hasattr(self.vggt, "gradient_checkpointing_enable"):
            try:
                self.vggt.gradient_checkpointing_enable(enabled=bool(enabled), use_reentrant=bool(use_reentrant))
                self._fallback_ckpt_encoder = False
            except TypeError:
                self.vggt.gradient_checkpointing_enable(bool(enabled))
                self._fallback_ckpt_encoder = False
        elif hasattr(self.vggt, "set_gradient_checkpointing"):
            self.vggt.set_gradient_checkpointing(bool(enabled))
            self._fallback_ckpt_encoder = False
        else:
            self._fallback_ckpt_encoder = bool(enabled)

        return self

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpointing_enable(enabled=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_vggt and self.cfg.encoder_eval_mode:
            self.vggt.eval()
        return self

    # -------------------------
    # internal helpers
    # -------------------------
    def _encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self.cfg.freeze_vggt and self.cfg.encoder_no_grad:
            with torch.no_grad():
                return self.vggt(images)

        if (
            self.cfg.activation_checkpointing
            and getattr(self, "_fallback_ckpt_encoder", False)
            and self.training
        ):
            def _vggt_forward(x: torch.Tensor):
                feat, patch_start_idx = self.vggt(x)
                patch_t = torch.tensor([int(patch_start_idx)], device=feat.device, dtype=torch.int64)
                return feat, patch_t

            feat, patch_t = _ckpt(_vggt_forward, images, use_reentrant=self.cfg.activation_ckpt_use_reentrant)
            patch_start_idx = int(patch_t[0].item())
            return feat, patch_start_idx

        return self.vggt(images)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        return_attn: bool = False,
    ):
        # ---- 1) encode images (ViT V2 with RoPE) ----
        vggt_feat, patch_start_idx = self._encode_images(images)

        # ---- 2) build KV from patch tokens (with 2x2 compression) ----
        kv = self.connector(vggt_feat, patch_start_idx)

        # PlannerDecoder 期望 img_feat: [B,S,P,D]
        img_feat_for_decoder = kv.unsqueeze(1)

        # ---- 3) build Q from state ----
        state_emb = self.state_embed(state)

        if state_emb.dim() == 3:
            q_embed = state_emb[:, -1, :]
        else:
            q_embed = state_emb

        # ---- 4) decode ----
        if return_attn:
            q_out, attn_list = self.decoder(
                q_embed=q_embed,
                img_feat=img_feat_for_decoder,
                kv_key_padding_mask=None,
                return_attn=True,
            )
            v_pred = self.head(q_out)
            return v_pred, attn_list

        q_out = self.decoder(
            q_embed=q_embed,
            img_feat=img_feat_for_decoder,
            kv_key_padding_mask=None,
            return_attn=False,
        )
        v_pred = self.head(q_out)
        return v_pred


# ----------------------------
# Minimal quick test
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = VelocityPlannerConfigV2(
        state_dim=7,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=1024,
        dropout=0.1,
        img_size=128,  # V2: 128x128
        patch_size=14,

        # encoder V2 with RoPE
        vggt_embed_dim=256,
        vit_num_frames=4,
        vit_nhead=8,
        vit_spatial_layers=6,
        vit_temporal_layers=2,
        vit_dropout=0.0,
        vit_use_cls_token=True,
        vit_use_time_embed=True,
        vit_use_rope=True,  # 启用 RoPE

        # connector V2 with 2x2 compression
        use_2x2_compression=True,
        compression_hidden_dim=512,
        compression_num_layers=2,

        max_patches=None,
        out_dim=3,

        freeze_vggt=False,
        encoder_eval_mode=True,
        encoder_no_grad=True,

        activation_checkpointing=True,
        activation_ckpt_use_reentrant=False,
        activation_ckpt_target="encoder",
    )

    model = VelocityPlannerModelV2(cfg).to(device)
    model.train()

    model.gradient_checkpointing_enable(True, use_reentrant=False, target="encoder")

    B, S = 2, 4
    H, W = 128, 128  # V2: 128x128
    images = torch.rand(B, S, 3, H, W, device=device)
    state = torch.rand(B, S, cfg.state_dim, device=device)

    v = model(images, state)
    print("v_pred shape:", tuple(v.shape))

    # sanity check
    n_frozen = sum(1 for p in model.vggt.parameters() if not p.requires_grad)
    n_trainable_conn = sum(1 for p in model.connector.parameters() if p.requires_grad)
    n_trainable_dec = sum(1 for p in model.decoder.parameters() if p.requires_grad)
    print("encoder frozen params:", n_frozen)
    print("connector trainable params:", n_trainable_conn)
    print("decoder trainable params:", n_trainable_dec)

    def count_params(m: nn.Module, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return sum(p.numel() for p in m.parameters())

    print("params encoder(total):", count_params(model.vggt))
    print("params decoder(total):", count_params(model.decoder))
    print("params connector(total):", count_params(model.connector))
    print("params head(total):", count_params(model.head))
