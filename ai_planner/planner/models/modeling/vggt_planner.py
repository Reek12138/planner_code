# /workspace/zhuy25@xiaopeng.com/planner_code/ai_planner/planner/models/modeling/velocity_planner_model.py
# -*- coding: utf-8 -*-

"""
VelocityPlannerModel

串联：
  VGGTFrameEncoder -> VGGTConnector(patch tokens) -> Transformer Decoder(Q=state, KV=image) -> VelocityHead

约定：
- d_model = 256（由 Fourier state embedding 输出对齐）
- KV：使用 vggt_feat 的 patch tokens（P 维从 patch_start_idx 开始），只取最后一帧 S=-1
- Q：state（目标速度+位姿等） -> Fourier -> [B,256] -> [B,1,256]
- 输出：速度向量（默认 3 维）

冻结策略（本文件新增接口）：
- 仅支持冻结 encoder（VGGTFrameEncoder）：cfg.freeze_vggt=True
- connector / decoder / head 始终可训练（不冻结）
- 冻结时：
  - encoder 参数 requires_grad=False
  - encoder 维持 eval()（即使外部调用 model.train()）
  - encoder forward 可选 no_grad（省显存）：cfg.encoder_no_grad=True
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..encoder.vggt_encoder import VGGTFrameEncoder
from ..connector.connector import VGGTConnector
from ..decoder.traj_decoder import PlannerDecoder, VelocityHead, VelocityHeadTanh


class StateFourierEmbedding(nn.Module):
    """
    将 state（速度+位姿等）做 Fourier features，再投影到 out_dim（这里默认 256 作为 d_model）
    """
    def __init__(self, in_dim: int, num_freq: int = 16, out_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.num_freq = num_freq
        self.out_dim = out_dim

        # 频率：1,2,4,... 也可改成 linspace/logspace
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

        # [..., in_dim, 1]
        x = x.unsqueeze(-1)
        y = x * self.freqs.to(x.device) * (2.0 * math.pi)  # broadcast
        emb = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)  # [..., in_dim, 2*num_freq]
        emb = emb.flatten(-2)  # [..., in_dim*2*num_freq]
        return self.proj(emb)


@dataclass
class VelocityPlannerConfig:
    # ---------- state ----------
    state_dim: int = 7           # 例如 [vx,vy,vz,x,y,z,yaw] => 7（你实际用多少改多少）
    fourier_freqs: int = 16

    # ---------- decoder dims ----------
    d_model: int = 256           # 你要求：Fourier 后 256 作为 d_model
    nhead: int = 8               # 256/8=32 OK
    num_layers: int = 4          # 推荐起步 4；更强可改 6
    dim_ff: int = 1024           # d_model=256 推荐 1024 或 1536
    dropout: float = 0.1

    # ---------- vggt encoder ----------
    img_size: int = 518
    patch_size: int = 14
    vggt_embed_dim: int = 1024   # Aggregator embed_dim；输出特征维 = 2*embed_dim
    pretrained_agg_path: Optional[str] = None
    strict_load: bool = True
    rebuild_from_cfg: bool = True

    # ---------- connector ----------
    max_patches: Optional[int] = None  # 想省显存可设 512/768 等；None 表示全用

    # ---------- output ----------
    out_dim: int = 3            # 输出速度维度（vx,vy,vz）

    # ---------- freeze ----------
    freeze_vggt: bool = False            # ✅ 仅冻结 VGGTFrameEncoder（encoder）
    encoder_eval_mode: bool = True       # 冻结时，encoder 是否强制 eval()（关闭 dropout / BN）
    encoder_no_grad: bool = True         # 冻结时，encoder forward 是否用 no_grad（省显存）


class VelocityPlannerModel(nn.Module):
    """
    输入：
      images: [B,S,3,H,W] 或 [S,3,H,W]
      state:  [B,state_dim] 或 [B,S,state_dim]

    输出：
      v_pred: [B,out_dim]
      (可选) attn_list: 每层 cross-attn 权重 [B, Nq, Nk]
    """
    def __init__(self, cfg: VelocityPlannerConfig):
        super().__init__()
        self.cfg = cfg

        # 1) State Fourier -> d_model
        self.state_embed = StateFourierEmbedding(
            in_dim=cfg.state_dim,
            num_freq=cfg.fourier_freqs,
            out_dim=cfg.d_model,
        )

        # 2) VGGT encoder
        self.vggt = VGGTFrameEncoder(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.vggt_embed_dim,
            pretrained_agg_path=cfg.pretrained_agg_path,
            map_location="cpu",
            strict=cfg.strict_load,
            rebuild_from_cfg=cfg.rebuild_from_cfg,
        )
        vggt_out_dim = 2 * cfg.vggt_embed_dim  # feat last dim = 2C

        # 3) Connector: patch tokens -> d_model, last frame only
        self.connector = VGGTConnector(
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
        )

        # 4) Decoder + velocity head
        self.decoder = PlannerDecoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_ff=cfg.dim_ff,
            dropout=cfg.dropout,
            use_self_attn=False,   # Nq=1 时 self-attn 贡献很小，先关掉更省
            pre_norm=True,
        )
        self.head = VelocityHead(d_model=cfg.d_model, out_dim=cfg.out_dim)
        # self.head = VelocityHeadTanh(d_model=cfg.d_model, out_dim=cfg.out_dim)

        # 5) Freeze ONLY encoder (VGGTFrameEncoder). Connector/Decoder/Head keep trainable.
        if cfg.freeze_vggt:
            for p in self.vggt.parameters():
                p.requires_grad = False

        # 冻结时通常希望 encoder 进入 eval（dropout/BN 固定）
        if cfg.freeze_vggt and cfg.encoder_eval_mode:
            self.vggt.eval()

    def train(self, mode: bool = True):
        """
        覆盖 train()：当冻结 encoder 时，确保外部 model.train() 不会把 vggt 切回 train()
        """
        super().train(mode)
        if self.cfg.freeze_vggt and self.cfg.encoder_eval_mode:
            self.vggt.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        return_attn: bool = False,
    ):
        # ---- 1) encode images (VGGT) ----
        # vggt_feat: [B,S,P,2C], patch_start_idx: int
        if self.cfg.freeze_vggt and self.cfg.encoder_no_grad:
            with torch.no_grad():
                vggt_feat, patch_start_idx = self.vggt(images)
        else:
            vggt_feat, patch_start_idx = self.vggt(images)

        # ---- 2) build KV from patch tokens (last frame) ----
        # kv: [B, P_patch, d_model]
        kv = self.connector(vggt_feat, patch_start_idx)

        # PlannerDecoder 期望 img_feat: [B,S,P,D] 并内部取 S=-1
        # 这里把 kv 包成 S=1： [B,1,P_patch,D]
        img_feat_for_decoder = kv.unsqueeze(1)

        # ---- 3) build Q from state ----
        # state 可以是 [B,state_dim] 或 [B,S,state_dim]
        state_emb = self.state_embed(state)  # -> [B,d_model] or [B,S,d_model]

        # 只用最后一帧 state（和你对 image 的处理一致）
        if state_emb.dim() == 3:
            q_embed = state_emb[:, -1, :]  # [B,d_model]
        else:
            q_embed = state_emb            # [B,d_model]

        # ---- 4) decode ----
        if return_attn:
            q_out, attn_list = self.decoder(
                q_embed=q_embed,                 # [B,d_model] -> decoder 内部会变成 [B,1,d_model]
                img_feat=img_feat_for_decoder,   # [B,1,P_patch,d_model]
                kv_key_padding_mask=None,
                return_attn=True,
            )
            v_pred = self.head(q_out)            # [B,out_dim]
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

    cfg = VelocityPlannerConfig(
        state_dim=7,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=1024,
        dropout=0.1,
        img_size=518,
        patch_size=14,
        vggt_embed_dim=1024,
        pretrained_agg_path=None,
        max_patches=512,
        out_dim=3,
        freeze_vggt=True,         # ✅ 只冻结 encoder
        encoder_eval_mode=True,
        encoder_no_grad=True,
    )

    model = VelocityPlannerModel(cfg).to(device)
    model.train()  # 即使 train()，vggt 也会保持 eval()（冻结策略生效）

    B, S = 2, 4
    H, W = 518, 518
    images = torch.rand(B, S, 3, H, W, device=device)
    state = torch.rand(B, S, cfg.state_dim, device=device)

    v = model(images, state)
    print("v_pred shape:", tuple(v.shape))

    # sanity check: encoder params should be frozen, others trainable
    n_frozen = sum(1 for p in model.vggt.parameters() if not p.requires_grad)
    n_trainable_conn = sum(1 for p in model.connector.parameters() if p.requires_grad)
    n_trainable_dec = sum(1 for p in model.decoder.parameters() if p.requires_grad)
    print("vggt frozen params:", n_frozen)
    print("connector trainable params:", n_trainable_conn)
    print("decoder trainable params:", n_trainable_dec)
