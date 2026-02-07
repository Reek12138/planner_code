# /workspace/zhuy25@xiaopeng.com/planner_code/ai_planner/planner/models/modeling/velocity_planner_model.py
# -*- coding: utf-8 -*-

"""
VelocityPlannerModel

串联：
  ViT4FrameEncoder -> VGGTConnector(patch tokens) -> Transformer Decoder(Q=state, KV=image) -> VelocityHead

约定：
- d_model = 256（由 Fourier state embedding 输出对齐）
- Encoder 输出 feat: [B,S,N,2*E]，N = (1 + P) if use_cls_token else P
  - patch tokens 从 patch_start_idx 开始
- KV：使用 encoder feat 的 patch tokens，只取最后一帧 S=-1（connector 内 use_last_frame=True）
- Q：state（目标速度+位姿等） -> Fourier -> [B,256] -> [B,1,256]
- 输出：速度向量（默认 3 维）

冻结策略（本文件新增接口）：
- 仅支持冻结 encoder：cfg.freeze_vggt=True   （字段名保留，但现在指 encoder）
- connector / decoder / head 始终可训练（不冻结）
- 冻结时：
  - encoder 参数 requires_grad=False
  - encoder 维持 eval()（即使外部调用 model.train()）
  - encoder forward 可选 no_grad（省显存）：cfg.encoder_no_grad=True

新增（本次）：
- Encoder 可选从 DINOv2 初始化（只初始化单帧/空间部分）：
    cfg.encoder_init_dinov2=True
    cfg.encoder_dinov2_model_name="dinov2_vitl14"
    cfg.encoder_dinov2_local_ckpt_path="/path/to/dinov2_vitl14.pth"  # 可选，离线加载
  注意：要与 DINOv2 对齐，默认 vit_embed_dim=1024, vit_nhead=16, patch_size=14

- Activation checkpointing 接口：
    model.gradient_checkpointing_enable(enabled=True, use_reentrant=False, target="encoder")
  优先把开关向下传给 ViT4FrameEncoder（推荐 encoder 内实现 block-level checkpoint）。
  如果 encoder 没实现接口，则 fallback：对 encoder 整段 forward 做 checkpoint（更慢，且节省不如 block-level）。
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint as _ckpt

# NEW: use your ViT encoder
from ..encoder.vision_encoder_dino import ViT4FrameEncoder, ViT4FrameEncoderConfig

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

        x = x.unsqueeze(-1)  # [..., in_dim, 1]
        y = x * self.freqs.to(x.device) * (2.0 * math.pi)
        emb = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)  # [..., in_dim, 2*num_freq]
        emb = emb.flatten(-2)  # [..., in_dim*2*num_freq]
        return self.proj(emb)


@dataclass
class VelocityPlannerConfig:
    # ---------- state ----------
    state_dim: int = 7
    fourier_freqs: int = 16

    # ---------- decoder dims ----------
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_ff: int = 1024
    dropout: float = 0.1

    # ---------- vision encoder (ViT 4-frame) ----------
    img_size: int = 518
    patch_size: int = 14

    # NOTE: 字段名保留 vggt_embed_dim，但现在表示 vit_embed_dim（E）
    # ✅ 默认与 DINOv2 ViT-L/14 对齐：E=1024
    # 输出特征维仍然是 2*E，保证 connector 能无缝接上
    vggt_embed_dim: int = 1024

    # ViT 超参：默认与 DINOv2 vitl14 对齐（heads=16）
    vit_num_frames: int = 4
    vit_nhead: int = 16
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.0
    vit_spatial_layers: int = 6     # 你可以按算力调小/调大；用于跨域 finetune 通常 2~12
    vit_temporal_layers: int = 2
    vit_use_cls_token: bool = True
    vit_use_time_embed: bool = True

    # ---------- encoder init (optional) ----------
    encoder_init_dinov2: bool = False
    encoder_dinov2_model_name: str = "dinov2_vitl14"
    # 离线加载：指向本地 dinov2 checkpoint（可为空则走 torch.hub 或 cache）
    encoder_dinov2_local_ckpt_path: Optional[str] = None

    # ---------- connector ----------
    max_patches: Optional[int] = None

    # ---------- output ----------
    out_dim: int = 3

    # ---------- freeze ----------
    freeze_vggt: bool = False            # ✅ 仅冻结 encoder（字段名不变）
    encoder_eval_mode: bool = True
    encoder_no_grad: bool = True

    # ---------- activation checkpointing ----------
    activation_checkpointing: bool = False
    activation_ckpt_use_reentrant: bool = False
    activation_ckpt_target: str = "encoder"  # {"encoder","all"}


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

        # 2) ViT 4-frame encoder (replace VGGTFrameEncoder)
        vit_cfg = ViT4FrameEncoderConfig(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            num_frames=cfg.vit_num_frames,
            embed_dim=cfg.vggt_embed_dim,     # E (default 1024 aligns with DINOv2 ViT-L/14)
            out_dim=None,                    # -> 2*E
            nhead=cfg.vit_nhead,
            mlp_ratio=cfg.vit_mlp_ratio,
            dropout=cfg.vit_dropout,
            spatial_layers=cfg.vit_spatial_layers,
            temporal_layers=cfg.vit_temporal_layers,
            use_cls_token=cfg.vit_use_cls_token,
            use_time_embed=cfg.vit_use_time_embed,
        )
        self.vggt = ViT4FrameEncoder(vit_cfg)

        # 2.1) Optional: init encoder spatial part from DINOv2 (single-frame init)
        if cfg.encoder_init_dinov2:
            try:
                self.vggt.init_from_dinov2(
                    model_name=cfg.encoder_dinov2_model_name,
                    local_ckpt_path=cfg.encoder_dinov2_local_ckpt_path,
                    verbose=True,
                )
            except Exception as e:
                print("[WARN] init_from_dinov2 failed:", repr(e))

        # encoder 输出 dim = 2*E（对齐你原 connector 习惯）
        vggt_out_dim = 2 * cfg.vggt_embed_dim

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
            use_self_attn=False,
            pre_norm=True,
        )
        self.head = VelocityHead(d_model=cfg.d_model, out_dim=cfg.out_dim)
        # self.head = VelocityHeadTanh(d_model=cfg.d_model, out_dim=cfg.out_dim)

        # 5) Freeze ONLY encoder. Connector/Decoder/Head keep trainable.
        if cfg.freeze_vggt:
            for p in self.vggt.parameters():
                p.requires_grad = False

        if cfg.freeze_vggt and cfg.encoder_eval_mode:
            self.vggt.eval()

        # 6) Activation checkpointing (optional)
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
        """
        Enable activation checkpointing.

        target:
          - "encoder": only checkpoint encoder path (recommended)
          - "all": (reserved) currently still only encoder+connector path in this file
        """
        self.cfg.activation_checkpointing = bool(enabled)
        self.cfg.activation_ckpt_use_reentrant = bool(use_reentrant)
        self.cfg.activation_ckpt_target = str(target)

        # Prefer encoder-native implementation (block-level checkpointing)
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
            # fallback: do checkpoint on the whole vggt forward in this file
            self._fallback_ckpt_encoder = bool(enabled)

        return self

    def gradient_checkpointing_disable(self):
        return self.gradient_checkpointing_enable(enabled=False)

    def train(self, mode: bool = True):
        """
        覆盖 train()：当冻结 encoder 时，确保外部 model.train() 不会把 encoder 切回 train()
        """
        super().train(mode)
        if self.cfg.freeze_vggt and self.cfg.encoder_eval_mode:
            self.vggt.eval()
        return self

    # -------------------------
    # internal helpers
    # -------------------------
    def _encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        returns:
          vggt_feat: [B,S,N,2E]
          patch_start_idx: int
        """
        # Case 1) frozen + no_grad
        if self.cfg.freeze_vggt and self.cfg.encoder_no_grad:
            with torch.no_grad():
                return self.vggt(images)

        # Case 2) activation checkpointing (encoder) fallback
        if (
            self.cfg.activation_checkpointing
            and getattr(self, "_fallback_ckpt_encoder", False)
            and self.training
        ):
            # torch.utils.checkpoint requires Tensor outputs only.
            # So we return patch_start_idx as a 1-element int64 tensor, then convert back.
            def _vggt_forward(x: torch.Tensor):
                feat, patch_start_idx = self.vggt(x)  # patch_start_idx is python int
                patch_t = torch.tensor([int(patch_start_idx)], device=feat.device, dtype=torch.int64)
                return feat, patch_t

            feat, patch_t = _ckpt(_vggt_forward, images, use_reentrant=self.cfg.activation_ckpt_use_reentrant)
            patch_start_idx = int(patch_t[0].item())
            return feat, patch_start_idx

        # Case 3) normal
        return self.vggt(images)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        return_attn: bool = False,
    ):
        # ---- 1) encode images (ViT) ----
        vggt_feat, patch_start_idx = self._encode_images(images)

        # ---- 2) build KV from patch tokens (last frame) ----
        kv = self.connector(vggt_feat, patch_start_idx)
        img_feat_for_decoder = kv.unsqueeze(1)  # [B,1,P_patch,D]

        # ---- 3) build Q from state ----
        state_emb = self.state_embed(state)  # -> [B,d_model] or [B,S,d_model]
        q_embed = state_emb[:, -1, :] if state_emb.dim() == 3 else state_emb

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

    cfg = VelocityPlannerConfig(
        state_dim=7,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=1024,
        dropout=0.1,
        img_size=518,
        patch_size=14,

        # ✅ default aligns with DINOv2 vitl14
        vggt_embed_dim=1024,
        vit_num_frames=4,
        vit_nhead=16,
        vit_spatial_layers=6,
        vit_temporal_layers=2,
        vit_dropout=0.0,
        vit_use_cls_token=True,
        vit_use_time_embed=True,

        # init from dinov2 (optional)
        encoder_init_dinov2=False,
        encoder_dinov2_model_name="dinov2_vitl14",
        encoder_dinov2_local_ckpt_path=None,  # "/path/to/dinov2_vitl14.pth"

        max_patches=512,
        out_dim=3,

        freeze_vggt=False,
        encoder_eval_mode=True,
        encoder_no_grad=True,

        activation_checkpointing=True,
        activation_ckpt_use_reentrant=False,
        activation_ckpt_target="encoder",
    )

    model = VelocityPlannerModel(cfg).to(device)
    model.train()

    model.gradient_checkpointing_enable(True, use_reentrant=False, target="encoder")

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
