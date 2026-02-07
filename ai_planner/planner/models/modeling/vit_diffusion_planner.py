"""
ViT + Connector + Diffusion Planner Model

Complete model architecture combining:
- ViT4FrameEncoder for image encoding
- VGGTConnector for feature projection
- DiffusionPlannerDecoder for velocity generation

This model replaces the autoregressive decoder with a diffusion model.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing components (assuming these paths are correct)
from ..encoder.vision_encoder import ViT4FrameEncoder, ViT4FrameEncoderConfig
from ..connector.connector import VGGTConnector
from ..decoder.diffusion_decoder import DiffusionPlanner


class StateFourierEmbedding(nn.Module):
    """Fourier embedding for state (same as in vit_planner)"""
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
        x: [B, in_dim] or [B, S, in_dim]
        return: [B, out_dim] or [B, S, out_dim]
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
class ViTDiffusionPlannerConfig:
    """Configuration for ViT Diffusion Planner"""
    # State
    state_dim: int = 10
    fourier_freqs: int = 16
    
    # Decoder (Diffusion)
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_ff: int = 2048
    dropout: float = 0.1
    num_velocity_tokens: int = 4
    
    # Diffusion params
    num_train_timesteps: int = 1000
    num_inference_steps: int = 50  # Can be less for faster inference
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Vision encoder
    img_size: int = 518
    patch_size: int = 14
    vggt_embed_dim: int = 1024  # ViT embed dim (E), output will be 2*E
    vit_num_frames: int = 4
    vit_nhead: int = 16
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.0
    vit_spatial_layers: int = 2
    vit_temporal_layers: int = 1
    vit_use_cls_token: bool = True
    vit_use_time_embed: bool = True
    
    # Connector
    max_patches: Optional[int] = 512
    
    # Output
    out_dim: int = 3
    
    # Freeze options
    freeze_vggt: bool = False
    encoder_eval_mode: bool = True
    encoder_no_grad: bool = True
    
    # Activation checkpointing
    activation_checkpointing: bool = False
    activation_ckpt_use_reentrant: bool = False


class ViTDiffusionPlannerModel(nn.Module):
    """
    ViT Encoder + Connector + Diffusion Decoder for Velocity Planning
    
    Forward modes:
    - Training: Takes (images, state, v_gt) and returns diffusion loss
    - Inference: Takes (images, state) and generates velocity via diffusion sampling
    """
    def __init__(self, cfg: ViTDiffusionPlannerConfig):
        super().__init__()
        self.cfg = cfg
        
        # 1) State Fourier embedding
        self.state_embed = StateFourierEmbedding(
            in_dim=cfg.state_dim,
            num_freq=cfg.fourier_freqs,
            out_dim=cfg.d_model,
        )
        
        # 2) ViT 4-frame encoder
        vit_cfg = ViT4FrameEncoderConfig(
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
        )
        self.vggt = ViT4FrameEncoder(vit_cfg)
        
        # Encoder output dim = 2*E
        vggt_out_dim = 2 * cfg.vggt_embed_dim
        
        # 3) Connector: patch tokens -> d_model
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
        
        # 4) Diffusion Planner (replaces Transformer Decoder + Head)
        self.diffusion = DiffusionPlanner(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_ff=cfg.dim_ff,
            dropout=cfg.dropout,
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            num_velocity_tokens=cfg.num_velocity_tokens,
        )
        
        # Freeze encoder if requested
        if cfg.freeze_vggt:
            for p in self.vggt.parameters():
                p.requires_grad = False
        
        if cfg.activation_checkpointing:
            self.gradient_checkpointing_enable(
                enabled=True,
                use_reentrant=cfg.activation_ckpt_use_reentrant,
            )
    
    def gradient_checkpointing_enable(self, enabled: bool = True, use_reentrant: bool = False):
        """Enable activation checkpointing"""
        if hasattr(self.vggt, "gradient_checkpointing_enable"):
            try:
                self.vggt.gradient_checkpointing_enable(enabled=enabled, use_reentrant=use_reentrant)
            except TypeError:
                self.vggt.gradient_checkpointing_enable(enabled)
        return self
    
    def gradient_checkpointing_disable(self):
        return self.gradient_checkpointing_enable(False)
    
    def train(self, mode: bool = True):
        """Override train to keep encoder in eval mode if frozen"""
        super().train(mode)
        if self.cfg.freeze_vggt and self.cfg.encoder_eval_mode:
            self.vggt.eval()
        return self
    
    def _encode_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Encode images with ViT"""
        if self.cfg.freeze_vggt and self.cfg.encoder_no_grad:
            with torch.no_grad():
                return self.vggt(images)
        return self.vggt(images)
    
    def encode_conditioning(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode conditioning information (images + state)
        
        Returns:
            img_feat: [B, P, d_model] image features from connector
            state_emb: [B, d_model] state embedding
        """
        # Encode images
        vggt_feat, patch_start_idx = self._encode_images(images)  # [B,S,N,2E], int
        
        # Connect to d_model
        img_feat = self.connector(vggt_feat, patch_start_idx)  # [B, P, d_model]
        
        # Encode state
        state_emb = self.state_embed(state)  # [B, d_model] or [B, S, d_model]
        if state_emb.dim() == 3:
            state_emb = state_emb[:, -1, :]  # Take last frame if sequential
        
        return img_feat, state_emb
    
    def forward_train(
        self,
        images: torch.Tensor,      # [B,S,3,H,W]
        state: torch.Tensor,       # [B,state_dim] or [B,S,state_dim]
        v_gt: torch.Tensor,        # [B,3] ground truth velocity (normalized)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Training forward pass
        
        Returns:
            loss: scalar diffusion loss
            metrics: dict of logging metrics
        """
        # Encode conditioning
        img_feat, state_emb = self.encode_conditioning(images, state)
        
        # Diffusion forward pass (adds noise and predicts it)
        loss, noise_pred = self.diffusion(v_gt, img_feat, state_emb)
        
        metrics = {
            "loss": loss.item(),
            "noise_magnitude": noise_pred.norm(dim=-1).mean().item(),
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def forward_inference(
        self,
        images: torch.Tensor,      # [B,S,3,H,W]
        state: torch.Tensor,       # [B,state_dim] or [B,S,state_dim]
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inference: generate velocity via diffusion sampling
        
        Returns:
            v_pred: [B,3] predicted velocity
        """
        num_steps = num_steps or self.cfg.num_inference_steps
        
        # Encode conditioning
        img_feat, state_emb = self.encode_conditioning(images, state)
        
        batch_size = images.shape[0]
        device = images.device
        
        # Sample from diffusion
        v_pred = self.diffusion.sample(
            img_feat=img_feat,
            state_emb=state_emb,
            batch_size=batch_size,
            num_inference_steps=num_steps,
            device=device,
        )
        
        return v_pred
    
    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        v_gt: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        """
        Unified forward pass
        
        If v_gt is provided: training mode (returns loss)
        If v_gt is None: inference mode (returns prediction)
        """
        if self.training or v_gt is not None:
            # Training mode
            loss, metrics = self.forward_train(images, state, v_gt)
            if return_dict:
                return {"loss": loss, **metrics}
            return loss
        else:
            # Inference mode
            v_pred = self.forward_inference(images, state)
            return v_pred


# ----------------------------
# Quick test
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = ViTDiffusionPlannerConfig(
        state_dim=10,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=1024,
        img_size=518,
        vggt_embed_dim=256,
        vit_num_frames=4,
        vit_spatial_layers=2,
        vit_temporal_layers=1,
        num_train_timesteps=100,
        num_inference_steps=20,
    )
    
    model = ViTDiffusionPlannerModel(cfg).to(device)
    model.train()
    
    B, S = 2, 4
    images = torch.randn(B, S, 3, 518, 518, device=device)
    state = torch.randn(B, 10, device=device)
    v_gt = torch.randn(B, 3, device=device)
    
    # Test training forward
    loss, metrics = model.forward_train(images, state, v_gt)
    print(f"Training loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test inference
    model.eval()
    v_pred = model.forward_inference(images, state, num_steps=20)
    print(f"Inference output shape: {v_pred.shape}")
    print(f"Predicted velocity: {v_pred[0]}")
