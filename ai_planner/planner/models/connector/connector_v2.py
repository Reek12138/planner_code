# /workspace/zhuy25@xiaopeng.com/planner_code/ai_planner/planner/models/connector/connector_v2.py
# -*- coding: utf-8 -*-
"""
Connector V2 with 2x2 token compression

主要改进：
1. 支持 2x2 patch 合并压缩为 1x1 token（使用 MLP）
2. 保持与 V1 的接口兼容

使用场景：
- 输入 128x128，patch_size=14 -> 约 9x9=81 patches
- 2x2 压缩后 -> 约 4x4=16 tokens（更高效）
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class MLPProjector(nn.Module):
    """
    通用 MLP 投影器：D_in -> D_out
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layernorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1
        act = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }[activation.lower()]()

        if num_layers == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            h = hidden_dim if hidden_dim is not None else max(in_dim, out_dim)
            layers = [nn.Linear(in_dim, h), act]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers += [nn.Linear(h, h), act]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(h, out_dim)]

        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.net(x))


class PatchCompressor2x2(nn.Module):
    """
    2x2 Patch Compressor: 将空间上相邻的 2x2 个 patch 合并为 1 个 token
    
    输入: [B, P, D_in] 其中 P = H_patches * W_patches
    输出: [B, P/4, D_out]
    
    要求: H_patches 和 W_patches 都必须是偶数
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_h: int,
        grid_w: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        
        # 检查 grid 尺寸是否为偶数
        assert grid_h % 2 == 0, f"grid_h must be even, got {grid_h}"
        assert grid_w % 2 == 0, f"grid_w must be even, got {grid_w}"
        
        self.out_grid_h = grid_h // 2
        self.out_grid_w = grid_w // 2
        self.num_out_patches = self.out_grid_h * self.out_grid_w
        
        # 4 个 patch 拼接后的维度 -> out_dim 的 MLP
        # 输入是 4 个 patch 的 concat: 4 * in_dim
        self.compress_mlp = MLPProjector(
            in_dim=in_dim * 4,  # 2x2 patches concatenated
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, P, D_in] where P = grid_h * grid_w
        return: [B, P/4, D_out]
        """
        B, P, D = x.shape
        assert P == self.grid_h * self.grid_w, \
            f"Input patches {P} doesn't match grid {self.grid_h}x{self.grid_w}={self.grid_h * self.grid_w}"
        assert D == self.in_dim, f"Input dim {D} doesn't match expected {self.in_dim}"
        
        # Reshape to spatial grid: [B, grid_h, grid_w, D]
        x = x.view(B, self.grid_h, self.grid_w, D)
        
        # Reshape to 2x2 blocks: [B, out_h, out_w, 2, 2, D]
        x = x.view(B, self.out_grid_h, 2, self.out_grid_w, 2, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, out_h, out_w, 2, 2, D]
        
        # Flatten 2x2 patches: [B, out_h, out_w, 4*D]
        x = x.view(B, self.out_grid_h, self.out_grid_w, 4 * D)
        
        # Flatten spatial dims: [B, num_out_patches, 4*D]
        x = x.view(B, self.num_out_patches, 4 * D)
        
        # Apply MLP compression: [B, num_out_patches, D_out]
        out = self.compress_mlp(x)
        
        return out


class VGGTConnectorV2(nn.Module):
    """
    VGGT Connector V2 with optional 2x2 token compression
    
    输入:
        vggt_feat: [B, S, P, D_in]  (P includes cls token if present)
        patch_start_idx: int
    
    输出:
        kv: [B, P_compressed, D_model] (if use_last_frame=True)
        
    新增参数:
        use_2x2_compression: 是否启用 2x2 patch 压缩
        grid_h, grid_w: 原始 patch grid 尺寸（压缩时需要）
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        use_last_frame: bool = True,
        keep_time: bool = False,
        max_patches: Optional[int] = None,
        # projector params
        proj_hidden_dim: Optional[int] = None,
        proj_num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layernorm: bool = True,
        # 2x2 compression params
        use_2x2_compression: bool = False,
        grid_h: int = 9,  # for 128x128 with patch_size=14: 128/14 ≈ 9
        grid_w: int = 9,
        compression_hidden_dim: Optional[int] = None,
        compression_num_layers: int = 2,
    ):
        super().__init__()
        if keep_time and use_last_frame:
            raise ValueError("keep_time=True 时不应 use_last_frame=True（矛盾）。")

        self.use_last_frame = use_last_frame
        self.keep_time = keep_time
        self.max_patches = max_patches
        self.use_2x2_compression = use_2x2_compression
        
        if use_2x2_compression:
            # 先压缩 2x2 patches，再投影到 d_model
            self.compressor = PatchCompressor2x2(
                in_dim=in_dim,
                out_dim=in_dim,  # 保持维度，后续 MLP 投影
                grid_h=grid_h,
                grid_w=grid_w,
                hidden_dim=compression_hidden_dim,
                num_layers=compression_num_layers,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm,
            )
            # 压缩后的投影 MLP（可选，也可以合并到 compressor）
            self.proj = MLPProjector(
                in_dim=in_dim,
                out_dim=d_model,
                hidden_dim=proj_hidden_dim,
                num_layers=proj_num_layers,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm,
            ) if proj_num_layers > 0 else nn.Identity()
        else:
            self.compressor = None
            self.proj = MLPProjector(
                in_dim=in_dim,
                out_dim=d_model,
                hidden_dim=proj_hidden_dim,
                num_layers=proj_num_layers,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm,
            )

    def forward(self, vggt_feat: torch.Tensor, patch_start_idx: int) -> torch.Tensor:
        """
        vggt_feat: [B,S,P,D_in]
        """
        if vggt_feat.dim() != 4:
            raise ValueError(f"vggt_feat must be [B,S,P,D], got {tuple(vggt_feat.shape)}")

        B, S, P, D_in = vggt_feat.shape
        if not (0 <= patch_start_idx < P):
            raise ValueError(f"patch_start_idx out of range: {patch_start_idx}, P={P}")

        # 1) 只取 patch tokens
        patch_tokens = vggt_feat[:, :, patch_start_idx:, :]
        P_patch = patch_tokens.shape[2]

        # 2) 可选：截断 patch 数量
        if self.max_patches is not None:
            patch_tokens = patch_tokens[:, :, : self.max_patches, :]
            P_patch = patch_tokens.shape[2]

        # 3) 决定是否只用最后一帧
        if self.use_last_frame:
            patch_tokens = patch_tokens[:, -1, :, :]  # [B, Pp, Din]

        # 4) 2x2 压缩（如果启用）
        if self.use_2x2_compression and self.compressor is not None:
            # patch_tokens: [B, Pp, Din]
            patch_tokens = self.compressor(patch_tokens)  # [B, Pp/4, Din]

        # 5) 投影到 d_model
        kv = self.proj(patch_tokens)

        # 6) 处理 keep_time 情况
        if self.keep_time:
            return kv

        # 默认：不保留 time 维
        if (not self.use_last_frame) and (not self.keep_time):
            kv = kv.reshape(B, -1, kv.shape[-1])

        return kv


class StateConnector(nn.Module):
    """
    保持兼容性的 State Connector
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        proj_hidden_dim: Optional[int] = None,
        proj_num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.proj = MLPProjector(
            in_dim=in_dim,
            out_dim=d_model,
            hidden_dim=proj_hidden_dim,
            num_layers=proj_num_layers,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm,
        )

    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        if state_emb.dim() == 2:
            return self.proj(state_emb).unsqueeze(1)
        elif state_emb.dim() == 3:
            return self.proj(state_emb)
        else:
            raise ValueError(f"state_emb must be [B,D] or [B,Nq,D], got {tuple(state_emb.shape)}")


if __name__ == "__main__":
    # Test PatchCompressor2x2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=== Test PatchCompressor2x2 ===")
    # 128x128 image, patch_size=14 -> grid = 9x9 (approx)
    # Actually 128/14 = 9.14, but we use floor or padding
    # Let's assume we have 9x9 = 81 patches for simplicity
    B, P, D = 2, 81, 512  # 9x9 patches
    grid_h, grid_w = 9, 9
    
    x = torch.randn(B, P, D, device=device)
    compressor = PatchCompressor2x2(
        in_dim=D,
        out_dim=256,
        grid_h=grid_h,
        grid_w=grid_w,
        hidden_dim=1024,
        num_layers=2,
    ).to(device)
    
    out = compressor(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")  # Should be [2, 20, 256] (81/4 ≈ 20 for 9x9->4x4=16 with proper padding)
    # Actually 9x9 -> 4x4 = 16 tokens
    
    print("\n=== Test VGGTConnectorV2 with compression ===")
    B, S, P, D = 2, 4, 82, 512  # 82 = 1 cls + 81 patches
    vggt_feat = torch.randn(B, S, P, D, device=device)
    
    connector = VGGTConnectorV2(
        in_dim=D,
        d_model=2048,
        use_last_frame=True,
        use_2x2_compression=True,
        grid_h=9,
        grid_w=9,
        compression_hidden_dim=1024,
        compression_num_layers=2,
    ).to(device)
    
    kv = connector(vggt_feat, patch_start_idx=1)
    print(f"Input vggt_feat: {vggt_feat.shape}")
    print(f"Output kv: {kv.shape}")  # Should be [2, 16, 2048]
