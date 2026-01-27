import torch
import torch.nn as nn
from typing import Optional, Tuple


class MLPProjector(nn.Module):
    """
    通用 MLP 投影器：D_in -> D_out
    可选 1层/2层/多层（用 hidden_dim 控制）
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


class VGGTConnector(nn.Module):
    """
    将 VGGT encoder 输出对齐到 decoder 的 token 表示。

    输入:
        vggt_feat: [B, S, P, D_in]
        patch_start_idx: int (patch token 起始位置)

    输出(默认):
        kv: [B, P_patch, D_model]  (只取最后一帧)
    也可以输出:
        kv: [B, S, P_patch, D_model] (如果 keep_time=True)

    你可控制:
        - 是否只取最后一帧: use_last_frame
        - 是否保留 time 维: keep_time
        - 是否截断 patch 数量: max_patches
        - MLP 投影结构: projector 参数
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        use_last_frame: bool = True,   # 只用最后一帧
        keep_time: bool = False,       # True 则输出 [B,S,Pp,D]
        max_patches: Optional[int] = None,  # 可选：只取前 N 个 patch
        # projector (MLP)
        proj_hidden_dim: Optional[int] = None,
        proj_num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ):
        super().__init__()
        if keep_time and use_last_frame:
            raise ValueError("keep_time=True 时不应 use_last_frame=True（矛盾）。")

        self.use_last_frame = use_last_frame
        self.keep_time = keep_time
        self.max_patches = max_patches

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

        # 1) 只取 patch tokens: [B,S,P_patch,D_in]
        patch_tokens = vggt_feat[:, :, patch_start_idx:, :]  # [B,S,Pp,Din]

        # 2) 可选：截断 patch 数量（比如太大时省显存）
        if self.max_patches is not None:
            patch_tokens = patch_tokens[:, :, : self.max_patches, :]

        # 3) 决定是否只用最后一帧
        if self.use_last_frame:
            # -> [B, Pp, Din]
            patch_tokens = patch_tokens[:, -1, :, :]

        # 4) feature 维用 MLP 投影到 d_model
        # 支持输入 [B,Pp,D] 或 [B,S,Pp,D]
        kv = self.proj(patch_tokens)

        # 5) 是否保留 time 维（keep_time=True 时输出 [B,S,Pp,D]）
        # 如果 use_last_frame=True，这里已经是 [B,Pp,D]，自然不保留
        if self.keep_time:
            # kv: [B,S,Pp,D]
            return kv

        # 默认：不保留 time 维（若 use_last_frame=False 且 keep_time=False，就需要 merge time）
        if (not self.use_last_frame) and (not self.keep_time):
            # 你说 S 只用最后一维，理论上这里更推荐直接 use_last_frame=True
            # 但如果你想把所有帧 patch 拼起来当 KV，也可以 merge:
            # [B,S,Pp,D] -> [B, S*Pp, D]
            kv = kv.reshape(B, -1, kv.shape[-1])

        return kv  # [B,Pp,D] or [B,S*Pp,D]


class StateConnector(nn.Module):
    """
    可选：把 Fourier embedding 输出对齐到 decoder 的 d_model。
    输入:
        state_emb: [B, D_in] or [B, Nq, D_in]
    输出:
        q: [B, 1, D_model] or [B, Nq, D_model]
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
            # [B,D] -> [B,1,D]
            return self.proj(state_emb).unsqueeze(1)
        elif state_emb.dim() == 3:
            # [B,Nq,D] -> [B,Nq,D]
            return self.proj(state_emb)
        else:
            raise ValueError(f"state_emb must be [B,D] or [B,Nq,D], got {tuple(state_emb.shape)}")
