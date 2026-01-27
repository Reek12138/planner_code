import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class StateFourierEmbedding(nn.Module):
    def __init__(self, in_dim=7, num_freq=16, out_dim=256):
        super().__init__()
        self.freqs = 2 ** torch.arange(num_freq)
        self.proj = nn.Linear(in_dim * 2 * num_freq, out_dim)

    def forward(self, x):
        # x: [B, S, 7] or [B, 7]
        x = x.unsqueeze(-1)  # [..., 7, 1]
        freqs = self.freqs.to(x.device)
        y = x * freqs * 2 * math.pi
        emb = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        emb = emb.flatten(-2)
        return self.proj(emb)


class CrossAttnDecoderBlock(nn.Module):
    """
    一个标准 Transformer Decoder Block（简化版）：
      - (可选) Self-Attention on Q tokens
      - Cross-Attention: Q attends to (K,V) from image tokens
      - FFN
    输入输出均为 [B, Nq, D]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        use_self_attn: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.pre_norm = pre_norm

        # --- Attn modules ---
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,  # (B, N, D)
            )
            self.norm_sa = nn.LayerNorm(d_model)
            self.drop_sa = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # (B, Nq, D) and (B, Nk, D)
        )
        self.norm_ca = nn.LayerNorm(d_model)
        self.drop_ca = nn.Dropout(dropout)

        # --- FFN ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.drop_ff = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,              # [B, Nq, D]
        kv: torch.Tensor,             # [B, Nk, D]  (image tokens)
        q_attn_mask: Optional[torch.Tensor] = None,      # (optional) for self-attn
        kv_key_padding_mask: Optional[torch.Tensor] = None,  # [B, Nk] True for pad
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            q_out: [B, Nq, D]
            cross_attn_weights (optional): [B, Nq, Nk] (avg over heads)
        """
        cross_w = None

        if self.pre_norm:
            # ----- (optional) Self-Attn -----
            if self.use_self_attn:
                q0 = q
                qn = self.norm_sa(q)
                sa_out, _ = self.self_attn(
                    query=qn, key=qn, value=qn,
                    attn_mask=q_attn_mask,
                    need_weights=False,
                )
                q = q0 + self.drop_sa(sa_out)

            # ----- Cross-Attn -----
            q0 = q
            qn = self.norm_ca(q)
            ca_out, ca_w = self.cross_attn(
                query=qn, key=kv, value=kv,
                key_padding_mask=kv_key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=True,  # -> [B, Nq, Nk]
            )
            q = q0 + self.drop_ca(ca_out)
            if need_weights:
                cross_w = ca_w

            # ----- FFN -----
            q0 = q
            qn = self.norm_ff(q)
            ff_out = self.ffn(qn)
            q = q0 + self.drop_ff(ff_out)

        else:
            # Post-Norm 版本（不太推荐，但给你留着）
            if self.use_self_attn:
                sa_out, _ = self.self_attn(
                    query=q, key=q, value=q,
                    attn_mask=q_attn_mask,
                    need_weights=False,
                )
                q = self.norm_sa(q + self.drop_sa(sa_out))

            ca_out, ca_w = self.cross_attn(
                query=q, key=kv, value=kv,
                key_padding_mask=kv_key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=True,
            )
            q = self.norm_ca(q + self.drop_ca(ca_out))
            if need_weights:
                cross_w = ca_w

            ff_out = self.ffn(q)
            q = self.norm_ff(q + self.drop_ff(ff_out))

        return q, cross_w


class PlannerDecoder(nn.Module):
    """
    堆叠 N 个 CrossAttnDecoderBlock
    输入：
      q_embed:   [B, D] 或 [B, Nq, D]
      img_feat:  [B, S, P, D]  （你说只用最后一帧 + P 全作为 KV）
    输出：
      q_out:     [B, Nq, D]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        use_self_attn: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnDecoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_ff=dim_ff,
                dropout=dropout,
                use_self_attn=use_self_attn,
                pre_norm=pre_norm,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(
        self,
        q_embed: torch.Tensor,            # [B, D] or [B, Nq, D]
        img_feat: torch.Tensor,           # [B, S, P, D]
        kv_key_padding_mask: Optional[torch.Tensor] = None,  # [B, P] if you have padding
        return_attn: bool = False,
    ):
        # ---- q shape normalize ----
        if q_embed.dim() == 2:
            q = q_embed.unsqueeze(1)  # [B, 1, D]
        elif q_embed.dim() == 3:
            q = q_embed               # [B, Nq, D]
        else:
            raise ValueError(f"q_embed must be [B,D] or [B,Nq,D], got {q_embed.shape}")

        # ---- use last frame only: [B, P, D] ----
        if img_feat.dim() != 4:
            raise ValueError(f"img_feat must be [B,S,P,D], got {img_feat.shape}")
        kv = img_feat[:, -1, :, :]  # [B, P, D]

        attn_list = [] if return_attn else None

        for blk in self.layers:
            q, w = blk(
                q=q,
                kv=kv,
                kv_key_padding_mask=kv_key_padding_mask,
                need_weights=return_attn,
            )
            if return_attn:
                attn_list.append(w)  # each: [B, Nq, P]

        q = self.final_norm(q)

        if return_attn:
            return q, attn_list
        return q


class VelocityHead(nn.Module):
    """
    把 decoder 输出的 q token -> 速度（例如 3D）
    你说“最后输出一个速度”，这里默认输出 [B, 3]
    """
    def __init__(self, d_model: int, out_dim: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, 1, D] or [B, Nq, D]
        # 这里只取第一个 query token
        if q.dim() == 3:
            q0 = q[:, 0, :]
        elif q.dim() == 2:
            q0 = q
        else:
            raise ValueError(f"q must be [B,D] or [B,Nq,D], got {q.shape}")
        return self.mlp(q0)


class VelocityHeadTanh(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 3:
            q0 = q[:, 0, :]
        elif q.dim() == 2:
            q0 = q
        else:
            raise ValueError(f"q must be [B,D] or [B,Nq,D], got {q.shape}")
        return 1.2*torch.tanh(self.mlp(q0))  # [-1,1]
