import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ================= 工具 ================
def _build_sincos_embed(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, "sincos 维度需为偶数"
    half = dim // 2
    idx = torch.arange(half, device=x.device, dtype=x.dtype)
    freq = torch.exp(- math.log(10000.0) * (2 * idx / dim))
    angles = x * (2 * math.pi) * freq
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

def _mlp(in_dim: int, out_dim: int, hidden: Optional[int] = None):
    h = hidden or max(64, out_dim)
    return nn.Sequential(nn.Linear(in_dim, h), nn.GELU(), nn.Linear(h, out_dim))

# ========== 空间位置编码（(u,v)∈[0,1]） ==========
class SpatialPosEnc(nn.Module):
    def __init__(self, mode: str, d_model: int, N_tokens: Optional[int] = None,
                 fourier_B: float = 64.0, mlp_hidden: Optional[int] = None):
        super().__init__()
        self.mode = mode.lower()
        self.d = d_model

        if self.mode == "learned":
            assert N_tokens is not None, "learned 模式需要固定 N_tokens"
            self.table = nn.Parameter(torch.randn(N_tokens, d_model) * 0.02)
        elif self.mode == "mlp":
            self.net = _mlp(2, d_model, mlp_hidden)
        elif self.mode == "fourier":
            self.K = max(1, d_model // 4)
            self.B = nn.Parameter(torch.randn(2, self.K) * fourier_B, requires_grad=False)
            self.proj = nn.Linear(4 * self.K, d_model)
        elif self.mode == "sincos":
            assert d_model >= 2
            if d_model % 2 == 0:
                self.du = self.dv = d_model // 2
                self.post = nn.Identity()
            else:
                self.du = self.dv = (d_model - 1) // 2
                self.post = nn.Linear(self.du + self.dv, d_model)
        else:
            raise ValueError(f"Unknown spatial mode: {mode}")

    def forward(self, coords_uv: torch.Tensor, B: int, T: int) -> torch.Tensor:
        N = coords_uv.size(0)
        if self.mode == "learned":
            pe = self.table
        elif self.mode == "mlp":
            pe = self.net(coords_uv)
        elif self.mode == "fourier":
            u = coords_uv[..., 0:1]
            v = coords_uv[..., 1:2]
            U = u @ self.B[0:1]
            V = v @ self.B[1:2]
            feats = torch.cat([torch.sin(U), torch.cos(U), torch.sin(V), torch.cos(V)], dim=-1)
            pe = self.proj(feats)
        else:  # sincos
            u = coords_uv[..., 0:1]
            v = coords_uv[..., 1:2]
            eu = _build_sincos_embed(u, self.du)
            ev = _build_sincos_embed(v, self.dv)
            pe = self.post(torch.cat([eu, ev], dim=-1))
        return pe.view(1, 1, N, self.d).expand(B, T, N, self.d)

# ========== 时间位置编码（帧索引 t） ==========
class TemporalPosEnc(nn.Module):
    def __init__(self, mode: str, d_model: int, Tmax: int = 64,
                 fourier_B: float = 32.0, mlp_hidden: Optional[int] = None):
        super().__init__()
        self.mode = mode.lower()
        self.d = d_model
        self.Tmax = Tmax

        if self.mode == "learned":
            self.table = nn.Embedding(Tmax, d_model)
        elif self.mode == "mlp":
            self.net = _mlp(1, d_model, mlp_hidden)
        elif self.mode == "fourier":
            self.K = max(1, d_model // 2)
            self.B = nn.Parameter(torch.randn(self.K) * fourier_B, requires_grad=False)
            self.proj = nn.Linear(2 * self.K, d_model)
        elif self.mode == "sincos":
            assert d_model % 2 == 0
        else:
            raise ValueError(f"Unknown temporal mode: {mode}")

    def forward(self, T: int, B: int, N: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(T, device=device, dtype=torch.float32)
        if self.mode == "learned":
            assert T <= self.Tmax
            pe_t = self.table(t.long())
        elif self.mode == "mlp":
            t_norm = (t / max(1, self.Tmax - 1)).unsqueeze(-1)
            pe_t = self.net(t_norm)
        elif self.mode == "fourier":
            angles = (t.unsqueeze(-1) * (2 * math.pi) * self.B)
            feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            pe_t = self.proj(feats)
        else:
            pe_t = _build_sincos_embed(t.view(T, 1), self.d)
        return pe_t.view(1, T, 1, self.d).expand(B, T, N, self.d)

# ========== Q-Former block ==========
class QFormerBlock(nn.Module):
    def __init__(self, d, nhead, mlp_ratio=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d*mlp_ratio), nn.GELU(), nn.Linear(d*mlp_ratio, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)

    def forward(self, q, kv, kv_mask=None):
        q = q + self.self_attn(self.n1(q), self.n1(q), self.n1(q))[0]
        q = q + self.cross_attn(self.n2(q), self.n2(kv), self.n2(kv), key_padding_mask=kv_mask)[0]
        q = q + self.mlp(self.n3(q))
        return q

# ========== 分层多模态 Q-Former 轨迹头 ==========
class HierTrajectoryQFormer(nn.Module):
    """
    分层结构：
      - 每个 mode m 有 Nq 个可学习 queries: [K, Nq, d_q]
      - 先对每个 mode 的 queries 与 KV 做多层 Q-Former 编码
      - mode 内部：用 per-query 的权重（detach 以稳定）对 query 轨迹做加权得到单条轨迹
      - mode 级别：对该 mode 的 query 表示做均值作为 mode 表示，再出一个 mode score
    输出：
      traj_out: [B, K, H, 2]
      score:    [B, K]
    """
    def __init__(
        self,
        d_vis: int,
        d_q: Optional[int] = None,
        nhead: int = 8,
        layers: int = 4,
        H: int = 12,
        K: int = 3,         # 模态数
        Nq: int = 32,       # 每个 mode 的 query 数
        Tmax: int = 16,
        sp_mode: str = "fourier",
        tp_mode: str = "sincos",
    ):
        super().__init__()
        self.d_q = d_q if d_q is not None else d_vis
        assert self.d_q % nhead == 0, "d_q 必须能被 nhead 整除"
        self.H, self.K, self.Nq = H, K, Nq

        # 维度对齐
        self.align = nn.Identity() if d_vis == self.d_q else nn.Linear(d_vis, self.d_q)

        # 位置编码
        self.spatial_enc = SpatialPosEnc(mode=sp_mode, d_model=self.d_q)
        self.temporal_enc = TemporalPosEnc(mode=tp_mode, d_model=self.d_q, Tmax=Tmax)

        # 每个 mode 的独立 queries
        self.mode_queries = nn.Parameter(torch.randn(K, Nq, self.d_q) * 0.02)

        # 共享的 Q-Former 堆叠（也可改为 per-mode 私有）
        self.blocks = nn.ModuleList([QFormerBlock(self.d_q, nhead) for _ in range(layers)])

        # per-query 轨迹头 & per-query 权重头（用于 mode 内聚合）
        self.traj_head_q  = nn.Linear(self.d_q, H * 2)      # 每个 query 直接解码一条 Hx2 轨迹
        self.weight_head_q = nn.Linear(self.d_q, 1)         # 每个 query 的聚合权重（标量）

        # mode 级别打分（对该 mode 的 queries 平均后得到表示）
        self.mode_score = nn.Linear(self.d_q, 1)

    def forward(self, image_feats: torch.Tensor, coords_uv: torch.Tensor,
                frame_mask: Optional[torch.Tensor] = None, cmd_embed: Optional[torch.Tensor] = None):
        """
        image_feats: [B, T, N, d_vis]
        coords_uv:   [N, 2]
        frame_mask:  [B, T] (True=pad)
        """
        B, T, N, _ = image_feats.shape
        device = image_feats.device

        # 1) 对齐 & 位置
        x = self.align(image_feats)                         # [B,T,N,d_q]
        pos_sp = self.spatial_enc(coords_uv.to(device), B=B, T=T)
        pos_t  = self.temporal_enc(T=T, B=B, N=N, device=device)
        x = x + pos_sp + pos_t
        kv = x.view(B, T*N, self.d_q)                       # [B,TN,d_q]
        kv_mask = None
        if frame_mask is not None:
            kv_mask = frame_mask.view(B, T, 1).expand(B, T, N).reshape(B, T*N)

        # 2) 展开成 (B*K) 个“子 batch”，每个子 batch 有 Nq 个 query
        q0 = self.mode_queries.unsqueeze(0).expand(B, -1, -1, -1)  # [B,K,Nq,d]
        if cmd_embed is not None:
            q0 = q0 + cmd_embed.unsqueeze(1).unsqueeze(2)          # 条件化
        q = q0.reshape(B * self.K, self.Nq, self.d_q)              # [B*K,Nq,d]
        kv_rep = kv.unsqueeze(1).expand(B, self.K, T*N, self.d_q).reshape(B*self.K, T*N, self.d_q)
        kv_mask_rep = None if kv_mask is None else kv_mask.unsqueeze(1).expand(B, self.K, T*N).reshape(B*self.K, T*N)

        # 3) 多层 Q-Former
        for blk in self.blocks:
            q = blk(q, kv_rep, kv_mask_rep)                        # [B*K,Nq,d]

        # 4) per-query 轨迹与权重（在每个 mode 内部聚合）
        traj_q = self.traj_head_q(q).view(B*self.K, self.Nq, self.H, 2)  # [B*K,Nq,H,2]
        w_q = self.weight_head_q(q).squeeze(-1)                           # [B*K,Nq]
        w_q = torch.softmax(w_q.detach(), dim=1)                          # detach 提升稳定性
        traj_mode = (traj_q * w_q.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) # [B*K,H,2]

        # 5) mode 分数（用该 mode 的 query 平均作为 mode 表示）
        mode_repr = q.mean(dim=1)                                         # [B*K,d]
        score = self.mode_score(mode_repr).view(B, self.K)                # [B,K]

        # 6) 还原形状
        traj_out = traj_mode.view(B, self.K, self.H, 2)                   # [B,K,H,2]
        return traj_out, score


# ================== 简单测试 ==================
if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, N, d_vis = 2, 3, 64, 1024
    image_feats = torch.randn(B, T, N, d_vis)
    coords_uv = torch.rand(N, 2)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)

    model = HierTrajectoryQFormer(
        d_vis=d_vis,
        d_q=1024,
        nhead=8,
        layers=4,
        H=12,
        K=1,        #  个模态
        Nq=32,      # 每个模态 32 个 queries
        Tmax=16,
        sp_mode="fourier",   # "sincos" / "learned" / "mlp" / "fourier"
        tp_mode="sincos",
    )

    traj_out, score = model(image_feats, coords_uv, frame_mask)

    print("✅ Forward successful!")
    print(f"输入: image_feats = {tuple(image_feats.shape)}")
    print(f"输出轨迹: traj_out = {tuple(traj_out.shape)}")  # [B,K,H,2]
    print(f"输出分数: score = {tuple(score.shape)}")        # [B,K]
    print("\n示例: 第一条样本、第0号模态，前三步：")
    print(traj_out[0, 0, :3])
