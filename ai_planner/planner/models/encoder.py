from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SWiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x) * F.sigmoid(x)

class PositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10):
        super(PositionEncoding, self).__init__()

        # 保证 dim 为偶数，如果是奇数则增加 1
        if dim % 2 != 0:
            dim += 1  # 如果是奇数，增加 1 使其变为偶数

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [B, T, D]
        Add position encoding to the input tensor.
        """
        B, T, D = x.shape
        
        # 确保 position encoding 的维度与输入的 T 和 D 一致
        pe = self.pe[:T, :].unsqueeze(0).repeat(B, 1, 1)  # [B, T, D]

        # 如果 dim 为偶数，但实际 out_dim 是奇数，则丢弃多余的维度
        if pe.size(2) > D:
            pe = pe[:, :, :D]  # 保证位置编码的维度与输入特征维度一致

        return x + pe  # Add position encoding to the input tensor



class FourierEncoder(nn.Module):
    def __init__(self, n_scales=4, include_input=True):
        super().__init__()
        self.n_scales = int(n_scales)
        self.include_input = bool(include_input)
        self.register_buffer("freqs", 2 * math.pi * torch.tensor([2**k for k in range(self.n_scales)], dtype=torch.float32))
        # 缺省为单位缩放（等价于不缩放）
        self.register_buffer("scale", torch.ones(1))     # 或 shape [D] 运行时再扩展
        self.register_buffer("mean",  torch.zeros(1))
        self.register_buffer("std",   torch.ones(1))
        self.mode = "identity"  # 'identity' | 'scale' | 'zscore'

    def set_scale(self, scale_vec: torch.Tensor):
        self.scale = scale_vec.detach().clone().float()  # [D]
        self.mode = "scale"

    def set_zscore(self, mean_vec: torch.Tensor, std_vec: torch.Tensor):
        self.mean = mean_vec.detach().clone().float()    # [D]
        self.std  = std_vec.detach().clone().float()     # [D]
        self.mode = "zscore"

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return x
        if self.mode == "scale":
            return x / self.scale
        if self.mode == "zscore":
            return (x - self.mean) / (self.std + 1e-6)
        raise ValueError(self.mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalize(x)
        xb = x_norm[..., None] * self.freqs
        sin = torch.sin(xb); cos = torch.cos(xb)
        feats = torch.cat([sin, cos], dim=-1).reshape(x.shape[0], -1)
        if self.include_input:
            feats = torch.cat([x_norm, feats], dim=-1)
        return feats

class LidarEncoder(nn.Module):
    def __init__(self, lidar_dim: int, out_dim: int, method: str = "raw"):
        """
        lidar_dim: 雷达输入的维度 (通常是 30)
        method: 选择处理方式 ("raw", "mlp", "conv1d")
        """
        super().__init__()
        self.method = method
        
        if method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(lidar_dim, 128),
                SWiGLU(),
                nn.Linear(128, out_dim),
                SWiGLU(),
            )
        elif method == "conv1d":
            self.conv1d = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, out_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
        # 默认的 "raw" 方法就是不做处理
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, lidar_dim] 雷达数据
        """
        if self.method == "raw":
            return x
        elif self.method == "mlp":
            return self.mlp(x)
        elif self.method == "conv1d":
            # 需要将输入数据重塑为 [B, 1, lidar_dim]
            x = x.unsqueeze(1)  # [B, 1, lidar_dim]
            return self.conv1d(x).squeeze(1)  # [B, lidar_dim] 返回卷积处理后的数据

class Feature_Encoder(nn.Module):
    """
    观测特征编码器（带队列）。
    设原始观测 x 的结构为:
      x = [ dx_car, dy_car, dist, heading_err(或 sincos展平后取其一), ego_v, lidar_feats... ]
      其中前 5 维为“标量基础特征”，后面为 lidar 特征 (lidar_dim = D-5)。

    功能:
      - 对前 5 维做 FourierEncoder；lidar 直接拼接（可按需改为卷积分支）
      - 支持两种输入:
          1) 单步: x: [B, D]
             - 若 use_stateful=True，内部维护 FIFO 窗口（长度 seq_len），返回 [B, T, out_dim]
             - 若 use_stateful=False，返回 [B, 1, out_dim]
          2) 整段: x: [B, T, D] -> 逐帧编码，返回 [B, T, out_dim]
      - 队列初始化支持用第一帧填满窗口 (fill_with_first=True)

    输出维度:
      out_dim = 5 * (1 + 2 * n_scales) + lidar_dim
      当 n_scales=4 时，out_dim = 5*9 + lidar_dim = 45 + lidar_dim
    """
    def __init__(
        self,
        input_dim: int,             # D = 5 + lidar_dim
        seq_len: int = 6,          # 队列窗口长度 T
        use_stateful: bool = True,  # 是否启用内部 FIFO
        fill_with_first: bool = True,  # 首帧是否填满队列
        n_scales: int = 4,
        include_input: bool = True,
        lidar_encode_mode: str = "mlp"
    ):
        super().__init__()
        assert input_dim >= 5, "input_dim 应为 5 + lidar_dim"
        self.input_dim = int(input_dim)
        self.lidar_dim = self.input_dim - 5

        self.seq_len = int(seq_len)
        self.use_stateful = bool(use_stateful)
        self.fill_with_first = bool(fill_with_first)

        # —— 标量前5维的 Fourier 编码器 ——
        self.fourier_encoder = FourierEncoder(n_scales=n_scales, include_input=include_input)
        # 示例：物理量纲缩放（可按你的场景调整/覆盖）
        # 对应 [dx_car, dy_car, dist, heading_err(弧度), ego_v]
        self.fourier_encoder.set_scale(torch.tensor([8.0, 8.0, 8.0, 1.0, 1.5])) #TODO 归一化尺寸需要调整

        # 计算输出维度
        per_dim = (1 + 2 * n_scales) if include_input else (2 * n_scales)
        self.scalar_enc_dim = 5 * per_dim
        # —— 内部 FIFO 状态 ——
        self._buf: Optional[torch.Tensor] = None   # [B, T, out_dim]
        self._filled: int = 0
        self._batch_size: Optional[int] = None

        self.lidar_encode_method = lidar_encode_mode
        self.lidar_out_dim = 64
        self.lidar_encoder = LidarEncoder(lidar_dim=self.lidar_dim, out_dim=self.lidar_out_dim, method=self.lidar_encode_method)
        if self.lidar_encode_method == "raw":
            self.out_dim = self.scalar_enc_dim + self.lidar_dim
        else:
            self.out_dim = self.scalar_enc_dim + self.lidar_out_dim



        # —— 位置编码器 —— 这里我们为 T 维度加位置编码
        self.position_encoder = PositionEncoding(dim=self.out_dim, max_len=seq_len)

    # 重置队列（建议每个 episode 开头调用）
    def reset(self):
        self._buf = None
        self._filled = 0
        self._batch_size = None

    def _ensure_buffer(self, B: int, device, dtype, first_frame: Optional[torch.Tensor] = None):
        """
        确保缓冲存在；若为空则初始化。
        若提供 first_frame 且 fill_with_first=True，则用其填满整个窗口。
        """
        need_init = (self._buf is None) or (self._batch_size != B) or (self._buf.device != device)
        if need_init:
            if first_frame is not None and self.fill_with_first:
                # 用首帧填满窗口
                self._buf = first_frame[:, None, :].repeat(1, self.seq_len, 1).to(device=device, dtype=dtype)
                self._filled = self.seq_len
            else:
                self._buf = torch.zeros(B, self.seq_len, self.out_dim, device=device, dtype=dtype)
                self._filled = 0
            self._batch_size = B

    # 编码单帧: [B, D] -> [B, out_dim]
    def _encode_frame(self, x_frame: torch.Tensor) -> torch.Tensor:
        assert x_frame.dim() == 2 and x_frame.size(1) == self.input_dim, \
            f"expected [B, {self.input_dim}], got {tuple(x_frame.shape)}"

        scalars = x_frame[:, :5]          # [B,5]  -> fourier
        lidar   = x_frame[:, 5:]          # [B,lidar_dim]  -> 直接拼（可替换为卷积分支）

        s_feat = self.fourier_encoder(scalars)      # [B, scalar_enc_dim]
        l_feat = self.lidar_encoder(lidar)
        out = torch.cat([s_feat, l_feat], dim=-1)    # [B, out_dim]
        return out

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - [B, D]   单步；若 use_stateful=True，返回 [B, T, out_dim]（内部 FIFO）
                      否则返回 [B, 1, out_dim]
          - [B, T, D] 整段 -> 返回 [B, T, out_dim]
        """
        if x.dim() == 3:
            # 整段编码
            B, T, D = x.shape
            assert D == self.input_dim, f"expected D={self.input_dim}, got {D}"
            x_flat = x.reshape(B * T, D)
            y_flat = self._encode_frame(x_flat)      # [B*T, out_dim]
            y = y_flat.view(B, T, self.out_dim)      # [B, T, out_dim]

            return y

        elif x.dim() == 2:
            # 单步编码（带/不带 FIFO）
            B, D = x.shape
            assert D == self.input_dim, f"expected D={self.input_dim}, got {D}"
            y = self._encode_frame(x)                # [B, out_dim]

            if not self.use_stateful:
                return y[:, None, :]                 # [B, 1, out_dim]

            # 需要 FIFO：首次用当前帧填满窗口（若配置允许）
            self._ensure_buffer(B, x.device, x.dtype, first_frame=y)

            # 滚动窗口：左移一格，最后写入新帧
            self._buf = torch.roll(self._buf, shifts=-1, dims=1)
            self._buf[:, -1, :] = y
            self._filled = min(self._filled + 1, self.seq_len)

            return self._buf                         # [B, T, out_dim]

        else:
            raise ValueError(f"FeatureEncoder expects x.dim() in {{2,3}}, got {x.dim()}")
    
    def Temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算时间维度上的注意力
        x: [B, T, D] 输入特征矩阵
        """
        B, T, D = x.shape
        
        # 获取每个时刻的 query 和 kv
        query = x  # [B, T, D]
        key_value = x  # [B, T, D]

        # 计算注意力得分：query 和 key的点积
        attention_scores = torch.matmul(query, key_value.transpose(-2, -1))  # [B, T, T]
        attention_scores = attention_scores / math.sqrt(D)  # 缩放
        
        # Softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, T, T]
        
        # 通过注意力权重加权key和value
        output = torch.matmul(attention_weights, key_value)  # [B, T, D]
        
        # # 将 context 拼接到原始输入
        # output = torch.cat([x, output], dim=-1)  # [B, T, 2D]

        return output  # [B, T, D]

    def Temporal_window_attention(self, x: torch.Tensor, window_size: int = 3) -> torch.Tensor:
        """
        计算时间维度上的局部注意力（优化版，支持并行计算）
        x: [B, T, D] 输入特征矩阵
        window_size: 每个时间步关注的上下文窗口大小，默认为3
        """
        B, T, D = x.shape
        
        # 创建一个掩码矩阵，指定每个 query 只能与 window_size 范围内的 key 进行计算
        mask = torch.zeros(B, T, T, device=x.device)
        
        # 为每个 query 定义一个有效的 key 范围
        for i in range(T):
            start_idx = max(0, i - window_size)
            end_idx = min(T, i + window_size + 1)
            mask[:, i, start_idx:end_idx] = 1.0  # 设置有效范围为 1
        
        # 计算注意力得分：使用局部窗口，只与窗口中的 key 进行计算
        attention_scores = torch.matmul(x, x.transpose(-2, -1))  # [B, T, T]
        attention_scores = attention_scores / math.sqrt(D)  # 缩放
        
        # 使用 mask 屏蔽无效的 key（将无效的位置设置为 -inf）
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax 得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, T, T]
        
        # 通过注意力权重加权 key 和 value
        output = torch.matmul(attention_weights, x)  # [B, T, D]

        # 将结果拼接到原始输入上
        # output = torch.cat([x, output], dim=-1)  # [B, T, 2D]

        return output  # [B, T, D]


        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.position_encoder(self._input(x))


        return self.Temporal_attention(feature)

        
