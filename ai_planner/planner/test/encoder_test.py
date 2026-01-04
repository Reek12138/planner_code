import torch

import sys
import os

# 将 encoder.py 所在的目录添加到系统路径中
sys.path.append(os.path.abspath('/home/zhuyi/planner_code/ai_planner/planner/models'))

# 然后你就可以导入 encoder 模块了
from encoder import Feature_Encoder


# 假设 input_dim 是 5 + lidar_dim，其中 lidar_dim 是 30
input_dim = 35  # 5 (基础特征) + 30 (雷达特征)
seq_len = 6     # 序列长度 T
B = 4           # 批次大小 B

# 创建一个随机的输入张量，形状为 [B, T, D]
x = torch.randn(B, seq_len, input_dim)

# 初始化 Feature_Encoder
encoder = Feature_Encoder(input_dim=input_dim, seq_len=seq_len, use_stateful=True, fill_with_first=True)

# 测试 encoder
output = encoder(x)

print(f"Output shape: {output.shape}")
