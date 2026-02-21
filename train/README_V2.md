# ViT Planner V2 使用说明

## 概述

ViT Planner V2 是基于 V1 的改进版本，保持原有训练逻辑不变，主要改进如下：

1. **输入尺寸灵活**: 默认 128x128 (可配置)
2. **RoPE 位置编码**: 空间维度使用旋转位置编码 (Rotary Position Embedding)
3. **2x2 Token 压缩**: Connector 中支持将空间相邻的 2x2 个 patch 压缩为 1 个 token

## 文件结构

```
planner_code/ai_planner/planner/models/
├── encoder/
│   ├── vision_encoder.py      # V1 原版（保持不变）
│   └── vision_encoder_v2.py   # V2: 增加 RoPE 支持
├── connector/
│   ├── connector.py           # V1 原版（保持不变）
│   └── connector_v2.py        # V2: 增加 2x2 压缩支持
└── modeling/
    ├── vit_planner.py         # V1 原版（保持不变）
    └── vit_planner_v2.py      # V2: 整合新组件

planner_code/train/
├── train_vit_planner.py       # V1 训练脚本（保持不变）
├── train_vit_planner_v2.py    # V2 训练脚本
├── train_vit_planner_v2.sh    # V2 启动脚本
└── README_V2.md               # 本文档
```

## 快速开始

### 1. 使用默认配置训练 (128x128 + RoPE + 2x2压缩)

```bash
cd /root/workspace/zhuy25@xiaopeng.com/planner_code/train
bash train_vit_planner_v2.sh
```

### 2. 自定义配置

```bash
# 使用 224x224 输入
torchrun --nproc_per_node=4 train_vit_planner_v2.py \
  --train_jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl \
  --img_size 224 \
  --use_rope \
  --use_2x2_compression \
  ...

# 禁用 RoPE (使用绝对位置编码)
torchrun --nproc_per_node=4 train_vit_planner_v2.py \
  --img_size 128 \
  --no_rope \
  --use_2x2_compression \
  ...

# 禁用 2x2 压缩
torchrun --nproc_per_node=4 train_vit_planner_v2.py \
  --img_size 128 \
  --use_rope \
  --no_2x2_compression \
  ...
```

## 命令行参数

### V2 特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--img_size` | 128 | 输入图像尺寸 (V2 默认改为 128) |
| `--use_rope` | True | 使用 RoPE 位置编码 |
| `--no_rope` | False | 禁用 RoPE，使用绝对位置编码 |
| `--use_2x2_compression` | True | 使用 2x2 token 压缩 |
| `--no_2x2_compression` | False | 禁用 2x2 token 压缩 |
| `--compression_hidden_dim` | None | 2x2 压缩 MLP 隐藏层维度 |
| `--compression_num_layers` | 2 | 2x2 压缩 MLP 层数 |

### 其他参数

V2 训练脚本完全兼容 V1 的所有参数，如：
- `--vggt_embed_dim`: embedding 维度 (default: 2048)
- `--patch_size`: patch 大小 (default: 14)
- `--frames`: 输入帧数 (default: 4)
- `--batch_size`, `--lr`, `--epochs` 等训练参数

## 技术细节

### 1. RoPE (Rotary Position Embedding)

```python
# 实现位置: vision_encoder_v2.py

class RoPEMultiheadSelfAttention(nn.Module):
    def forward(self, x):
        # 计算 RoPE 的 cos/sin 缓存
        cos, sin = self._compute_rope(seq_len, device, dtype)
        # 应用到 q 和 k
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # 标准 attention 计算
        ...
```

- 空间维度使用 RoPE
- 时间维度仍使用可学习的绝对位置编码 (`time_embed`)

### 2. 2x2 Token 压缩

```python
# 实现位置: connector_v2.py

class PatchCompressor2x2(nn.Module):
    def forward(self, x):
        # 输入: [B, H*W, D] 空间上的 patches
        # 重排成 [B, H, W, D]
        # 分割成 2x2 块: [B, H/2, W/2, 2, 2, D]
        # 拼接 4 个 patch: [B, H/2, W/2, 4*D]
        # MLP 压缩到 [B, H/2, W/2, D_out]
        ...
```

例如 128x128 输入：
- Patch 数量: (128/14)^2 ≈ 81 (9x9)
- 2x2 压缩后: 约 16 个 tokens (4x4)

### 3. 与 V1 的关系

| 特性 | V1 | V2 |
|------|-----|-----|
| 默认输入尺寸 | 518x518 | 128x128 |
| 空间位置编码 | 绝对位置编码 | RoPE (可选) |
| Token 压缩 | 无 | 2x2 压缩 (可选) |
| Decoder | Transformer | Transformer (相同) |
| Head | VelocityHeadUnit | VelocityHeadUnit (相同) |

## 实验建议

1. **从默认配置开始**: 128x128 + RoPE + 2x2压缩
2. **比较实验**:
   - 对比 `--no_rope` 和 `--use_rope` 的效果
   - 对比 `--no_2x2_compression` 和 `--use_2x2_compression` 的效果
3. **调整输入尺寸**: 如果精度不够，可尝试 `--img_size 224`

## 注意事项

1. V2 模型使用全新的类名，与 V1 不兼容（不能直接加载 V1 的检查点）
2. 如需从 V1 迁移，需要重新训练
3. 128x128 输入显存占用更小，可以尝试更大的 batch_size
