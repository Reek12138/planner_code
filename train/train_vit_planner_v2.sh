#!/bin/bash

export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

# ============================================================
# ViT Planner V2 训练脚本
# 
# V2 新特性：
#   1. 输入图像 128x128（更小，更快）
#   2. Encoder 使用 RoPE (Rotary Position Embedding)
#   3. Connector 使用 2x2 token 压缩
# 
# 保持 Decoder 和训练逻辑与 V1 完全一致
# ============================================================

torchrun --nproc_per_node=4 train_vit_planner_v2.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_train.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --img_size 140 \
  --patch_size 14 \
  --vggt_embed_dim 2048 \
  --use_rope \
  --use_2x2_compression \
  --compression_num_layers 2 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260221_vit_planner_v2_140_rope_compression \
  --tensorboard \
  --val_log_every 10 \
  --val_print_denorm_n 20 \
  --save_every_steps 5000 \
  --epochs 100 \
  --batch_size 64 \
  --num_workers 32 \
  --lr 1e-6 \
  --lr_schedule cosine \
  --warmup_steps 2000 \
  --min_lr_ratio 0.05 \
  --cos_weight 0.4 \
  --mse_weight 0.4 \
  --l1_weight 0.2 \
  --use_dynamic_weights \
  --axis_weights "1.0,2.5,2.5" \
  --hard_mining_mode focal \
  --hard_mining_ratio 0.5 \
  --augment \
  --aug_noise_scale 0.01 \
  --aug_dropout_prob 0.05 \
  --grad_ckpt

# ============================================================
# V2 模型配置说明：
# 
# 1. 输入尺寸: 128x128 (默认) 可通过 --img_size 修改
#    - 原始尺寸 518x518 时 patch 数量: (518/14)^2 ≈ 1369
#    - 128x128 时 patch 数量: (128/14)^2 ≈ 81
# 
# 2. RoPE 位置编码:
#    - 空间维度使用旋转位置编码 (RoPE)
#    - 时间维度仍使用可学习的绝对位置编码
#    - 可通过 --no_rope 禁用
#
# 3. 2x2 Token 压缩:
#    - 将空间上相邻的 2x2 个 patch 合并为 1 个 token
#    - 使用 MLP 进行压缩: 4*D_in -> D_out
#    - 压缩后 token 数量减少为 1/4
#    - 可通过 --no_2x2_compression 禁用
#
# 4. Decoder 逻辑保持不变
# ============================================================
