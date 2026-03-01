#!/bin/bash

export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

# ============================================================
# ViT Planner V2 Lightweight 训练脚本
# 
# 轻量级配置：参数量约 500MB（相比原版 3.6GB 减少约 7 倍）
#
# 主要改动：
#   1. Encoder: embed_dim 2048->896, layers 6->4, temporal 2->1
#   2. Decoder: d_model 2048->512, layers 8->6, dim_ff 2048->1152
#   3. 其他配置保持与 V2 一致
# ============================================================

torchrun --nproc_per_node=4 train_vit_planner_v2_lightweight.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_train.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --img_size 112 \
  --patch_size 14 \
  --vggt_embed_dim 896 \
  --vit_spatial_layers 4 \
  --vit_temporal_layers 1 \
  --vit_nhead 8 \
  --d_model 512 \
  --decoder_nhead 8 \
  --decoder_num_layers 6 \
  --decoder_dim_ff 1152 \
  --use_rope \
  --use_2x2_compression \
  --compression_num_layers 2 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260224_vit_planner_v3_lightweight_500m \
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
# Lightweight 模型配置说明：
# 
# 1. Encoder 参数调整:
#    - embed_dim: 2048 -> 896 (2.3x 减少)
#    - spatial_layers: 6 -> 4 
#    - temporal_layers: 2 -> 1 (2x 减少)
#    - nhead: 16 -> 8
#    
# 2. Decoder 参数调整:
#    - d_model: 2048 -> 512 (4x 减少)
#    - num_layers: 8 -> 6
#    - dim_ff: 2048 -> 1152
#    - nhead: 16 -> 8
#    
# 3. 预期参数量对比:
#    - 原版: ~3.6 GB (d_model=2048, embed_dim=2048)
#    - Lightweight: ~505 MB (约 1/7)
#
# 4. 图像尺寸说明:
#    - 使用 112x112 而非 128x128，因为 112 = 14 * 8
#    - 这样可以保证 grid 为 8x8 = 64 patches，是偶数，支持 2x2 压缩
#    - 压缩后 tokens: 64 / 4 = 16
#
# 5. 保持的功能:
#    - RoPE 位置编码
#    - 2x2 token 压缩
#    - 相同的训练逻辑和损失函数
# ============================================================
