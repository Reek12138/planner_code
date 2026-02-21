#!/bin/bash

export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

# ============================================================
# ViT Planner 轴向加权训练脚本 V3 - 深度Encoder版本
# 
# 针对问题：侧向(Y)和垂直(Z)方向精度不足 + 验证集过拟合
# 解决方案：
#   1. 增加 ViT Encoder 深度 (spatial_layers 2->4, temporal_layers 1->2)
#   2. 调整损失权重，更关注幅度误差
#   3. 提高学习率，增加训练轮数
#   4. 增强数据增强
#
# 需要配合修改 train_vit_planner.py 中的模型配置！
# ============================================================

torchrun --nproc_per_node=4 train_vit_planner.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260219_deeper_encoder_lr_small_mix_v3 \
  --tensorboard \
  --val_log_every 10 \
  --val_print_denorm_n 20 \
  --save_every_steps 2000 \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 16 \
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
# ⚠️ 重要：必须同时修改 train_vit_planner.py 中的模型配置！
# 
# 修改位置：在 train_vit_planner.py 中搜索 "vit_spatial_layers" 和 "vit_temporal_layers"
# 大约在第 1358-1359 行附近
#
# 原配置：
#   vit_spatial_layers=2,
#   vit_temporal_layers=1,
#
# 改为：
#   vit_spatial_layers=4,      # 帧内 attention 层数从 2 增加到 4
#   vit_temporal_layers=2,     # 跨帧 attention 层数从 1 增加到 2
#
# 如果显存允许，可以进一步改为：
#   vit_spatial_layers=6,
#   vit_temporal_layers=2,
# ============================================================
