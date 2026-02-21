#!/bin/bash

export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

# ============================================================
# ViT Planner 改进版训练脚本
# 新功能：
#   1. 分层学习率 (encoder 慢, decoder/head 快)
#   2. 动态损失权重 (前期重方向, 后期重幅度)
#   3. Focal Loss 难例挖掘
#   4. 数据增强 (噪声 + 随机遮挡)
# ============================================================

torchrun --nproc_per_node=4 train_vit_planner.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_train.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260210_vit_improved_v1 \
  --tensorboard \
  --val_log_every 10 \
  --val_print_denorm_n 20 \
  --save_every_steps 2000 \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 16 \
  --lr 2e-5 \
  --lr_schedule cosine \
  --warmup_steps 2000 \
  --min_lr_ratio 0.05 \
  --cos_weight 1.0 \
  --mse_weight 0.1 \
  --l1_weight 0.05 \
  --use_dynamic_weights \
  --hard_mining_mode focal \
  --hard_mining_ratio 0.5 \
  --augment \
  --aug_noise_scale 0.01 \
  --aug_dropout_prob 0.05 \
  --grad_ckpt                     # 启用 gradient checkpointing
  
# 注意：如需调整模型结构 (更深的 decoder, 更高的 dropout)，
# 请直接修改 train_vit_planner.py 中第 1193-1220 行的 VelocityPlannerConfig
