#!/bin/bash

export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

# ===== Loss 权重说明 =====
# cos_weight: 方向损失权重（默认1.0）
# mse_weight: MSE幅度损失（默认0.1，建议0.05）
# l1_weight: L1幅度损失（默认0.05，建议0.02）
#
# 难例挖掘模式:
# --hard_mining_mode none      # 禁用
# --hard_mining_mode weighted  # 加权难例（推荐）
# --hard_mining_mode topk      # 只保留最难K个

# ===== 可选配置（取消注释使用）=====
# RESUME="--resume /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260127_v1/epoch_25.pt"
# DINOV2_CKPT="--dinov2_ckpt /workspace/zhuy25@xiaopeng.com/3dgs/model/dinov2_vitl14_reg4_pretrain.pth"

torchrun --nproc_per_node=4 train_vit_planner.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_train.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260206_v1 \
  --tensorboard \
  --val_log_every 10 \
  --save_every_steps 2000 \
  --epochs 50 --batch_size 32 --num_workers 16 \
  --lr 1e-5 \
  --lr_schedule cosine \
  --warmup_steps 1000 \
  --min_lr_ratio 0.1 \
  --grad_ckpt \
  --val_print_denorm_n 20 \
  --cos_weight 1.0 \
  --mse_weight 0.05 \
  --l1_weight 0.02 \
  --hard_mining_mode weighted \
  --hard_mining_ratio 0.7 \
  --hard_mining_boost 2.0
  # ${RESUME}
  # ${DINOV2_CKPT}
