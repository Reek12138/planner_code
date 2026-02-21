#!/bin/bash

export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

# ============================================================
# ViT Planner 轴向加权训练脚本 V2
# 
# 针对问题：侧向(Y)和垂直(Z)方向精度不足
# 解决方案：给 Y/Z 轴更高的损失权重
#
# 观察到的现象：
#   - #17: Y方向偏差大 (0.263 vs 0.378)
#   - #18: Y方向偏差大 (-0.081 vs -0.390)
#   - X方向（前进）预测准确
# ============================================================

torchrun --nproc_per_node=4 train_vit_planner.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_train.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260215_axis_weighted_v1 \
  --tensorboard \
  --val_log_every 10 \
  --val_print_denorm_n 20 \
  --save_every_steps 2000 \
  --epochs 60 \
  --batch_size 32 \
  --num_workers 16 \
  --lr 2e-5 \
  --lr_schedule cosine \
  --warmup_steps 2000 \
  --min_lr_ratio 0.05 \
  --cos_weight 0.8 \
  --mse_weight 0.15 \
  --l1_weight 0.08 \
  --use_dynamic_weights \
  --axis_weights "1.0,2.5,1.5" \
  --hard_mining_mode focal \
  --hard_mining_ratio 0.5 \
  --augment \
  --aug_noise_scale 0.005 \
  --aug_dropout_prob 0.03 \
  --grad_ckpt

# ============================================================
# 其他建议的轴向权重配置：
#
# 1. 平衡模式: "1.0,1.0,1.0" (默认，均匀权重)
# 2. 侧向重点: "1.0,2.0,1.0" (Y轴加倍)
# 3. 垂直重点: "1.0,1.0,2.0" (Z轴加倍)
# 4. 全方位: "1.0,1.5,1.5" (Y/Z都加强)
#
# 根据你的数据，建议使用 "1.0,2.0,1.5" 或 "1.0,2.0,1.0"
# ============================================================
