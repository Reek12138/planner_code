#!/bin/bash

# Training script for ViT + Diffusion Planner

export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

# Default configuration for diffusion model
# You can override these via command line arguments

torchrun --nproc_per_node=4 train_vit_diffusion_planner.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_train.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260207_diffusion_v1 \
  --tensorboard \
  --val_log_every 10 \
  --save_every_steps 2000 \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 16 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --warmup_steps 1000 \
  --min_lr_ratio 0.1 \
  --grad_clip 1.0 \
  \
  # ===== Diffusion-specific params =====
  --d_model 256 \
  --nhead 8 \
  --num_layers 6 \
  --dim_ff 2048 \
  --dropout 0.1 \
  --num_velocity_tokens 4 \
  \
  # Diffusion schedule
  --num_train_timesteps 1000 \
  --num_inference_steps 50 \
  --beta_start 0.0001 \
  --beta_end 0.02 \
  \
  # ViT params
  --vggt_embed_dim 1024 \
  --vit_spatial_layers 2 \
  --vit_temporal_layers 1 \
  --max_patches 512 \
  \
  # --resume /path/to/checkpoint.pt
