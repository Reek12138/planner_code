export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

torchrun --nproc_per_node=4 train_vit_planner.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_train.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260127_v1 \
  --freeze_vggt \
  --tensorboard \
  --val_log_every 10 \
  --save_every_steps 2000 \
  --epochs 25 --batch_size 32 --num_workers 8 \
  --lr 2e-4 \
  --lr_schedule cosine \
  --warmup_steps 2000 \
  --min_lr_ratio 0.1 \
  --grad_ckpt
  # --resume /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260122_v1/best.pt
