export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner
export PYTHONPATH=$PLANNER_PYROOT:$PYTHONPATH

torchrun --nproc_per_node=4 train_vit_planner_dino.py \
  --train_jsonl /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_train.jsonl \
  --val_jsonl   /workspace/zhuy25@xiaopeng.com/planner_code/datasets/metadata_val.jsonl \
  --use_state \
  --state_dim 10 \
  --out_dir  /workspace/group_share/adc-perception-mlinfra/zhuy25/ai_planner/output/20260128_v1 \
  --tensorboard \
  --val_log_every 10 \
  --save_every_steps 2000 \
  --epochs 25 --batch_size 32 --num_workers 8 \
  --lr 1e-5 \
  --lr_schedule cosine \
  --warmup_steps 1000 \
  --min_lr_ratio 0.1 \
  --grad_ckpt \
  --normalize imagenet \
  --depth_mode repeat \
  --dinov2_ckpt /workspace/zhuy25@xiaopeng.com/3dgs/model/dinov2_vitl14_reg4_pretrain.pth \
  --val_print_denorm_n 20

  # --dinov2_ckpt /workspace/group_share/adc-perception-autolabel/zhouhy_tmp/dinov2_vitl14_reg4_pretrain.pth
  # /workspace/zhuy25@xiaopeng.com/3dgs/model/dinov2_vitl14_reg4_pretrain.pth
