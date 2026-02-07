#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ViT Diffusion Planner

Training script for ViT + Connector + Diffusion architecture.
Similar interface to train_vit_planner.py but adapted for diffusion training.
"""

import os
import sys
import json
import time
import argparse
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False

# Add project root to path
def _add_project_root():
    env_pyroot = os.environ.get("PLANNER_PYROOT", "").strip()
    if env_pyroot:
        sys.path.insert(0, env_pyroot)
_add_project_root()

# Import diffusion model
from planner.models.modeling.vit_diffusion_planner import (
    ViTDiffusionPlannerModel, 
    ViTDiffusionPlannerConfig
)

# Re-use dataset and utilities from train_vit_planner
# For simplicity, we inline the necessary components here

# =========================================================
# DDP utilities
# =========================================================
def ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_rank0() -> bool:
    return ddp_rank() == 0

def ddp_setup() -> None:
    if not ddp_is_enabled():
        return
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(ddp_local_rank())

def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m

def ddp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_enabled():
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def _resize_tensor_chw(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    try:
        import torchvision.transforms.functional as TF
        return TF.resize(x, size_hw, antialias=True)
    except:
        from PIL import Image
        import numpy as np
        c, h, w = x.shape
        outs = []
        for i in range(c):
            img = Image.fromarray((x[i].clamp(0, 1).cpu().numpy() * 255.0).astype("uint8"))
            img = img.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
            arr = torch.from_numpy(np.array(img)).float() / 255.0
            outs.append(arr)
        return torch.stack(outs, dim=0)

def load_depth_png_as_3ch_tensor(
    path: str,
    img_size: int,
    normalize: str = "unit",
) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    
    im = Image.open(path)
    arr = np.array(im)

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.dtype == np.uint16:
        x = torch.from_numpy(arr.astype("float32")) / 65535.0
    else:
        x = torch.from_numpy(arr.astype("float32"))
        if x.max() > 1.5:
            x = x / 255.0

    x = x.unsqueeze(0).repeat(3, 1, 1)
    x = _resize_tensor_chw(x, (img_size, img_size))

    if normalize == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = (x - mean) / std

    return x.float()

def find_auto_resume_ckpt(out_dir: str) -> str:
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        return ""
    pts = [os.path.join(out_dir, n) for n in os.listdir(out_dir) if n.endswith(".pt")]
    if not pts:
        return ""
    best = os.path.join(out_dir, "best.pt")
    if os.path.exists(best):
        return best
    pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pts[0]

# =========================================================
# TensorBoard helpers
# =========================================================
def tb_add_scalar(writers, tag, value, step):
    if writers:
        for w in writers:
            if w:
                w.add_scalar(tag, value, step)

def tb_flush(writers):
    if writers:
        for w in writers:
            if w:
                w.flush()

def tb_close(writers):
    if writers:
        for w in writers:
            if w:
                w.close()

# =========================================================
# Dataset (same as original)
# =========================================================
class Depth4FrameVelCmdDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        img_size: int,
        frames: int = 4,
        state_dim: int = 10,
        use_state: bool = False,
        normalize: str = "unit",
        filter_max_dt: Optional[float] = None,
        eps: float = 1e-6,
        desired_vel_min: float = 0.1,
    ):
        self.items = read_jsonl(jsonl_path)
        self.img_size = img_size
        self.frames = frames
        self.state_dim = state_dim
        self.use_state = use_state
        self.normalize = normalize
        self.filter_max_dt = filter_max_dt
        self.eps = float(eps)
        self.desired_vel_min = float(desired_vel_min)

        if self.filter_max_dt is not None:
            kept = []
            for it in self.items:
                dt = float(it.get("match_dt_sec", 0.0))
                if dt <= self.filter_max_dt:
                    kept.append(it)
            self.items = kept

    def __len__(self) -> int:
        return len(self.items)

    def _safe_unit3(self, v: List[float]) -> List[float]:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        n = (x * x + y * y + z * z) ** 0.5
        if n < self.eps:
            return [0.0, 0.0, 0.0]
        inv = 1.0 / n
        return [x * inv, y * inv, z * inv]

    def _get_desired_vel_den(self, it: Dict[str, Any], label_velcmd: List[float]) -> float:
        cs = it.get("curr_state", {}) or {}
        dv = cs.get("desired_vel", None)
        if dv is not None:
            try:
                dv = float(dv)
            except Exception:
                dv = None
        if dv is None:
            lx, ly, lz = float(label_velcmd[0]), float(label_velcmd[1]), float(label_velcmd[2])
            dv = (lx * lx + ly * ly + lz * lz) ** 0.5
        if dv < self.desired_vel_min:
            dv = self.desired_vel_min
        return float(dv)

    def _build_state(self, it: Dict[str, Any]) -> torch.Tensor:
        if not self.use_state:
            return torch.zeros(self.state_dim, dtype=torch.float32)

        cs = it.get("curr_state", None)
        gn = it.get("goal_nav", None)
        if cs is None or gn is None:
            return torch.zeros(self.state_dim, dtype=torch.float32)

        quat = cs.get("quat_wxyz", None)
        vel_world = cs.get("vel_world", None)
        delta = gn.get("delta_pos_body", None)

        if quat is None or vel_world is None or delta is None:
            return torch.zeros(self.state_dim, dtype=torch.float32)

        quat4 = [float(x) for x in quat]
        goal_dir = self._safe_unit3([float(x) for x in delta])
        vel_dir = self._safe_unit3([float(x) for x in vel_world])

        s = quat4 + goal_dir + vel_dir
        return torch.tensor(s, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        frame_paths = it["frame_paths"]

        imgs = [load_depth_png_as_3ch_tensor(p, self.img_size, normalize=self.normalize) for p in frame_paths]
        images = torch.stack(imgs, dim=0)

        label = it.get("label_velcmd", None)
        den = self._get_desired_vel_den(it, label)
        y = torch.tensor([float(label[0]) / den, float(label[1]) / den, float(label[2]) / den], dtype=torch.float32)

        state = self._build_state(it)

        return {"images": images, "state": state, "label": y, "label_den": torch.tensor(den, dtype=torch.float32)}

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["images"] for b in batch], dim=0)
    state = torch.stack([b["state"] for b in batch], dim=0)
    label = torch.stack([b["label"] for b in batch], dim=0)
    label_den = torch.stack([b["label_den"] for b in batch], dim=0)
    return {"images": images, "state": state, "label": label, "label_den": label_den}

# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    grad_clip: float,
    log_every: int,
    writers: Optional[List],
    global_step: int,
    save_every_steps: int,
    out_dir: str,
    epoch: int,
    cfg_dict: Dict[str, Any],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[Dict[str, float], int]:
    model.train()
    
    total_loss = 0.0
    total_noise_mag = 0.0
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
            # Diffusion forward pass
            loss, metrics = model.forward_train(images, state, label)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_noise_mag += float(metrics.get("noise_magnitude", 0)) * bs
        n += bs

        if is_rank0() and writers is not None:
            tb_add_scalar(writers, "train/loss", float(loss.item()), global_step)
            tb_add_scalar(writers, "train/noise_magnitude", float(metrics.get("noise_magnitude", 0)), global_step)
            tb_add_scalar(writers, "train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

        if log_every > 0 and step % log_every == 0 and is_rank0():
            dt = time.time() - t0
            rate = n / max(1e-6, dt)
            avg = total_loss / max(1, n)
            print(f"[train] step={step}/{len(loader)} loss={avg:.6f} samples/s={rate:.1f}")

        if save_every_steps and (global_step > 0) and (global_step % save_every_steps == 0) and is_rank0():
            ckpt_path = os.path.join(out_dir, f"step_{global_step}.pt")
            save_ckpt(ckpt_path, model, optimizer, epoch, cfg=cfg_dict, global_step=global_step, scheduler=scheduler)
            print(f"[ckpt] saved: {ckpt_path}")

        global_step += 1

        if scheduler is not None:
            scheduler.step(global_step)

    epoch_avg = total_loss / max(1, n)
    return {"loss": epoch_avg}, global_step

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_inference_steps: int = 50,
    val_log_every: int = 0,
) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")

    loss_sum = torch.tensor(0.0, device=device)
    denorm_sum = torch.tensor(0.0, device=device)
    n_sum = torch.tensor(0.0, device=device)
    
    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        label_den = batch["label_den"].to(device, non_blocking=True)

        # Diffusion inference
        pred = model.forward_inference(images, state, num_steps=num_inference_steps)
        
        # Compute loss on denormalized values
        den = label_den.view(-1, 1).clamp_min(1e-6)
        pred_den = pred * den
        label_denorm = label * den
        loss = loss_fn(pred_den, label_denorm)

        bs = images.size(0)
        loss_sum += loss.detach() * bs
        denorm_sum += loss.detach() * bs
        n_sum += torch.tensor(float(bs), device=device)

        if val_log_every and (step % val_log_every == 0) and is_rank0():
            dt = time.time() - t0
            seen = float(n_sum.item())
            avg = float((loss_sum / torch.clamp(n_sum, min=1.0)).item())
            print(f"[val] step={step}/{len(loader)} partial_mse={avg:.6f} seen={seen:.0f}")

    loss_sum = ddp_all_reduce_sum(loss_sum)
    n_sum = ddp_all_reduce_sum(n_sum)

    mse = (loss_sum / torch.clamp(n_sum, min=1.0)).item()

    return {"loss": float(mse), "mse": float(mse)}

def save_ckpt(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Dict[str, Any],
    global_step: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)

def build_warmup_cosine_lambda(total_steps: int, warmup_steps: int, min_lr_ratio: float):
    total_steps = int(max(1, total_steps))
    warmup_steps = int(max(0, warmup_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int):
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda

# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--normalize", type=str, default="unit", choices=["unit", "imagenet", "none"])
    ap.add_argument("--use_state", action="store_true")
    ap.add_argument("--state_dim", type=int, default=10)
    ap.add_argument("--desired_vel_min", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=1e-6)

    # Model (Diffusion specific)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--dim_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_velocity_tokens", type=int, default=4)
    
    # Diffusion params
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--beta_start", type=float, default=0.0001)
    ap.add_argument("--beta_end", type=float, default=0.02)
    
    # ViT params
    ap.add_argument("--vggt_embed_dim", type=int, default=1024)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_patches", type=int, default=512)
    ap.add_argument("--vit_spatial_layers", type=int, default=2)
    ap.add_argument("--vit_temporal_layers", type=int, default=1)

    # Training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_log_every", type=int, default=0)
    ap.add_argument("--save_every_steps", type=int, default=2000)
    
    # LR schedule
    ap.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    
    # Misc
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./outputs_diffusion")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--resume", type=str, default="")

    args = ap.parse_args()
    set_seed(args.seed)
    ddp_setup()

    device = f"cuda:{ddp_local_rank()}" if ddp_is_enabled() else ("cuda" if torch.cuda.is_available() else "cpu")
    if is_rank0():
        print(f"device: {device}, ddp: {ddp_is_enabled()}, world_size: {ddp_world_size()}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Datasets
    train_ds = Depth4FrameVelCmdDataset(
        jsonl_path=args.train_jsonl,
        img_size=args.img_size,
        frames=args.frames,
        state_dim=args.state_dim,
        use_state=args.use_state,
        normalize=args.normalize,
    )
    val_ds = Depth4FrameVelCmdDataset(
        jsonl_path=args.val_jsonl,
        img_size=args.img_size,
        frames=args.frames,
        state_dim=args.state_dim,
        use_state=args.use_state,
        normalize=args.normalize,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if ddp_is_enabled() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if ddp_is_enabled() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Model config
    cfg = ViTDiffusionPlannerConfig(
        state_dim=args.state_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        num_velocity_tokens=args.num_velocity_tokens,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        img_size=args.img_size,
        vggt_embed_dim=args.vggt_embed_dim,
        vit_num_frames=args.frames,
        vit_spatial_layers=args.vit_spatial_layers,
        vit_temporal_layers=args.vit_temporal_layers,
        max_patches=args.max_patches,
    )

    model = ViTDiffusionPlannerModel(cfg).to(device)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # Resume
    start_epoch = 1
    global_step = 0
    resume_path = (args.resume or "").strip() or find_auto_resume_ckpt(args.out_dir)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        if is_rank0():
            print(f"[resume] from {resume_path}, start_epoch={start_epoch}, global_step={global_step}")

    if ddp_is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp_local_rank()], output_device=ddp_local_rank()
        )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    # Scheduler
    scheduler = None
    total_steps = int(args.epochs) * len(train_loader)
    if args.lr_schedule == "cosine":
        lr_lambda = build_warmup_cosine_lambda(total_steps, args.warmup_steps, args.min_lr_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if global_step > 0:
            scheduler.step(global_step)

    # TensorBoard
    tb_writers = None
    if args.tensorboard and is_rank0():
        if _HAS_TB:
            local_tb = os.path.join(args.out_dir, "tb")
            os.makedirs(local_tb, exist_ok=True)
            tb_writers = [SummaryWriter(log_dir=local_tb)]
            print(f"[tb] {local_tb}")

    cfg_dict = asdict(cfg)

    best_val = float("inf")
    epoch = start_epoch

    while epoch <= args.epochs:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_rank0():
            print(f"\n===== Epoch {epoch} =====")

        tr, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            writers=tb_writers,
            global_step=global_step,
            save_every_steps=args.save_every_steps,
            out_dir=args.out_dir,
            epoch=epoch,
            cfg_dict=cfg_dict,
            scheduler=scheduler,
        )

        if is_rank0():
            print(f"[train] epoch={epoch} loss={tr['loss']:.6f}")

        va = evaluate(
            model,
            val_loader,
            device=device,
            num_inference_steps=args.num_inference_steps,
            val_log_every=args.val_log_every,
        )
        
        if is_rank0():
            print(f"[val]   epoch={epoch} mse={va['mse']:.6f}")
            if tb_writers:
                tb_add_scalar(tb_writers, "val/mse", va["mse"], epoch)

            if va["loss"] < best_val:
                best_val = va["loss"]
                save_ckpt(
                    os.path.join(args.out_dir, "best.pt"),
                    model, optimizer, epoch, cfg=cfg_dict, global_step=global_step, scheduler=scheduler
                )

        epoch += 1

    ddp_cleanup()

if __name__ == "__main__":
    main()
