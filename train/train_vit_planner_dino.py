#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train VelocityPlannerModel with metadata.jsonl (4-frame depth -> 3-axis velcmd)

✅ 支持：
- 单卡 / 多卡多 rank（torchrun DDP）
- TensorBoard 记录（train/val loss、lr、samples/s）
- eval 指标在 DDP 下做全局聚合（all-reduce）
- 自动断点续训（保持原逻辑）
- 可选 warmup + cosine decay（默认 constant）
- 可选 activation checkpointing（默认关闭）

✅ 本次你要的改动（保持训练流程/接口不变，新增能力）：
1) 默认 DINO 对齐：
   - normalize 默认 imagenet
   - vit/vggt_embed_dim 默认 1024
   - vit_nhead 默认 16
   - patch_size 默认 14
   - decoder d_model 仍用 256（不必对齐 DINO，训练更稳）
2) 支持 --depth_mode：
   - repeat / log / standardize
3) 支持 --dinov2_ckpt 本地 init：
   - 在 DDP wrap 前调用：model.vggt.init_from_dinov2(local_ckpt_path=...)
   - 需要你的 vision_encoder.py 里 init_from_dinov2 支持 local_ckpt_path
     （下面我在本脚本里做了“尽量兼容”的调用：优先 local_ckpt_path，否则传 state_dict）

用法示例：
  torchrun --nproc_per_node=8 train.py \
    --train_jsonl xxx --val_jsonl yyy \
    --use_state \
    --dinov2_ckpt /path/to/dinov2_vitl14_pretrain.pth \
    --depth_mode log \
    --normalize imagenet

"""

import os
import json
import time
import argparse
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("PIL is required: pip install pillow") from e

try:
    import torchvision.transforms.functional as TF
    _HAS_TV = True
except Exception:
    _HAS_TV = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False

import sys


def _add_project_root():
    candidates = []

    env_pyroot = os.environ.get("PLANNER_PYROOT", "").strip()
    if env_pyroot:
        candidates.append(env_pyroot)

    env_root = os.environ.get("PLANNER_ROOT", "").strip()
    if env_root:
        candidates.append(os.path.join(env_root, "ai_planner"))
        candidates.append(env_root)

    this = os.path.abspath(__file__)
    cur = os.path.dirname(this)
    for _ in range(10):
        candidates.append(os.path.join(cur, "ai_planner"))
        candidates.append(cur)
        cur = os.path.dirname(cur)

    uniq = []
    seen = set()
    for p in candidates:
        p = os.path.abspath(p)
        if p in seen:
            continue
        seen.add(p)
        if os.path.isdir(p):
            uniq.append(p)

    for p in uniq:
        if os.path.isdir(os.path.join(p, "planner")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return

    msg = ["[FATAL] Cannot find python root that contains `planner/` package."]
    msg.append("Tried these candidates:")
    msg.extend([f"  - {p} (exists={os.path.isdir(p)})" for p in uniq[:20]])
    msg.append("Hint: set env var:")
    msg.append("  export PLANNER_PYROOT=/workspace/zhuy25@xiaopeng.com/planner_code/ai_planner")
    msg.append("Current sys.path head:")
    msg.extend([f"  - {p}" for p in sys.path[:10]])
    raise RuntimeError("\n".join(msg))


_add_project_root()

# ✅ IMPORTANT: switch planner import (keep interface, change backend)
from planner.models.modeling.vit_planner_dino import VelocityPlannerConfig, VelocityPlannerModel  # noqa: E402


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
    if _HAS_TV:
        return TF.resize(x, size_hw, antialias=True)

    c, h, w = x.shape
    outs = []
    import numpy as np
    for i in range(c):
        img = Image.fromarray((x[i].clamp(0, 1).cpu().numpy() * 255.0).astype("uint8"))
        img = img.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        outs.append(arr)
    return torch.stack(outs, dim=0)


def load_depth_png_as_3ch_tensor(
    path: str,
    img_size: int,
    normalize: str = "imagenet",
    depth_mode: str = "log",   # NEW: repeat/log/standardize
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Load depth png -> 3ch tensor [3,H,W], then resize to img_size, then optional imagenet norm.

    depth_mode:
      - repeat: depth in [0,1] repeat to 3ch
      - log:    log(depth) then standardize then map to [0,1] then repeat
      - standardize: z-score then map to [0,1] then repeat
    """
    im = Image.open(path)
    import numpy as np
    arr = np.array(im)

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.dtype == np.uint16:
        d = torch.from_numpy(arr.astype("float32")) / 65535.0
    else:
        d = torch.from_numpy(arr.astype("float32"))
        if d.max() > 1.5:
            d = d / 255.0

    # d: [H,W]
    if depth_mode == "repeat":
        x = d.unsqueeze(0).repeat(3, 1, 1)

    elif depth_mode == "log":
        d2 = torch.log(d.clamp_min(eps))
        d2 = (d2 - d2.mean()) / (d2.std() + eps)
        d2 = d2.clamp(-3, 3)
        d2 = (d2 + 3) / 6.0  # -> ~[0,1]
        x = d2.unsqueeze(0).repeat(3, 1, 1)

    elif depth_mode == "standardize":
        d2 = (d - d.mean()) / (d.std() + eps)
        d2 = d2.clamp(-3, 3)
        d2 = (d2 + 3) / 6.0
        x = d2.unsqueeze(0).repeat(3, 1, 1)

    else:
        raise ValueError(f"Unknown depth_mode={depth_mode}")

    x = _resize_tensor_chw(x, (img_size, img_size))

    if normalize == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        x = (x - mean) / std
    elif normalize in ("unit", "none"):
        pass
    else:
        raise ValueError(f"Unknown normalize={normalize}")

    return x.float()


def find_auto_resume_ckpt(out_dir: str) -> str:
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        return ""

    pts = []
    for name in os.listdir(out_dir):
        if name.endswith(".pt"):
            pts.append(os.path.join(out_dir, name))

    if not pts:
        return ""

    best = os.path.join(out_dir, "best.pt")
    if os.path.exists(best) and os.path.isfile(best):
        return best

    pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pts[0]


# =========================================================
# TensorBoard dual-writer helpers
# =========================================================
def _sanitize_job_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return name
    import re
    name = re.sub(r"-master-\d+$", "", name)
    return name


def _infer_job_name() -> str:
    for k in ["FUYAO_JOB_NAME", "JOB_NAME", "SLURM_JOB_NAME", "LSB_JOBNAME"]:
        v = os.environ.get(k, "")
        v2 = _sanitize_job_name(v)
        if v2:
            return v2

    try:
        import socket
        hn = socket.gethostname()
        hn2 = _sanitize_job_name(hn)
        if hn2:
            return hn2
    except Exception:
        pass

    return "velocity_planner_job"


def _infer_job_logdir(base_root: str = "/workspace/zhuy25@xiaopeng.com") -> str:
    job_name = _infer_job_name()
    job_dir = os.path.join(base_root, job_name)
    return os.path.join(job_dir, "xflow_logs")


def tb_add_scalar(writers: Optional[List["SummaryWriter"]], tag: str, value: float, step: int) -> None:
    if not writers:
        return
    for w in writers:
        if w is None:
            continue
        w.add_scalar(tag, value, step)


def tb_flush(writers: Optional[List["SummaryWriter"]]) -> None:
    if not writers:
        return
    for w in writers:
        if w is None:
            continue
        w.flush()


def tb_close(writers: Optional[List["SummaryWriter"]]) -> None:
    if not writers:
        return
    for w in writers:
        if w is None:
            continue
        w.close()


# =========================================================
# LR Scheduler helpers (optional)
# =========================================================
def build_warmup_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    total_steps = int(max(1, total_steps))
    warmup_steps = int(max(0, warmup_steps))
    min_lr_ratio = float(min_lr_ratio)

    if warmup_steps >= total_steps:
        def lr_lambda(step: int):
            step = int(step)
            if warmup_steps <= 0:
                return 1.0
            return min(1.0, float(step + 1) / float(warmup_steps))
        return lr_lambda

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
# Dataset
# =========================================================
class Depth4FrameVelCmdDataset(Dataset):
    """
    jsonl item expected:
      - frame_paths: List[str] length=4
      - label_velcmd: List[float] length=3
      - curr_state.quat_wxyz: [4]
      - curr_state.vel_world: [3]
      - curr_state.desired_vel: float
      - goal_nav.delta_pos_body: [3]
    """
    def __init__(
        self,
        jsonl_path: str,
        img_size: int,
        frames: int = 4,
        state_dim: int = 10,
        use_state: bool = False,
        normalize: str = "imagenet",
        depth_mode: str = "log",
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
        self.depth_mode = depth_mode
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

        if len(quat) != 4 or len(vel_world) != 3 or len(delta) != 3:
            raise ValueError(f"Bad fields: quat({len(quat)}), vel_world({len(vel_world)}), delta({len(delta)})")

        quat4 = [float(x) for x in quat]
        quat4 = [0.0 if (not math.isfinite(x)) else x for x in quat4]

        qn = (quat4[0] * quat4[0] + quat4[1] * quat4[1] + quat4[2] * quat4[2] + quat4[3] * quat4[3]) ** 0.5
        if qn < self.eps:
            quat4 = [1.0, 0.0, 0.0, 0.0]
        else:
            inv = 1.0 / qn
            quat4 = [quat4[0] * inv, quat4[1] * inv, quat4[2] * inv, quat4[3] * inv]

        goal_dir = self._safe_unit3([float(x) for x in delta])
        vel_dir = self._safe_unit3([float(x) for x in vel_world])

        s = quat4 + goal_dir + vel_dir
        if len(s) != self.state_dim:
            raise ValueError(f"state_dim mismatch: expected {self.state_dim}, got {len(s)}")

        return torch.tensor(s, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        frame_paths: List[str] = it["frame_paths"]
        if len(frame_paths) != self.frames:
            raise ValueError(f"Expected {self.frames} frame_paths, got {len(frame_paths)}")

        imgs = [
            load_depth_png_as_3ch_tensor(
                p,
                self.img_size,
                normalize=self.normalize,
                depth_mode=self.depth_mode,
                eps=self.eps,
            )
            for p in frame_paths
        ]
        images = torch.stack(imgs, dim=0)  # [S,3,H,W]

        label = it.get("label_velcmd", None)
        if label is None:
            raise ValueError("Missing label_velcmd in jsonl item")
        if len(label) != 3:
            raise ValueError(f"label_velcmd must be len=3, got {len(label)}")

        den = self._get_desired_vel_den(it, label)
        y = torch.tensor([float(label[0]) / den, float(label[1]) / den, float(label[2]) / den], dtype=torch.float32)

        state = self._build_state(it)

        return {"images": images, "state": state, "label": y, "label_den": torch.tensor(den, dtype=torch.float32)}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["images"] for b in batch], dim=0)
    state = torch.stack([b["state"] for b in batch], dim=0)
    label = torch.stack([b["label"] for b in batch], dim=0)
    label_den = torch.stack([b["label_den"] for b in batch], dim=0)  # [B]
    return {"images": images, "state": state, "label": label, "label_den": label_den}


# =========================================================
# Train / Eval
# =========================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    val_log_every: int = 0,
    print_denorm_n: int = 0,          # NEW
) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")

    loss_sum = torch.tensor(0.0, device=device)
    denorm_sum = torch.tensor(0.0, device=device)
    n_sum = torch.tensor(0.0, device=device)

    printed = 0  # NEW

    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)          # normalized
        label_den = batch["label_den"].to(device, non_blocking=True)  # [B]

        pred = model(images, state)                                   # normalized
        loss = loss_fn(pred, label)

        den = label_den.view(-1, 1).clamp_min(1e-6)
        pred_den = pred * den
        label_denorm = label * den
        denorm_loss = loss_fn(pred_den, label_denorm)

        bs = images.size(0)
        loss_sum += loss.detach() * bs
        denorm_sum += denorm_loss.detach() * bs
        n_sum += torch.tensor(float(bs), device=device)

        # -------- NEW: print a few denorm samples (rank0 only) --------
        if is_rank0() and print_denorm_n > 0 and printed < print_denorm_n:
            # move small tensors to cpu for nice printing
            pd = pred_den.detach().float().cpu()
            ld = label_denorm.detach().float().cpu()
            den_cpu = label_den.detach().float().cpu()

            for i in range(bs):
                if printed >= print_denorm_n:
                    break
                p3 = pd[i].tolist()
                l3 = ld[i].tolist()
                dv = float(den_cpu[i].item())
                print(
                    f"[val_denorm] #{printed+1} desired_vel_den={dv:.3f} m/s | "
                    f"pred(m/s)=[{p3[0]:+.3f},{p3[1]:+.3f},{p3[2]:+.3f}] "
                    f"label(m/s)=[{l3[0]:+.3f},{l3[1]:+.3f},{l3[2]:+.3f}]"
                )
                printed += 1

        if val_log_every and (step % val_log_every == 0) and is_rank0():
            dt = time.time() - t0
            seen = float(n_sum.item())
            avg_norm = float((loss_sum / torch.clamp(n_sum, min=1.0)).item())
            avg_den = float((denorm_sum / torch.clamp(n_sum, min=1.0)).item())
            print(
                f"[val] step={step}/{len(loader)} "
                f"partial_mse_norm={avg_norm:.6f} partial_mse_denorm={avg_den:.6f} "
                f"seen={seen:.0f} time={dt/60.0:.1f}min"
            )

    loss_sum = ddp_all_reduce_sum(loss_sum)
    denorm_sum = ddp_all_reduce_sum(denorm_sum)
    n_sum = ddp_all_reduce_sum(n_sum)

    mse_norm = (loss_sum / torch.clamp(n_sum, min=1.0)).item()
    mse_denorm = (denorm_sum / torch.clamp(n_sum, min=1.0)).item()

    return {"mse": float(mse_norm), "loss": float(mse_norm), "denorm_mse": float(mse_denorm)}


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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    grad_clip: float,
    log_every: int,
    writers: Optional[List["SummaryWriter"]],
    global_step: int,
    save_every_steps: int,
    out_dir: str,
    epoch: int,
    cfg_dict: Dict[str, Any],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[Dict[str, float], int]:
    model.train()
    loss_fn = nn.MSELoss(reduction="mean")

    total_loss = 0.0
    total_denorm = 0.0
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        label_den = batch["label_den"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
            pred = model(images, state)
            loss = loss_fn(pred, label)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        with torch.no_grad():
            den = label_den.view(-1, 1).clamp_min(1e-6)
            pred_den = pred * den
            label_denorm = label * den
            denorm_loss = loss_fn(pred_den, label_denorm)
            total_denorm += float(denorm_loss.item()) * bs

        if is_rank0() and writers is not None:
            tb_add_scalar(writers, "train/loss", float(loss.item()), global_step)
            tb_add_scalar(writers, "train/mse", float(loss.item()), global_step)
            tb_add_scalar(writers, "train/denorm_mse", float(denorm_loss.item()), global_step)
            tb_add_scalar(writers, "train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

        if log_every > 0 and step % log_every == 0 and is_rank0():
            dt = time.time() - t0
            rate = n / max(1e-6, dt)
            avg = total_loss / max(1, n)
            avg_den = total_denorm / max(1, n)
            print(f"[train] step={step}/{len(loader)} mse_norm={avg:.6f} mse_denorm={avg_den:.6f} samples/s={rate:.1f}")
            if writers is not None:
                tb_add_scalar(writers, "train/samples_per_sec", float(rate), global_step)

        if save_every_steps and (global_step > 0) and (global_step % save_every_steps == 0) and is_rank0():
            ckpt_path = os.path.join(out_dir, f"step_{global_step}.pt")
            save_ckpt(ckpt_path, model, optimizer, epoch, cfg=cfg_dict, global_step=global_step, scheduler=scheduler)
            print(f"[ckpt] saved: {ckpt_path}")

        global_step += 1

        if scheduler is not None:
            scheduler.step(global_step)

    epoch_avg = total_loss / max(1, n)
    epoch_den = total_denorm / max(1, n)
    return {"mse": epoch_avg, "loss": epoch_avg, "denorm_mse": epoch_den}, global_step


def _count_params(m: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in m.parameters())


def print_vit_decoder_param_ratio(model: nn.Module) -> None:
    if not is_rank0():
        return

    m = unwrap_model(model)
    enc = getattr(m, "vggt", None)
    dec = getattr(m, "decoder", None)

    if enc is None or dec is None:
        print("[params] cannot find m.vggt or m.decoder, skip vit/decoder ratio.")
        return

    enc_total = _count_params(enc, trainable_only=False)
    dec_total = _count_params(dec, trainable_only=False)
    enc_train = _count_params(enc, trainable_only=True)
    dec_train = _count_params(dec, trainable_only=True)

    ratio_total = float(enc_total) / float(max(1, dec_total))
    ratio_train = float(enc_train) / float(max(1, dec_train))

    print(
        "[params][vit_vs_decoder] "
        f"enc_total={enc_total:,} dec_total={dec_total:,} ratio_total={ratio_total:.3f} | "
        f"enc_trainable={enc_train:,} dec_trainable={dec_train:,} ratio_trainable={ratio_train:.3f}"
    )


def print_trainable_params(model: nn.Module) -> None:
    m = unwrap_model(model)
    total = 0
    trainable = 0
    for p in m.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(f"[params] trainable={trainable:,} / total={total:,} ({100.0*trainable/max(1,total):.2f}%)")


def _enable_grad_ckpt(m: nn.Module, use_reentrant: bool) -> None:
    base = unwrap_model(m)

    if hasattr(base, "gradient_checkpointing_enable") and callable(getattr(base, "gradient_checkpointing_enable")):
        try:
            base.gradient_checkpointing_enable(True, use_reentrant=use_reentrant)  # type: ignore
            if is_rank0():
                print(f"[gc] enabled via model.gradient_checkpointing_enable(use_reentrant={use_reentrant})")
            return
        except TypeError:
            base.gradient_checkpointing_enable(True, use_reentrant)  # type: ignore
            if is_rank0():
                print(f"[gc] enabled via model.gradient_checkpointing_enable(positional, use_reentrant={use_reentrant})")
            return

    if hasattr(base, "vggt") and hasattr(base.vggt, "gradient_checkpointing_enable") and callable(getattr(base.vggt, "gradient_checkpointing_enable")):
        try:
            base.vggt.gradient_checkpointing_enable(True, use_reentrant=use_reentrant)  # type: ignore
            if is_rank0():
                print(f"[gc] enabled via model.vggt.gradient_checkpointing_enable(use_reentrant={use_reentrant})")
            return
        except TypeError:
            base.vggt.gradient_checkpointing_enable(True, use_reentrant)  # type: ignore
            if is_rank0():
                print(f"[gc] enabled via model.vggt.gradient_checkpointing_enable(positional, use_reentrant={use_reentrant})")
            return

    if is_rank0():
        print("[warn] --grad_ckpt set but model has no gradient_checkpointing_enable() on model or model.vggt. Skip.")


def _extract_state_dict_from_ckpt(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Try best-effort to get a plain state_dict mapping from torch.load output.
    """
    if isinstance(obj, dict):
        for k in ["model", "state_dict", "teacher", "student"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    if isinstance(obj, dict):
        # maybe already a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Cannot extract state_dict from ckpt. Expected dict with keys like 'model'/'state_dict' or a raw state_dict.")


def _maybe_init_dinov2(model: nn.Module, dinov2_ckpt: str, device: str) -> None:
    """
    Call model.vggt.init_from_dinov2 with local ckpt if available.

    Compatible with two possible encoder implementations:
      - init_from_dinov2(local_ckpt_path=..., model_name=..., verbose=...)
      - init_from_dinov2(state_dict=..., model_name=..., verbose=...)
    """
    if not dinov2_ckpt:
        return
    if not os.path.isfile(dinov2_ckpt):
        raise FileNotFoundError(f"--dinov2_ckpt not found: {dinov2_ckpt}")

    base = unwrap_model(model)
    enc = getattr(base, "vggt", None)
    if enc is None or not hasattr(enc, "init_from_dinov2"):
        raise RuntimeError("Model has no vggt.init_from_dinov2(). Please update vision_encoder.py accordingly.")

    if is_rank0():
        print(f"[DINOv2] init from local ckpt: {dinov2_ckpt}")

    # Preferred: pass local_ckpt_path if encoder supports it
    try:
        enc.init_from_dinov2(model_name="dinov2_vitl14", local_ckpt_path=dinov2_ckpt, verbose=True)  # type: ignore
        return
    except TypeError:
        pass

    # Fallback: load state_dict and pass in
    ck = torch.load(dinov2_ckpt, map_location="cpu")
    sd = _extract_state_dict_from_ckpt(ck)
    try:
        enc.init_from_dinov2(model_name="dinov2_vitl14", state_dict=sd, verbose=True)  # type: ignore
        return
    except Exception as e:
        raise RuntimeError(f"Failed to init_from_dinov2 via state_dict fallback: {e}") from e


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # ----- data -----
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--frames", type=int, default=4)

    # ✅ default align to DINO input pipeline
    ap.add_argument("--normalize", type=str, default="imagenet", choices=["unit", "imagenet", "none"])

    # ✅ NEW: depth -> 3ch mode
    ap.add_argument("--depth_mode", type=str, default="log", choices=["repeat", "log", "standardize"])

    ap.add_argument("--filter_max_dt", type=float, default=-1.0, help="if >0, drop samples with match_dt_sec > this")

    # state interface
    ap.add_argument("--use_state", action="store_true")
    ap.add_argument("--state_dim", type=int, default=10)

    # normalization knobs
    ap.add_argument("--desired_vel_min", type=float, default=0.1, help="clamp desired_vel to at least this when normalizing label")
    ap.add_argument("--eps", type=float, default=1e-6)

    # ----- model -----
    # ✅ default align to DINOv2 vitl14
    ap.add_argument("--vggt_embed_dim", type=int, default=1024)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_patches", type=int, default=512)

    # ✅ NEW: local dinov2 init path
    ap.add_argument("--dinov2_ckpt", type=str, default="", help="local dinov2 vitl14 checkpoint to init spatial encoder")

    ap.add_argument("--freeze_vggt", action="store_true", help="(kept for compatibility) will be ignored for ViT: train whole model")

    # ----- train -----
    ap.add_argument("--epochs", type=int, default=10, help="train epochs (ignored if --max_steps > 0)")
    ap.add_argument("--max_steps", type=int, default=-1, help="if >0, train by total steps (overrides epochs)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--val_log_every", type=int, default=0, help="print eval progress every N val steps (rank0 only)")
    ap.add_argument("--save_every_steps", type=int, default=0, help="save ckpt every N global steps (rank0 only)")

    # ----- LR schedule -----
    ap.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"],
                    help="lr schedule type. constant keeps original behavior; cosine enables warmup+cosine decay.")
    ap.add_argument("--warmup_steps", type=int, default=0, help="linear warmup steps (only for --lr_schedule cosine)")
    ap.add_argument("--min_lr_ratio", type=float, default=0.0,
                    help="final lr ratio at the end of cosine decay, e.g. 0.1 means end lr = 0.1*base_lr")

    # ----- activation checkpointing -----
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="enable activation checkpointing in ViT encoder blocks (save memory, slower)")
    ap.add_argument("--grad_ckpt_use_reentrant", action="store_true",
                    help="use reentrant checkpointing (legacy). default False recommended.")

    # ----- misc -----
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--save_every", type=int, default=1)  # epoch-based
    ap.add_argument("--resume", type=str, default="", help="explicit ckpt path; if empty, may auto-resume from out_dir")
    ap.add_argument("--tensorboard", action="store_true")
    
    ap.add_argument("--val_print_denorm_n", type=int, default=0,
                help="(rank0) during val, print first N samples' denorm pred/label in m/s")
    ap.add_argument("--val_print_every", type=int, default=1,
                    help="(rank0) print denorm samples every N epochs (1=every epoch)")


    args = ap.parse_args()
    set_seed(args.seed)

    ddp_setup()

    # device
    if ddp_is_enabled():
        device = f"cuda:{ddp_local_rank()}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_rank0():
        print("device:", device, "ddp:", ddp_is_enabled(), "world_size:", ddp_world_size())

    os.makedirs(args.out_dir, exist_ok=True)

    # -------- auto resume --------
    resume_path = (args.resume or "").strip()
    if not resume_path:
        auto_ckpt = find_auto_resume_ckpt(args.out_dir)
        if auto_ckpt:
            resume_path = auto_ckpt
            if is_rank0():
                print(f"[auto-resume] found ckpt in out_dir, will resume from: {resume_path}")

    filter_max_dt = None if args.filter_max_dt <= 0 else args.filter_max_dt

    # datasets
    train_ds = Depth4FrameVelCmdDataset(
        jsonl_path=args.train_jsonl,
        img_size=args.img_size,
        frames=args.frames,
        state_dim=args.state_dim,
        use_state=args.use_state,
        normalize=args.normalize,
        depth_mode=args.depth_mode,
        filter_max_dt=filter_max_dt,
        eps=args.eps,
        desired_vel_min=args.desired_vel_min,
    )
    val_ds = Depth4FrameVelCmdDataset(
        jsonl_path=args.val_jsonl,
        img_size=args.img_size,
        frames=args.frames,
        state_dim=args.state_dim,
        use_state=args.use_state,
        normalize=args.normalize,
        depth_mode=args.depth_mode,
        filter_max_dt=filter_max_dt,
        eps=args.eps,
        desired_vel_min=args.desired_vel_min,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if ddp_is_enabled() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if ddp_is_enabled() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # model cfg
    if args.freeze_vggt and is_rank0():
        print("[warn] --freeze_vggt is ignored in this ViT training script: training the whole model (encoder+decoder).")

    # ✅ default: DINO aligned encoder, but decoder stays compact (stable)
    cfg = VelocityPlannerConfig(
        state_dim=args.state_dim,

        # ===== decoder params (keep small & stable) =====
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_ff=1024,
        dropout=0.1,

        img_size=args.img_size,
        patch_size=args.patch_size,

        # ===== encoder params (align DINOv2 vitl14) =====
        vggt_embed_dim=args.vggt_embed_dim,   # default 1024
        vit_num_frames=args.frames,
        vit_nhead=16,
        vit_mlp_ratio=4.0,
        vit_dropout=0.0,
        vit_spatial_layers=6,
        vit_temporal_layers=2,
        vit_use_cls_token=True,
        vit_use_time_embed=True,

        max_patches=(args.max_patches if args.max_patches > 0 else None),
        out_dim=3,

        freeze_vggt=False,
    )

    model = VelocityPlannerModel(cfg).to(device)

    # -------- NEW: enable activation checkpointing (BEFORE DDP wrap) --------
    if args.grad_ckpt:
        _enable_grad_ckpt(model, use_reentrant=bool(args.grad_ckpt_use_reentrant))

    # -------- NEW: init dinov2 from local ckpt (BEFORE resume/DDP wrap) --------
    if args.dinov2_ckpt:
        _maybe_init_dinov2(model, args.dinov2_ckpt, device=device)

    # resume BEFORE DDP wrap
    start_epoch = 1
    global_step = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        if is_rank0():
            print(f"[resume] from {resume_path}, start_epoch={start_epoch}, global_step={global_step}")

    if ddp_is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            if is_rank0():
                print("[warn] resume ckpt has no optimizer state, optimizer will start fresh.")

    cfg_dict = asdict(cfg)

    if is_rank0():
        with open(os.path.join(args.out_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_dir, "model_cfg.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

        print(f"[cfg] normalize={args.normalize} depth_mode={args.depth_mode}")
        print_trainable_params(model)
        print_vit_decoder_param_ratio(model)

        if not args.use_state:
            print("[warn] --use_state is OFF: state will be zeros. You probably want --use_state.")
        if args.state_dim != 10:
            print(f"[warn] expected state_dim=10 for quat+goal_dir+vel_dir, but got {args.state_dim}")

    # -------- TensorBoard (dual writers) --------
    tb_writers: Optional[List["SummaryWriter"]] = None
    if args.tensorboard and is_rank0():
        if not _HAS_TB:
            print("[warn] tensorboard not available. pip install tensorboard")
        else:
            local_tb = os.path.join(args.out_dir, "tb")
            os.makedirs(local_tb, exist_ok=True)

            job_tb = _infer_job_logdir("/workspace/zhuy25@xiaopeng.com")
            os.makedirs(job_tb, exist_ok=True)

            tb_writers = [
                SummaryWriter(log_dir=local_tb),
                SummaryWriter(log_dir=job_tb),
            ]
            print(f"[tb] local  : {local_tb}")
            print(f"[tb] joblvl : {job_tb}")

    # -------- Scheduler (optional) --------
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    target_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None
    if target_steps is None:
        total_steps = int(args.epochs) * int(len(train_loader))
    else:
        total_steps = int(target_steps)

    if args.lr_schedule == "cosine":
        lr_lambda = build_warmup_cosine_lambda(
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        if global_step > 0:
            scheduler.step(global_step)

        if is_rank0():
            print(f"[lr] schedule=cosine total_steps={total_steps} warmup_steps={args.warmup_steps} min_lr_ratio={args.min_lr_ratio}")
            print(f"[lr] base_lr={optimizer.param_groups[0]['lr']:.6g} (note: resume may override base lr via optimizer state)")

        if resume_path:
            ckpt = torch.load(resume_path, map_location="cpu")
            if "scheduler" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                    if is_rank0():
                        print("[resume] scheduler state restored.")
                except Exception as e:
                    if is_rank0():
                        print(f"[warn] failed to load scheduler state from ckpt: {e}. Will fallback to global_step alignment.")
    else:
        if is_rank0():
            print("[lr] schedule=constant (original behavior)")

    best_val = float("inf")

    if target_steps is not None and is_rank0():
        print(f"[mode] training by steps: max_steps={target_steps}")

    epoch = start_epoch
    while True:
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
            print(f"[train] epoch={epoch} mse_norm={tr['mse']:.6f} denorm_mse={tr['denorm_mse']:.6f}")
            if tb_writers is not None:
                tb_add_scalar(tb_writers, "train/epoch_mse_norm", tr["mse"], epoch)
                tb_add_scalar(tb_writers, "train/epoch_denorm_mse", tr["denorm_mse"], epoch)

        # va = evaluate(model, val_loader, device=device, val_log_every=args.val_log_every)
        print_n = 0
        if args.val_print_denorm_n > 0 and (epoch % max(1, args.val_print_every) == 0):
            print_n = args.val_print_denorm_n

        va = evaluate(
            model,
            val_loader,
            device=device,
            val_log_every=args.val_log_every,
            print_denorm_n=print_n,
        )

        if is_rank0():
            print(f"[val]   epoch={epoch} mse_norm={va['mse']:.6f} denorm_mse={va['denorm_mse']:.6f}")
            if tb_writers is not None:
                tb_add_scalar(tb_writers, "val/mse_norm", va["mse"], epoch)
                tb_add_scalar(tb_writers, "val/denorm_mse", va["denorm_mse"], epoch)

            if va["mse"] < best_val:
                best_val = va["mse"]
                save_ckpt(
                    os.path.join(args.out_dir, "best.pt"),
                    model,
                    optimizer,
                    epoch,
                    cfg=cfg_dict,
                    global_step=global_step,
                    scheduler=scheduler,
                )

            if args.save_every > 0 and epoch % args.save_every == 0:
                save_ckpt(
                    os.path.join(args.out_dir, f"epoch_{epoch}.pt"),
                    model,
                    optimizer,
                    epoch,
                    cfg=cfg_dict,
                    global_step=global_step,
                    scheduler=scheduler,
                )

        if target_steps is not None:
            if global_step >= target_steps:
                if is_rank0():
                    print(f"[stop] reached max_steps={target_steps}, global_step={global_step}")
                break
        else:
            if epoch >= args.epochs:
                break

        epoch += 1

    tb_flush(tb_writers)
    tb_close(tb_writers)

    if is_rank0():
        print("Training finished.")
        print("best_val_mse_norm:", best_val)

    ddp_cleanup()


if __name__ == "__main__":
    main()
