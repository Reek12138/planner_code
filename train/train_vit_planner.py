#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train VelocityPlannerModel with metadata.jsonl (4-frame depth -> 3-axis velcmd)

✅ 支持：
- 单卡 / 多卡多 rank（torchrun DDP）
- TensorBoard 记录（train/val loss、lr、samples/s）
- eval 指标在 DDP 下做全局聚合（all-reduce）
- 可选冻结 VGGT encoder（--freeze_vggt）   <-- 接口保留，但本脚本在 ViT 训练模式下会忽略冻结，整网训练

✅ 数据适配（按你最终确认的规则，保持训练逻辑不变）：
- decoder Q(state) = quat_wxyz(4) + goal_dir_body(3) + vel_dir_world(3) => state_dim=10
  - quat_wxyz：不额外归一化（直接使用日志四元数）
  - goal_dir_body = delta_pos_body / (||delta_pos_body|| + eps)   # 把尺度变成方向
  - vel_dir_world = vel_world / (||vel_world|| + eps)             # 当前速度方向（world 三轴）

- label 归一化：
  y_norm = label_velcmd / clamp(desired_vel, desired_vel_min)
  desired_vel 从 curr_state.desired_vel 读取；缺失则回退到 ||label_velcmd|| 或 1.0

✅ 自动断点续训（保持原逻辑）：
- 如果 --resume 未指定：
  - 若 out_dir 下存在任何 *.pt 文件：
    - 优先从 out_dir/best.pt 恢复（存在才恢复）
    - 否则从 out_dir 下“最新修改时间”的 *.pt 恢复

✅ 验证时新增 print（rank0 only）：
- 打印 partial normalized mse
- 同时打印 partial denorm mse（把 y_norm * desired_vel_den 还原到 m/s 口径做 mse，仅用于观测）

✅ 本次修改（仅改 TensorBoard 写入逻辑，训练逻辑不变）：
- rank0 同时写两份 TB logs：
  1) local: 仍然写到 args.out_dir/tb
  2) job-level: 按 job 级别目录推导，写到 /workspace/zhuy25@xiaopeng.com/<job_name>/xflow_logs
- 所有 add_scalar 同步写入两份 writer

✅ 新增：可选 warmup + cosine decay（默认不启用，保持原来 constant LR）
- 通过 --lr_schedule cosine 启用
- 支持 --warmup_steps / --min_lr_ratio
- scheduler 状态会保存到 ckpt 并在 resume 时恢复（老 ckpt 没有 scheduler 也能兼容）

✅ 新增：Activation Checkpointing（可选，默认不启用，训练逻辑不变）
- 通过 --grad_ckpt 启用
- 默认 use_reentrant=False（推荐）
- 会在 DDP wrap 前调用 model.gradient_checkpointing_enable() 或 model.vggt.gradient_checkpointing_enable()
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
from planner.models.modeling.vit_planner import VelocityPlannerConfig, VelocityPlannerModel  # noqa: E402


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
import torch.nn.functional as F

def cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred/target: [B,3]
    返回标量 mean loss: 1 - cosine_sim
    """
    pred_u = F.normalize(pred, p=2, dim=-1, eps=eps)
    tgt_u  = F.normalize(target, p=2, dim=-1, eps=eps)
    cos = (pred_u * tgt_u).sum(dim=-1)          # [B]
    return (1.0 - cos).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_den: torch.Tensor,
    cos_weight: float = 1.0,
    mse_weight: float = 0.1,
    l1_weight: float = 0.05,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss for velocity prediction.
    
    Args:
        pred: [B, 3] normalized prediction (unit vector)
        target: [B, 3] normalized target (unit vector)
        label_den: [B] desired velocity for denormalization
        cos_weight: weight for cosine loss
        mse_weight: weight for MSE loss (denorm)
        l1_weight: weight for L1 loss (denorm)
        eps: epsilon for numerical stability
    
    Returns:
        total_loss: combined loss for backward
        cos_loss: cosine loss (for logging)
        mse_loss: MSE loss on denormed values (for logging)
        l1_loss: L1 loss on denormed values (for logging)
    """
    # 1. Cosine Loss (方向)
    pred_u = F.normalize(pred, p=2, dim=-1, eps=eps)
    tgt_u = F.normalize(target, p=2, dim=-1, eps=eps)
    cos_sim = (pred_u * tgt_u).sum(dim=-1)
    cos_loss = (1.0 - cos_sim).mean()
    
    # Denormalize to m/s scale for magnitude losses
    den = label_den.view(-1, 1).clamp_min(eps)
    pred_den = pred * den
    target_den = target * den
    
    # 2. MSE Loss (幅度)
    mse_loss = F.mse_loss(pred_den, target_den)
    
    # 3. L1 Loss (更鲁棒)
    l1_loss = F.l1_loss(pred_den, target_den)
    
    # Combined
    total_loss = cos_weight * cos_loss + mse_weight * mse_loss + l1_weight * l1_loss
    
    return total_loss, cos_loss, mse_loss, l1_loss


def combined_loss_with_hard_mining(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_den: torch.Tensor,
    cos_weight: float = 1.0,
    mse_weight: float = 0.1,
    l1_weight: float = 0.05,
    eps: float = 1e-6,
    hard_mining_mode: str = "none",  # "none", "weighted", "topk"
    hard_mining_ratio: float = 0.7,   # for weighted: threshold quantile; for topk: keep ratio
    hard_mining_boost: float = 2.0,   # weight boost for hard examples
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss with hard example mining.
    
    Args:
        hard_mining_mode: 
            - "none": no hard mining, same as combined_loss
            - "weighted": increase weight for hard examples (loss > quantile)
            - "topk": only keep top-k hardest examples
        hard_mining_ratio: 
            - for "weighted": examples with loss > this quantile get boosted
            - for "topk": keep this ratio of hardest examples
        hard_mining_boost: weight multiplier for hard examples (only for "weighted" mode)
    
    Returns:
        total_loss, cos_loss, mse_loss, l1_loss, hard_ratio (for logging)
    """
    B = pred.shape[0]
    
    # 1. Cosine Loss per sample [B]
    pred_u = F.normalize(pred, p=2, dim=-1, eps=eps)
    tgt_u = F.normalize(target, p=2, dim=-1, eps=eps)
    cos_sim = (pred_u * tgt_u).sum(dim=-1)
    cos_loss_per_sample = 1.0 - cos_sim  # [B]
    
    # Denormalize
    den = label_den.view(-1, 1).clamp_min(eps)
    pred_den = pred * den
    target_den = target * den
    
    # 2. MSE Loss per sample [B]
    mse_loss_per_sample = F.mse_loss(pred_den, target_den, reduction='none').mean(dim=-1)  # [B]
    
    # 3. L1 Loss per sample [B]
    l1_loss_per_sample = F.l1_loss(pred_den, target_den, reduction='none').mean(dim=-1)  # [B]
    
    # Combined per-sample loss
    total_loss_per_sample = (
        cos_weight * cos_loss_per_sample + 
        mse_weight * mse_loss_per_sample + 
        l1_weight * l1_loss_per_sample
    )  # [B]
    
    # Hard mining
    hard_ratio = torch.tensor(0.0, device=pred.device)
    
    if hard_mining_mode == "none" or B <= 1:
        # No hard mining
        total_loss = total_loss_per_sample.mean()
        
    elif hard_mining_mode == "weighted":
        # Weighted hard mining: boost loss for hard examples
        with torch.no_grad():
            # Compute threshold
            threshold = total_loss_per_sample.quantile(hard_mining_ratio)
            # Create weights: hard examples get boost, easy examples get 1.0
            weights = torch.where(
                total_loss_per_sample > threshold,
                torch.tensor(hard_mining_boost, device=pred.device),
                torch.tensor(1.0, device=pred.device)
            )
            # Normalize weights so that average weight is 1.0
            weights = weights / weights.mean()
            hard_ratio = (total_loss_per_sample > threshold).float().mean()
        
        total_loss = (total_loss_per_sample * weights).mean()
        
    elif hard_mining_mode == "topk":
        # TopK hard mining: only keep k hardest examples
        k = max(1, int(B * hard_mining_ratio))
        with torch.no_grad():
            # Get indices of top-k hardest examples
            _, topk_indices = torch.topk(total_loss_per_sample, k, largest=True, sorted=False)
            hard_ratio = torch.tensor(float(k) / B, device=pred.device)
        
        # Only compute loss on hard examples
        total_loss = total_loss_per_sample[topk_indices].mean()
        
        # Also compute cos/mse/l1 only on hard examples for logging
        cos_loss_per_sample = cos_loss_per_sample[topk_indices]
        mse_loss_per_sample = mse_loss_per_sample[topk_indices]
        l1_loss_per_sample = l1_loss_per_sample[topk_indices]
        
    else:
        raise ValueError(f"Unknown hard_mining_mode: {hard_mining_mode}")
    
    # For logging: mean of all samples (not just hard ones)
    cos_loss = cos_loss_per_sample.mean()
    mse_loss = mse_loss_per_sample.mean()
    l1_loss = l1_loss_per_sample.mean()
    
    return total_loss, cos_loss, mse_loss, l1_loss, hard_ratio


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
    normalize: str = "unit",
) -> torch.Tensor:
    im = Image.open(path)
    import numpy as np
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
    elif normalize in ("unit", "none"):
        pass
    else:
        raise ValueError(f"Unknown normalize={normalize}")

    return x.float()


def find_auto_resume_ckpt(out_dir: str) -> str:
    """
    Auto resume policy:
      - If out_dir has any *.pt:
        - Prefer out_dir/best.pt if exists
        - Else pick most recently modified *.pt
      - Else return ""
    """
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

    state = quat_wxyz(4) + goal_dir_body(3) + vel_dir_world(3) => 10
      goal_dir_body = delta_pos_body / (||delta|| + eps)
      vel_dir_world = vel_world / (||vel|| + eps)

    label normalized:
      y_norm = label_velcmd / clamp(desired_vel, desired_vel_min)
    """

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

        if len(quat) != 4 or len(vel_world) != 3 or len(delta) != 3:
            raise ValueError(f"Bad fields: quat({len(quat)}), vel_world({len(vel_world)}), delta({len(delta)})")

        quat4 = [float(x) for x in quat]
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

        imgs = [load_depth_png_as_3ch_tensor(p, self.img_size, normalize=self.normalize) for p in frame_paths]
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
    cos_weight: float = 1.0,
    mse_weight: float = 0.1,
    l1_weight: float = 0.05,
) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")

    total_sum = torch.tensor(0.0, device=device)
    cos_sum = torch.tensor(0.0, device=device)
    mse_sum = torch.tensor(0.0, device=device)
    l1_sum = torch.tensor(0.0, device=device)
    denorm_sum = torch.tensor(0.0, device=device)
    n_sum = torch.tensor(0.0, device=device)
    
    printed = 0  # NEW

    t0 = time.time()
    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device, non_blocking=True)
        state = batch["state"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        label_den = batch["label_den"].to(device, non_blocking=True)

        pred = model(images, state)
        # Use combined_loss for consistency with training
        loss, cos_loss, mse_loss, l1_loss = combined_loss(
            pred, label, label_den,
            cos_weight=cos_weight, mse_weight=mse_weight, l1_weight=l1_weight, eps=1e-6
        )

        den = label_den.view(-1, 1).clamp_min(1e-6)
        pred_den = pred * den
        label_denorm = label * den
        denorm_loss = loss_fn(pred_den, label_denorm)

        bs = images.size(0)
        total_sum += loss.detach() * bs
        cos_sum += cos_loss.detach() * bs
        mse_sum += mse_loss.detach() * bs
        l1_sum += l1_loss.detach() * bs
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
            avg_total = float((total_sum / torch.clamp(n_sum, min=1.0)).item())
            avg_den = float((denorm_sum / torch.clamp(n_sum, min=1.0)).item())
            print(
                f"[val] step={step}/{len(loader)} "
                f"partial_total={avg_total:.6f} partial_denorm={avg_den:.6f} "
                f"seen={seen:.0f} time={dt/60.0:.1f}min"
            )

    total_sum = ddp_all_reduce_sum(total_sum)
    cos_sum = ddp_all_reduce_sum(cos_sum)
    mse_sum = ddp_all_reduce_sum(mse_sum)
    l1_sum = ddp_all_reduce_sum(l1_sum)
    denorm_sum = ddp_all_reduce_sum(denorm_sum)
    n_sum = ddp_all_reduce_sum(n_sum)

    total_loss = (total_sum / torch.clamp(n_sum, min=1.0)).item()
    cos_loss = (cos_sum / torch.clamp(n_sum, min=1.0)).item()
    mse_loss = (mse_sum / torch.clamp(n_sum, min=1.0)).item()
    l1_loss = (l1_sum / torch.clamp(n_sum, min=1.0)).item()
    mse_denorm = (denorm_sum / torch.clamp(n_sum, min=1.0)).item()

    return {
        "loss": float(total_loss),
        "cos_loss": float(cos_loss),
        "mse_loss": float(mse_loss),
        "l1_loss": float(l1_loss),
        "denorm_mse": float(mse_denorm),
    }


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
    cos_weight: float = 1.0,
    mse_weight: float = 0.1,
    l1_weight: float = 0.05,
    hard_mining_mode: str = "none",
    hard_mining_ratio: float = 0.7,
    hard_mining_boost: float = 2.0,
) -> Tuple[Dict[str, float], int]:
    model.train()
    loss_fn = nn.MSELoss(reduction="mean")

    total_loss = 0.0
    total_denorm = 0.0
    total_cos = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_hard_ratio = 0.0
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
            # Combined loss with optional hard mining
            loss, cos_loss, mse_loss, l1_loss, hard_ratio = combined_loss_with_hard_mining(
                pred, label, label_den,
                cos_weight=cos_weight, mse_weight=mse_weight, l1_weight=l1_weight, eps=1e-6,
                hard_mining_mode=hard_mining_mode,
                hard_mining_ratio=hard_mining_ratio,
                hard_mining_boost=hard_mining_boost,
            )

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_cos += float(cos_loss.item()) * bs
        total_mse += float(mse_loss.item()) * bs
        total_l1 += float(l1_loss.item()) * bs
        if hard_mining_mode != "none":
            total_hard_ratio += float(hard_ratio.item()) * bs
        n += bs

        with torch.no_grad():
            den = label_den.view(-1, 1).clamp_min(1e-6)
            pred_den = pred * den
            label_denorm = label * den
            denorm_loss = loss_fn(pred_den, label_denorm)
            total_denorm += float(denorm_loss.item()) * bs

        if is_rank0() and writers is not None:
            tb_add_scalar(writers, "train/loss", float(loss.item()), global_step)
            tb_add_scalar(writers, "train/cos_loss", float(cos_loss.item()), global_step)
            tb_add_scalar(writers, "train/mse_loss", float(mse_loss.item()), global_step)
            tb_add_scalar(writers, "train/l1_loss", float(l1_loss.item()), global_step)
            tb_add_scalar(writers, "train/denorm_mse", float(denorm_loss.item()), global_step)
            tb_add_scalar(writers, "train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            if hard_mining_mode != "none":
                tb_add_scalar(writers, "train/hard_ratio", float(hard_ratio.item()), global_step)

        if log_every > 0 and step % log_every == 0 and is_rank0():
            dt = time.time() - t0
            rate = n / max(1e-6, dt)
            avg = total_loss / max(1, n)
            avg_den = total_denorm / max(1, n)
            avg_cos = total_cos / max(1, n)
            avg_mse = total_mse / max(1, n)
            avg_l1 = total_l1 / max(1, n)
            print(f"[train] step={step}/{len(loader)} total={avg:.6f} cos={avg_cos:.6f} mse={avg_mse:.6f} l1={avg_l1:.6f} denorm={avg_den:.6f} samples/s={rate:.1f}")
            if writers is not None:
                tb_add_scalar(writers, "train/samples_per_sec", float(rate), global_step)

        if save_every_steps and (global_step > 0) and (global_step % save_every_steps == 0) and is_rank0():
            ckpt_path = os.path.join(out_dir, f"step_{global_step}.pt")
            save_ckpt(ckpt_path, model, optimizer, epoch, cfg=cfg_dict, global_step=global_step, scheduler=scheduler)
            print(f"[ckpt] saved: {ckpt_path}")

        global_step += 1

        if scheduler is not None:
            # keep original behavior: step by global_step
            scheduler.step(global_step)

    epoch_avg = total_loss / max(1, n)
    epoch_den = total_denorm / max(1, n)
    epoch_cos = total_cos / max(1, n)
    epoch_mse = total_mse / max(1, n)
    epoch_l1 = total_l1 / max(1, n)
    result = {
        "loss": epoch_avg,
        "denorm_mse": epoch_den,
        "cos_loss": epoch_cos,
        "mse_loss": epoch_mse,
        "l1_loss": epoch_l1,
    }
    if hard_mining_mode != "none":
        result["hard_ratio"] = total_hard_ratio / max(1, n)
    return result, global_step

def _count_params(m: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in m.parameters())


def print_vit_decoder_param_ratio(model: nn.Module) -> None:
    """
    打印 encoder(vit/vggt) vs decoder 的参数量及比例（rank0 only）。
    兼容 DDP（会 unwrap）。
    """
    if not is_rank0():
        return

    m = unwrap_model(model)

    # 兼容你模型字段命名：encoder 是 vggt，decoder 是 decoder
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
    """
    Enable activation checkpointing without assuming exact API layout.
    Priority:
      1) model.gradient_checkpointing_enable(...)
      2) model.vggt.gradient_checkpointing_enable(...)
    """
    base = unwrap_model(m)

    if hasattr(base, "gradient_checkpointing_enable") and callable(getattr(base, "gradient_checkpointing_enable")):
        try:
            base.gradient_checkpointing_enable(True, use_reentrant=use_reentrant)  # type: ignore
            if is_rank0():
                print(f"[gc] enabled via model.gradient_checkpointing_enable(use_reentrant={use_reentrant})")
            return
        except TypeError:
            # some implementations only accept (enabled, use_reentrant) positional
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
    ap.add_argument("--normalize", type=str, default="unit", choices=["unit", "imagenet", "none"])
    ap.add_argument("--filter_max_dt", type=float, default=-1.0, help="if >0, drop samples with match_dt_sec > this")

    # state interface
    ap.add_argument("--use_state", action="store_true")
    ap.add_argument("--state_dim", type=int, default=10)

    # normalization knobs
    ap.add_argument("--desired_vel_min", type=float, default=0.1, help="clamp desired_vel to at least this when normalizing label")
    ap.add_argument("--eps", type=float, default=1e-6)

    # ----- model -----
    ap.add_argument("--vggt_embed_dim", type=int, default=2048)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--pretrained_agg_path", type=str, default="")
    ap.add_argument("--max_patches", type=int, default=512)

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

    # ----- NEW: activation checkpointing -----
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="enable activation checkpointing in ViT encoder blocks (save memory, slower)")
    ap.add_argument("--grad_ckpt_use_reentrant", action="store_true",
                    help="use reentrant checkpointing (legacy). default False recommended.")

    # ----- loss weights -----
    ap.add_argument("--cos_weight", type=float, default=1.0,
                    help="weight for cosine loss (direction)")
    ap.add_argument("--mse_weight", type=float, default=0.1,
                    help="weight for MSE loss on denormalized velocity magnitude")
    ap.add_argument("--l1_weight", type=float, default=0.05,
                    help="weight for L1 loss on denormalized velocity magnitude")

    # ----- hard example mining -----
    ap.add_argument("--hard_mining_mode", type=str, default="none", 
                    choices=["none", "weighted", "topk"],
                    help="hard example mining mode: none=disabled, weighted=boost hard examples, topk=keep only hard examples")
    ap.add_argument("--hard_mining_ratio", type=float, default=0.7,
                    help="for weighted: quantile threshold for hard examples; for topk: ratio of examples to keep")
    ap.add_argument("--hard_mining_boost", type=float, default=2.0,
                    help="weight boost for hard examples (only for weighted mode)")

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

    cfg = VelocityPlannerConfig(
        state_dim=args.state_dim,

        # ===== decoder params =====
        d_model=2048,
        nhead=16,
        num_layers=8,
        dim_ff=2048,
        dropout=0.1,

        img_size=args.img_size,
        patch_size=args.patch_size,

        vggt_embed_dim=args.vggt_embed_dim,
        vit_num_frames=args.frames,
        vit_nhead=16,
        vit_mlp_ratio=4.0,
        vit_dropout=0.0,
        vit_spatial_layers=2,
        vit_temporal_layers=1,
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

        print(f"[cfg] freeze_vggt(forced)={False} (args.freeze_vggt={args.freeze_vggt})")
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

        print(f"[loss] cos_weight={args.cos_weight} mse_weight={args.mse_weight} l1_weight={args.l1_weight}")
        if args.hard_mining_mode != "none":
            print(f"[hard_mining] mode={args.hard_mining_mode} ratio={args.hard_mining_ratio} boost={args.hard_mining_boost}")

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
            cos_weight=args.cos_weight,
            mse_weight=args.mse_weight,
            l1_weight=args.l1_weight,
            hard_mining_mode=args.hard_mining_mode,
            hard_mining_ratio=args.hard_mining_ratio,
            hard_mining_boost=args.hard_mining_boost,
        )

        if is_rank0():
            hard_str = f" hard={tr.get('hard_ratio', 0):.2f}" if args.hard_mining_mode != "none" else ""
            print(f"[train] epoch={epoch} total={tr['loss']:.6f} cos={tr['cos_loss']:.6f} mse={tr['mse_loss']:.6f} l1={tr['l1_loss']:.6f} denorm={tr['denorm_mse']:.6f}{hard_str}")
            if tb_writers is not None:
                tb_add_scalar(tb_writers, "train/epoch_loss", tr["loss"], epoch)
                tb_add_scalar(tb_writers, "train/epoch_cos_loss", tr["cos_loss"], epoch)
                tb_add_scalar(tb_writers, "train/epoch_mse_loss", tr["mse_loss"], epoch)
                tb_add_scalar(tb_writers, "train/epoch_l1_loss", tr["l1_loss"], epoch)
                tb_add_scalar(tb_writers, "train/epoch_denorm_mse", tr["denorm_mse"], epoch)
                if args.hard_mining_mode != "none" and "hard_ratio" in tr:
                    tb_add_scalar(tb_writers, "train/epoch_hard_ratio", tr["hard_ratio"], epoch)

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
            cos_weight=args.cos_weight,
            mse_weight=args.mse_weight,
            l1_weight=args.l1_weight,
        )
        if is_rank0():
            print(f"[val]   epoch={epoch} total={va['loss']:.6f} cos={va['cos_loss']:.6f} mse={va['mse_loss']:.6f} l1={va['l1_loss']:.6f} denorm={va['denorm_mse']:.6f}")
            if tb_writers is not None:
                tb_add_scalar(tb_writers, "val/loss", va["loss"], epoch)
                tb_add_scalar(tb_writers, "val/cos_loss", va["cos_loss"], epoch)
                tb_add_scalar(tb_writers, "val/mse_loss", va["mse_loss"], epoch)
                tb_add_scalar(tb_writers, "val/l1_loss", va["l1_loss"], epoch)
                tb_add_scalar(tb_writers, "val/denorm_mse", va["denorm_mse"], epoch)

            if va["loss"] < best_val:
                best_val = va["loss"]
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
