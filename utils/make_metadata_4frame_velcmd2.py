#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate metadata.json for 4-frame depth input -> 3-axis velocity command label.

✅ 在原逻辑基础上新增：
- 记录“当前时刻（窗口最后一帧）”的状态：vel_world / pos_world / quat_wxyz / desired_vel(若有)
- 以“未来 N 帧（默认 10）之后”的位置作为 goal_nav：
  - future_pos_world
  - delta_pos_world = future_pos_world - curr_pos_world
  - delta_pos_body  = R_bw(curr_quat) @ delta_pos_world   (转到当前机体系)

Assumptions (same as before):
- Each sequence directory contains many *.png named as "<timestamp>.png" (float seconds).
- Each directory contains data.csv with column 'timestamp' and label columns:
  velcmd_x, velcmd_y, velcmd_z
- A sample = 4 consecutive frames (stride=1 by default)
- Label is taken from CSV row nearest to the last frame timestamp.
- Goal is taken from CSV row nearest to the future frame timestamp (last_frame + future_offset frames).
"""

import os
import glob
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Utils: timestamp parse
# -----------------------------
def parse_ts_from_png(path: str) -> Optional[float]:
    base = os.path.basename(path)
    if not base.endswith(".png"):
        return None
    s = base[:-4]
    try:
        return float(s)
    except Exception:
        return None


# -----------------------------
# Utils: CSV load
# -----------------------------
def load_csv(csv_path: str) -> pd.DataFrame:
    # data.csv 里第一列可能是空列（index），pandas 默认会读进来成为 "Unnamed: 0"
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError(f"CSV missing 'timestamp': {csv_path}")

    # 兼容 is_collide 是 "False"/"True" 字符串或 bool
    if "is_collide" in df.columns:
        if df["is_collide"].dtype == object:
            df["is_collide"] = df["is_collide"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            df["is_collide"] = df["is_collide"].astype(bool)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# -----------------------------
# Utils: nearest row by timestamp
# -----------------------------
def nearest_row(df: pd.DataFrame, t: float) -> Tuple[int, float]:
    """
    return (row_index, abs_time_diff)
    df must be sorted by timestamp.
    """
    ts = df["timestamp"].to_numpy()
    import bisect

    i = bisect.bisect_left(ts, t)
    cand = []
    if i < len(ts):
        cand.append(i)
    if i - 1 >= 0:
        cand.append(i - 1)
    best_i = min(cand, key=lambda j: abs(ts[j] - t))
    return int(best_i), float(abs(ts[best_i] - t))


# -----------------------------
# Utils: quaternion -> rotation
# NOTE: assumes quat order is w,x,y,z
# -----------------------------
def quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Return R_wb (body -> world) rotation matrix.
    Assumes right-handed coordinates.
    """
    ww, xx, yy, zz = qw * qw, qx * qx, qy * qy, qz * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz

    R = np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )
    return R


def world_delta_to_body(delta_world: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """
    delta_world: (3,)
    quat_wxyz: (4,) [w,x,y,z]
    return delta_body: (3,)
    """
    qw, qx, qy, qz = [float(x) for x in quat_wxyz]
    R_wb = quat_wxyz_to_rotmat(qw, qx, qy, qz)
    R_bw = R_wb.T  # world -> body
    return R_bw @ delta_world


# -----------------------------
# Core: make samples for one sequence dir
# -----------------------------
def make_samples_for_dir(
    seq_dir: str,
    s: int,
    stride: int,
    max_time_diff: float,
    drop_collide: bool,
    future_offset: int,
    require_future_match: bool = True,
) -> List[Dict[str, Any]]:
    csv_path = os.path.join(seq_dir, "data.csv")
    if not os.path.exists(csv_path):
        return []

    df = load_csv(csv_path)

    # required label columns
    required = ["velcmd_x", "velcmd_y", "velcmd_z"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing '{c}' in {csv_path}")

    # required state columns for goal/state logging
    required_state = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "quat_1", "quat_2", "quat_3", "quat_4"]
    missing_state = [c for c in required_state if c not in df.columns]
    if missing_state:
        raise ValueError(f"CSV missing state cols {missing_state} in {csv_path}")

    # collect png frames
    pngs = glob.glob(os.path.join(seq_dir, "*.png"))
    items: List[Tuple[float, str]] = []
    for p in pngs:
        t = parse_ts_from_png(p)
        if t is None:
            continue
        items.append((t, p))
    items.sort(key=lambda x: x[0])

    if len(items) < s:
        return []

    samples: List[Dict[str, Any]] = []
    for start in range(0, len(items) - s + 1, stride):
        window = items[start : start + s]
        ts_list = [x[0] for x in window]
        paths = [os.path.abspath(x[1]) for x in window]
        t_last = ts_list[-1]

        # ---- current row (label + state at t_last) ----
        ridx0, dt0 = nearest_row(df, t_last)
        if dt0 > max_time_diff:
            continue
        row0 = df.iloc[ridx0]

        if drop_collide and ("is_collide" in df.columns) and bool(row0["is_collide"]):
            continue

        velcmd = [float(row0["velcmd_x"]), float(row0["velcmd_y"]), float(row0["velcmd_z"])]

        # ---- future frame timestamp (by frame index, not by dt) ----
        idx_last = start + s - 1
        idx_future = idx_last + future_offset
        if idx_future >= len(items):
            continue  # not enough future frames

        t_future = float(items[idx_future][0])

        # ---- future row (for future position) ----
        ridx1, dt1 = nearest_row(df, t_future)
        if require_future_match and (dt1 > max_time_diff):
            continue
        row1 = df.iloc[ridx1]

        # ---- build current state ----
        curr_pos = np.array([row0["pos_x"], row0["pos_y"], row0["pos_z"]], dtype=np.float64)
        curr_vel = np.array([row0["vel_x"], row0["vel_y"], row0["vel_z"]], dtype=np.float64)
        curr_quat = np.array([row0["quat_1"], row0["quat_2"], row0["quat_3"], row0["quat_4"]], dtype=np.float64)

        # ---- build future goal ----
        future_pos = np.array([row1["pos_x"], row1["pos_y"], row1["pos_z"]], dtype=np.float64)
        delta_world = future_pos - curr_pos
        delta_body = world_delta_to_body(delta_world, curr_quat)

        sample: Dict[str, Any] = {
            "seq_dir": os.path.abspath(seq_dir),
            "frame_paths": paths,  # 4 帧 depth png 的绝对路径
            "frame_timestamps": ts_list,  # 4 帧时间戳（秒）
            "label_timestamp": float(row0["timestamp"]),
            "match_dt_sec": float(dt0),  # label 与最后一帧的时间差
            "label_velcmd": velcmd,  # 监督三轴速度命令

            # ✅ 新增：当前状态
            "curr_state": {
                "timestamp": float(row0["timestamp"]),
                "pos_world": curr_pos.tolist(),
                "vel_world": curr_vel.tolist(),
                "quat_wxyz": curr_quat.tolist(),
                "desired_vel": float(row0["desired_vel"]) if "desired_vel" in df.columns else None,
            },

            # ✅ 新增：goal_nav（未来 N 帧之后的位置差，含 body/world 两种）
            "goal_nav": {
                "future_offset_frames": int(future_offset),
                "future_frame_timestamp": float(t_future),
                "future_label_timestamp": float(row1["timestamp"]),
                "future_match_dt_sec": float(dt1),
                "future_pos_world": future_pos.tolist(),
                "delta_pos_world": delta_world.tolist(),
                "delta_pos_body": delta_body.tolist(),
            },
        }

        samples.append(sample)

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="datasets root, contains many sequence dirs")
    ap.add_argument("--frames", type=int, default=4, help="number of consecutive frames (S)")
    ap.add_argument("--stride", type=int, default=1, help="sliding window stride in frames")
    ap.add_argument("--max_time_diff", type=float, default=0.05, help="max abs time diff (sec) for csv label matching")
    ap.add_argument("--drop_collide", action="store_true", help="drop samples whose matched row has is_collide=True")

    # ✅ 新增：goal offset
    ap.add_argument("--future_offset", type=int, default=10, help="goal: use future N-th frame after last frame")
    ap.add_argument(
        "--no_require_future_match",
        action="store_true",
        help="do NOT enforce max_time_diff for future row matching (not recommended)",
    )

    ap.add_argument("--write_per_dir", action="store_true", help="write metadata.json into each sequence dir")
    ap.add_argument("--out_jsonl", type=str, default="", help="optional: write all samples into one JSONL file")
    args = ap.parse_args()

    root = args.root
    seq_dirs = [os.path.join(root, d) for d in os.listdir(root)]
    seq_dirs = [d for d in seq_dirs if os.path.isdir(d)]

    total = 0
    written_dirs = 0

    jsonl_f = None
    if args.out_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
        jsonl_f = open(args.out_jsonl, "w", encoding="utf-8")

    require_future_match = not args.no_require_future_match

    for sd in tqdm(sorted(seq_dirs), desc="Processing sequences"):
        samples = make_samples_for_dir(
            sd,
            s=args.frames,
            stride=args.stride,
            max_time_diff=args.max_time_diff,
            drop_collide=args.drop_collide,
            future_offset=args.future_offset,
            require_future_match=require_future_match,
        )
        if not samples:
            continue

        total += len(samples)

        if args.write_per_dir:
            out_path = os.path.join(sd, "metadata.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "seq_dir": os.path.abspath(sd),
                        "num_samples": len(samples),
                        "frames": args.frames,
                        "stride": args.stride,
                        "max_time_diff": args.max_time_diff,
                        "drop_collide": args.drop_collide,
                        "future_offset": args.future_offset,
                        "require_future_match": require_future_match,
                        "samples": samples,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            written_dirs += 1

        if jsonl_f is not None:
            for s in samples:
                jsonl_f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if jsonl_f is not None:
        jsonl_f.close()

    print("done.")
    print("dirs_scanned:", len(seq_dirs))
    print("dirs_written:", written_dirs if args.write_per_dir else 0)
    print("total_samples:", total)
    if args.out_jsonl:
        print("jsonl:", os.path.abspath(args.out_jsonl))


if __name__ == "__main__":
    main()
