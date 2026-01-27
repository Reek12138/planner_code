#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate metadata.json for 4-frame depth input -> 3-axis velocity command label.

Assumptions:
- Each sequence directory contains many *.png named as "<timestamp>.png" (float seconds).
- Each directory contains data.csv with column 'timestamp' and label columns:
  velcmd_x, velcmd_y, velcmd_z
- A sample = 4 consecutive frames (stride=1 by default)
- Label is taken from CSV row nearest to the last frame timestamp.
"""

import os
import glob
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate metadata.json for 4-frame depth input -> 3-axis velocity command label.

Assumptions:
- Each sequence directory contains many *.png named as "<timestamp>.png" (float seconds).
- Each directory contains data.csv with column 'timestamp' and label columns:
  velcmd_x, velcmd_y, velcmd_z
- A sample = 4 consecutive frames (stride=1 by default)
- Label is taken from CSV row nearest to the last frame timestamp.
"""

import os
import glob
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm


def parse_ts_from_png(path: str) -> Optional[float]:
    base = os.path.basename(path)
    if not base.endswith(".png"):
        return None
    s = base[:-4]
    try:
        return float(s)
    except Exception:
        return None


def load_csv(csv_path: str) -> pd.DataFrame:
    # data.csv 里第一列是空列（index），pandas 默认会读进来成为 "Unnamed: 0"
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


def nearest_row(df: pd.DataFrame, t: float) -> Tuple[int, float]:
    """
    return (row_index, abs_time_diff)
    df must be sorted by timestamp.
    """
    # binary search via pandas
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


def make_samples_for_dir(
    seq_dir: str,
    s: int,
    stride: int,
    max_time_diff: float,
    drop_collide: bool,
) -> List[Dict[str, Any]]:
    csv_path = os.path.join(seq_dir, "data.csv")
    if not os.path.exists(csv_path):
        return []

    df = load_csv(csv_path)
    required = ["velcmd_x", "velcmd_y", "velcmd_z"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing '{c}' in {csv_path}")

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

        ridx, dt = nearest_row(df, t_last)
        if dt > max_time_diff:
            continue

        row = df.iloc[ridx]

        if drop_collide and ("is_collide" in df.columns) and bool(row["is_collide"]):
            continue

        velcmd = [float(row["velcmd_x"]), float(row["velcmd_y"]), float(row["velcmd_z"])]

        sample = {
            "seq_dir": os.path.abspath(seq_dir),
            "frame_paths": paths,                 # 4 帧 depth png 的绝对路径
            "frame_timestamps": ts_list,          # 4 帧时间戳（秒）
            "label_timestamp": float(row["timestamp"]),
            "match_dt_sec": dt,                   # label 与最后一帧的时间差
            "label_velcmd": velcmd,               # 监督三轴速度命令
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

    # for sd in sorted(seq_dirs):
    for sd in tqdm(sorted(seq_dirs), desc="Processing sequences"):
        samples = make_samples_for_dir(
            sd,
            s=args.frames,
            stride=args.stride,
            max_time_diff=args.max_time_diff,
            drop_collide=args.drop_collide,
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


def parse_ts_from_png(path: str) -> Optional[float]:
    base = os.path.basename(path)
    if not base.endswith(".png"):
        return None
    s = base[:-4]
    try:
        return float(s)
    except Exception:
        return None


def load_csv(csv_path: str) -> pd.DataFrame:
    # data.csv 里第一列是空列（index），pandas 默认会读进来成为 "Unnamed: 0"
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


def nearest_row(df: pd.DataFrame, t: float) -> Tuple[int, float]:
    """
    return (row_index, abs_time_diff)
    df must be sorted by timestamp.
    """
    # binary search via pandas
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


def make_samples_for_dir(
    seq_dir: str,
    s: int,
    stride: int,
    max_time_diff: float,
    drop_collide: bool,
) -> List[Dict[str, Any]]:
    csv_path = os.path.join(seq_dir, "data.csv")
    if not os.path.exists(csv_path):
        return []

    df = load_csv(csv_path)
    required = ["velcmd_x", "velcmd_y", "velcmd_z"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing '{c}' in {csv_path}")

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

        ridx, dt = nearest_row(df, t_last)
        if dt > max_time_diff:
            continue

        row = df.iloc[ridx]

        if drop_collide and ("is_collide" in df.columns) and bool(row["is_collide"]):
            continue

        velcmd = [float(row["velcmd_x"]), float(row["velcmd_y"]), float(row["velcmd_z"])]

        sample = {
            "seq_dir": os.path.abspath(seq_dir),
            "frame_paths": paths,                 # 4 帧 depth png 的绝对路径
            "frame_timestamps": ts_list,          # 4 帧时间戳（秒）
            "label_timestamp": float(row["timestamp"]),
            "match_dt_sec": dt,                   # label 与最后一帧的时间差
            "label_velcmd": velcmd,               # 监督三轴速度命令
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

    # for sd in sorted(seq_dirs):
    for sd in tqdm(sorted(seq_dirs), desc="Processing sequences"):
        samples = make_samples_for_dir(
            sd,
            s=args.frames,
            stride=args.stride,
            max_time_diff=args.max_time_diff,
            drop_collide=args.drop_collide,
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
