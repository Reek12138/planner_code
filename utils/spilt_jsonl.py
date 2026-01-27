#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split one JSONL into train/val with reproducible shuffle.

Default: 9:1 split (val_ratio=0.1), seed=42
Input:  metadata_all.jsonl (one JSON object per line)
Output: train.jsonl, val.jsonl
"""

import os
import json
import argparse
import random
from typing import List


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--out_train", type=str, required=True)
    ap.add_argument("--out_val", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1, help="e.g. 0.1 => 9:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_order", action="store_true", help="no shuffle, just first (1-val) train then val")
    args = ap.parse_args()

    in_jsonl = os.path.abspath(args.in_jsonl)
    out_train = os.path.abspath(args.out_train)
    out_val = os.path.abspath(args.out_val)

    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    os.makedirs(os.path.dirname(out_val), exist_ok=True)

    total = count_lines(in_jsonl)
    if total <= 0:
        raise RuntimeError(f"Empty jsonl: {in_jsonl}")

    val_n = int(round(total * float(args.val_ratio)))
    val_n = max(1, min(val_n, total - 1))
    train_n = total - val_n

    # Build shuffled indices (memory: O(N) ints; N~1e5 OK)
    idxs: List[int] = list(range(total))
    if not args.keep_order:
        rnd = random.Random(args.seed)
        rnd.shuffle(idxs)

    train_set = set(idxs[:train_n])
    # idxs[train_n:] are val

    n_train = 0
    n_val = 0

    with open(in_jsonl, "r", encoding="utf-8") as fin, \
         open(out_train, "w", encoding="utf-8") as ftr, \
         open(out_val, "w", encoding="utf-8") as fva:

        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            # optional quick sanity check: valid json
            # (comment out if you want max speed)
            try:
                json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Invalid JSON at line {i+1}: {e}")

            if i in train_set:
                ftr.write(line + "\n")
                n_train += 1
            else:
                fva.write(line + "\n")
                n_val += 1

    print("done.")
    print("in_jsonl:", in_jsonl)
    print("total:", total)
    print("seed:", args.seed)
    print("val_ratio:", args.val_ratio)
    print("train:", n_train, "->", out_train)
    print("val:", n_val, "->", out_val)


if __name__ == "__main__":
    main()
