#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import math

if len(sys.argv) != 2:
    print("Usage: python check_label_norm_range.py <metadata.jsonl>")
    sys.exit(1)

jsonl_path = sys.argv[1]

mins = [1e9, 1e9, 1e9]
maxs = [-1e9, -1e9, -1e9]
abs_max = [0.0, 0.0, 0.0]

count = 0
over_1 = 0
over_1_2 = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        label = item["label_velcmd"]
        desired = item["curr_state"].get("desired_vel", None)

        if desired is None or desired <= 0:
            continue

        norm = [v / desired for v in label]

        for i in range(3):
            mins[i] = min(mins[i], norm[i])
            maxs[i] = max(maxs[i], norm[i])
            abs_max[i] = max(abs_max[i], abs(norm[i]))

            if abs(norm[i]) > 1.0:
                over_1 += 1
            if abs(norm[i]) > 1.2:
                over_1_2 += 1

        count += 1

print("========== Normalized label statistics ==========")
print(f"Total samples checked: {count}")
print()
print("Per-dimension min:")
print(mins)
print("Per-dimension max:")
print(maxs)
print("Per-dimension abs max:")
print(abs_max)
print()
print(f"|value| > 1.0 count : {over_1}")
print(f"|value| > 1.2 count : {over_1_2}")
print("=================================================")
