# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import treetable as tt

LOGS = Path("logs")
STD_KEY = "seed"
METRIC = "best"

parser = argparse.ArgumentParser("result_table.py")
parser.add_argument("-p",
                    "--paper",
                    action="store_true",
                    help="show results from the paper experiment")
parser.add_argument("-i", "--individual", action="store_true", help="no aggregation by seed")
args = parser.parse_args()

if args.paper:
    LOGS = Path("results/logs")

all_stats = defaultdict(list)

for path in LOGS.iterdir():
    if path.suffix == ".json" and (args.paper or path.with_suffix(".done").exists()):
        metric = json.load(open(path))[-1][METRIC]
        name = path.stem
        model = "Demucs"
        if "tasnet" in name:
            model = "Tasnet"
        if name == "default":
            parts = []
        else:
            parts = [p.split("=") for p in name.split(" ") if "tasnet" not in p]
        if not args.individual:
            parts = [(k, v) for k, v in parts if k != STD_KEY]
        name = model + " " + " ".join(f"{k}={v}" for k, v in parts)
        all_stats[name].append(metric)

metrics = [tt.leaf("score", ".4f"), tt.leaf("std", ".3f"), tt.leaf("count", ".2f")]

mytable = tt.table([tt.leaf("name"), tt.group("valid", metrics)])

lines = []
for name, stats in all_stats.items():
    line = {"name": name}
    stats = np.array(stats)
    line["valid"] = {
        "score": stats.mean(),
        "std": stats.std() / stats.shape[0]**0.5,
        "count": stats.shape[0]
    }
    lines.append(line)
lines.sort(key=lambda x: x["valid"]["score"])
print(tt.treetable(lines, mytable, colors=['33', '0']))
