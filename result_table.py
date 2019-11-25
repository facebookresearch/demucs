# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import treetable as tt

BASELINES = [
    'WaveUNet',
    'MMDenseLSTM',
    'OpenUnmix',
    'IRM2',
]
EVALS = Path("evals")
LOGS = Path("logs")
BASELINE_EVALS = Path("baselines")
STD_KEY = "seed"

parser = argparse.ArgumentParser("result_table.py")
parser.add_argument("-p",
                    "--paper",
                    action="store_true",
                    help="show results from the paper experiment")
parser.add_argument("-i", "--individual", action="store_true", help="no aggregation by seed")
parser.add_argument("-l", "--latex", action="store_true", help="output easy to copy latex")
parser.add_argument("metric", default="SDR", nargs="?")
args = parser.parse_args()

if args.paper:
    EVALS = Path("results/evals")
    LOGS = Path("results/logs")


def read_track(metric, results, pool=np.nanmedian):
    all_metrics = {}
    for target in results["targets"]:
        source = target["name"]
        metrics = [frame["metrics"][metric] for frame in target["frames"]]
        metrics = pool(metrics)
        all_metrics[source] = metrics
    return all_metrics


def read(metric, path, pool=np.nanmedian):
    all_metrics = defaultdict(list)
    for f in path.iterdir():
        if f.name.endswith(".json.gz"):
            results = json.load(gzip.open(f, "r"))
            metrics = read_track(metric, results, pool=pool)
            for source, value in metrics.items():
                all_metrics[source].append(value)
    return {key: np.array(value) for key, value in all_metrics.items()}


all_stats = defaultdict(list)
for name in BASELINES:
    all_stats[name] = [read(args.metric, BASELINE_EVALS / name / "test")]

for path in EVALS.iterdir():
    results = path / "results" / "test"
    if not results.exists():
        continue
    if not args.paper and not (LOGS / (path.name + ".done")).exists():
        continue
    name = path.name
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
    stats = read(args.metric, results)
    if (not stats or len(stats["drums"]) != 50):
        print(f"Missing stats for {results}", file=sys.stderr)
    else:
        all_stats[name].append(stats)

metrics = [tt.leaf("score", ".2f"), tt.leaf("std", ".2f")]
sources = ["drums", "bass", "other", "vocals"]

mytable = tt.table([tt.leaf("name"), tt.group("all", metrics + [tt.leaf("count")])] +
                   [tt.group(source, metrics) for idx, source in enumerate(sources)])

lines = []
for name, stats in all_stats.items():
    line = {"name": name}
    if 'accompaniment' in stats:
        del stats['accompaniment']
    alls = []
    for source in sources:
        stat = [np.nanmedian(s[source]) for s in stats]
        alls.append(stat)
        line[source] = {"score": np.mean(stat), "std": np.std(stat) / len(stat)**0.5}
    alls = np.array(alls)
    line["all"] = {
        "score": alls.mean(),
        "std": alls.mean(0).std() / alls.shape[1]**0.5,
        "count": alls.shape[1]
    }
    lines.append(line)


def latex_number(m):
    out = f"{m['score']:.2f}"
    if m["std"] > 0:
        std = "{:.2f}".format(m["std"])[1:]
        out += f" $\\scriptstyle\\pm {std}$"
    return out


lines.sort(key=lambda x: -x["all"]["score"])
if args.latex:
    for line in lines:
        cols = [
            line['name'],
            latex_number(line["all"]),
            latex_number(line["drums"]),
            latex_number(line["bass"]),
            latex_number(line["other"]),
            latex_number(line["vocals"])
        ]
        print(" & ".join(cols) + r" \\")
else:
    print(tt.treetable(lines, mytable, colors=['33', '0']))
