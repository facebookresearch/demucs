# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Dora the Explorer, special thank to @pierrestock.
"""
import argparse
import json
import logging
import shlex
import subprocess as sp
import time
from collections import namedtuple
from functools import partial
from itertools import product  # noqa
from pathlib import Path

import treetable as tt  # really great package for ascii art tables

from demucs.parser import get_name, get_parser

logger = logging.getLogger(__name__)
parser = get_parser()
logs = Path("logs")
logs.mkdir(exist_ok=True)

Job = namedtuple("Job", "args name sid")


def fname(name, kind):
    return logs / f"{name}.{kind}"


def get_sid(name):
    sid_file = fname(name, "sid")
    try:
        return int(open(sid_file).read().strip())
    except IOError:
        return None


def cancel(sid):
    sp.run(["scancel", str(sid)], check=True)


def reset_job(name):
    sid_file = fname(name, "sid")
    if sid_file.is_file():
        sid_file.unlink()


def get_done(name):
    done_file = fname(name, "done")
    return done_file.exists()


def get_metrics(name):
    json_file = fname(name, "json")
    try:
        return json.load(open(json_file))
    except IOError:
        return []


def schedule(name, args, nodes=1, partition="priority", time=2 * 24 * 60, large=True, gpus=8):
    log = fname(name, "log")
    command = [
        "sbatch",
        f"--job-name={name}",
        f"--output={log}.%t",
        "--mem=460G",
        f"--cpus-per-task={8*gpus}",
        f"--gpus={gpus}",
        f"--nodes={nodes}",
        "--tasks-per-node=1",
        f"--partition={partition}",
        # "--exclude=learnfair0748,learnfair0821",
        "--comment=Old codebase, not requeue, very few jobs",
        f"--time={time}",
    ]
    if large:
        command += ["--constraint=volta32gb"]
    srun_flags = f"--output={shlex.quote(str(log))}.%t"

    run_cmd = ["#!/bin/bash"]
    run_cmd.append(f"srun {srun_flags} python3 run_slurm.py " + " ".join(args))
    result = sp.run(command, stdout=sp.PIPE, input="\n".join(run_cmd).encode('utf-8'),
                    check=True).stdout.decode('utf-8')
    sid = int(result.strip().rsplit(' ', 1)[1])
    open(fname(name, "sid"), "w").write(str(sid))
    return sid


def _check(sids):
    cs_ids = ','.join(map(str, sids))
    result = sp.run(['squeue', f'-j{cs_ids}', '-o%A,%T,%P', '--noheader'],
                    check=True,
                    capture_output=True)
    lines = result.stdout.decode('utf-8').strip().split('\n')
    results = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        sid, status, partition = line.split(',', 2)
        sid = int(sid)
        results[sid] = status.lower()
    for sid in sids:
        if sid not in results:
            results[sid] = 'failed'
    return results


class Monitor:
    def __init__(self, cancel=False, base=[]):
        self.cancel = cancel
        self.base = base
        self.jobs = []

    def schedule(self, args, *vargs, **kwargs):
        args = self.base + args
        name = get_name(parser, parser.parse_args(args))
        sid = get_sid(name)
        if sid is None and not self.cancel:
            sid = schedule(name, args, *vargs, **kwargs)
        self.jobs.append(Job(sid=sid, name=name, args=args))

    def gc(self):
        names = set(job.name for job in self.jobs)
        for f in logs.iterdir():
            stem, suffix = f.name.rsplit(".", 1)
            if suffix == "sid":
                if stem not in names:
                    sid = get_sid(stem)
                    if sid is not None:
                        print(f"GCing {stem} / {sid}")
                        cancel(sid)
                        f.unlink()

    def check(self, trim=None, reset=False):
        to_check = []
        statuses = {}
        for job in self.jobs:
            if get_done(job.name):
                statuses[job.sid] = "done"
            elif job.sid is not None:
                to_check.append(job.sid)
        statuses.update(_check(to_check))

        if trim is not None:
            trim = len(get_metrics(self.jobs[trim].name))

        lines = []
        for index, job in enumerate(self.jobs):
            status = statuses.get(job.sid, "failed")
            if status in ["failed", "completing"] and reset:
                reset_job(job.name)
                status = "reset"

            meta = {'name': job.name, 'sid': job.sid, 'status': status[:2], "index": index}
            metrics = get_metrics(job.name)
            if trim is not None:
                metrics = metrics[:trim]
            meta["epoch"] = len(metrics)
            if metrics:
                metrics = metrics[-1]
            else:
                metrics = {}
            lines.append({'meta': meta, 'metrics': metrics})

        table = tt.table(shorten=True,
                         groups=[
                             tt.group("meta", [
                                 tt.leaf("index", align=">"),
                                 tt.leaf("name"),
                                 tt.leaf("sid", align=">"),
                                 tt.leaf("status"),
                                 tt.leaf("epoch", align=">")
                             ]),
                             tt.group("metrics", [
                                 tt.leaf("train", ".2%"),
                                 tt.leaf("valid", ".2%"),
                                 tt.leaf("best", ".2%"),
                                 tt.leaf("true_model_size", ".2f"),
                                 tt.leaf("compressed_model_size", ".2f"),
                             ])
                         ])
        print(tt.treetable(lines, table, colors=["0", "38;5;245"]))


def main():
    parser = argparse.ArgumentParser("grid.py")
    parser.add_argument("-c", "--cancel", action="store_true", help="Cancel all jobs")
    parser.add_argument(
        "-r",
        "--reset",
        action="store_true",
        help="Will reset the state of failed jobs. Next invocation will reschedule them")
    parser.add_argument("-t", "--trim", type=int, help="Trim metrics to match job with given index")
    args = parser.parse_args()

    monitor = Monitor(base=[], cancel=args.cancel)
    sched = partial(monitor.schedule, nodes=1)

    tasnet = ["--tasnet", "--split_valid", "--samples=80000", "--X=10", "-b", "32"]
    extra_path = Path.home() / "musdb_raw_44_allstems"
    extra = [f"--raw={extra_path}"]

    sched([])
    sched(extra)
    sched(tasnet)
    sched(tasnet + ["--repitch=0"])
    sched(tasnet + extra + ["--repitch=0"])

    ch48 = ["--channels=48"]
    sched(ch48)

    ch32 = ["--channels=32"]
    sched(ch32)

    # Main models
    for seed in [43, 44]:
        base = [f"--seed={seed}"]
        sched(base)
        sched(base + extra)
        sched(base + tasnet)
        sched(base + tasnet + extra)

    # Ablation study

    sched(["--no_glu"])
    sched(["--no_rewrite"])
    sched(["--context=1"])
    sched(["--rescale=0"])
    sched(["--mse"])
    sched(["--lstm_layers=0"])
    sched(["--lstm_layers=0", "--depth=7"])
    sched(["--no_resample"])
    sched(["--repitch=0"])

    # Quantization
    sched(["--diffq=0.0003"])

    if args.cancel:
        for job in monitor.jobs:
            if job.sid is not None:
                print(f"Canceling {job.name}/{job.sid}")
                cancel(job.sid)
        return

    names = [job.name for job in monitor.jobs]
    json.dump(names, open(logs / "herd.json", "w"))

    # Cancel any running job that was removed from the above sched calls.
    monitor.gc()
    while True:
        if args.reset:
            monitor.check(reset=True)
            return
        monitor.check(trim=args.trim)
        time.sleep(5 * 60)


if __name__ == "__main__":
    main()
