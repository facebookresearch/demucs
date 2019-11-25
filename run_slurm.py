#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Run training from Slurm on all visible GPUs. Start only
one task per node as this script will spawn one child for each GPU.
This will not schedule a job but instead should be launched from srun/sbatch.
"""
import os
import subprocess as sp
import sys
import time

import torch as th

from demucs.utils import free_port


def main():
    args = sys.argv[1:]
    gpus = th.cuda.device_count()
    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])
    job_id = int(os.environ['SLURM_JOBID'])

    rank_offset = gpus * node_id
    hostnames = sp.run(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']],
                       capture_output=True,
                       check=True).stdout
    master_addr = hostnames.split()[0].decode('utf-8')

    if n_nodes == 1:
        port = free_port()
    else:
        port = 20_000 + (job_id % 40_000)
    args += ["--world_size", str(n_nodes * gpus), "--master", f"{master_addr}:{port}"]
    tasks = []

    print("About to go live", master_addr, node_id, n_nodes, file=sys.stderr)
    sys.stderr.flush()

    for gpu in range(gpus):
        kwargs = {}
        if gpu > 0:
            kwargs['stdin'] = sp.DEVNULL
            kwargs['stdout'] = sp.DEVNULL
            # We keep stderr to see tracebacks from children.
        tasks.append(
            sp.Popen(["python3", "-m", "demucs"] + args +
                     ["--rank", str(rank_offset + gpu)], **kwargs))
        tasks[-1].rank = rank_offset + gpu

    failed = False
    try:
        while tasks:
            for task in tasks:
                try:
                    exitcode = task.wait(0.1)
                except sp.TimeoutExpired:
                    continue
                else:
                    tasks.remove(task)
                    if exitcode:
                        print(f"Task {task.rank} died with exit code "
                              f"{exitcode}",
                              file=sys.stderr)
                        failed = True
                    else:
                        print(f"Task {task.rank} exited successfully")
            if failed:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        for task in tasks:
            task.terminate()
        raise
    if failed:
        for task in tasks:
            task.terminate()

        sp.run(["scancel", str(job_id)], check=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
