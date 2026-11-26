#!/bin/bash
# Spec 016 CPU queue. Run from the repo root.
set -a && source .env && set +a
export PYTHONPATH=.:experiments/016-cross-matrix
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
D=experiments/016-cross-matrix
# The first version (one pass over all blocks) was killed at aggregation by the per-user cgroup memory limit
# (~252 GiB); it was rerun as two disjoint hash subsets (what is in results/):
$PY $D/scan.py hash deepseek --workers 48 --sub 2 --part 0 > $D/results/b1_deepseek.log 2>&1
$PY $D/scan.py hash deepseek --workers 48 --sub 2 --part 1 > $D/results/b1_deepseek_p1.log 2>&1
$PY $D/pairs.py comp deepseek --workers 32 > $D/results/pairs_comp_deepseek.log 2>&1
$PY $D/pairs.py comp kimi --workers 32 > $D/results/pairs_comp_kimi.log 2>&1
$PY $D/joint.py deepseek --workers 24 > $D/results/joint_deepseek.log 2>&1
$PY $D/scan.py hash kimi --workers 48 > $D/results/b1_kimi.log 2>&1      # default: 1 of 16 hash subsets
$PY $D/joint.py kimi --workers 24 > $D/results/joint_kimi.log 2>&1
# after both queues: read back the raw bytes of every repeated hash
$PY $D/scan.py verify deepseek --workers 32 > $D/results/b1_verify_deepseek.log 2>&1
$PY $D/scan.py verify kimi --workers 32 > $D/results/b1_verify_kimi.log 2>&1
echo CPU_QUEUE_DONE
