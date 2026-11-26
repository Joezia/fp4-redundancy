#!/bin/bash
# Spec 016 follow-up analyses (after run_gpu.sh / run_cpu.sh). Run from the repo root.
set -e
set -a && source .env && set +a
export PYTHONPATH=.:experiments/016-cross-matrix
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
D=experiments/016-cross-matrix
$PY $D/patterns.py deepseek; $PY $D/patterns.py kimi                     # form of the neuron similarity
$PY $D/shared_dirs.py deepseek --layers 0 9 12 20 35 39                   # shared directions or pairwise?
$PY $D/twins.py deepseek; $PY $D/twins.py deepseek --null                 # code-length check (not used in the report)
$PY $D/summarize.py > /dev/null
$PY $D/threshold_p99.py
$PY $D/make_tables.py > $D/results/tables.md
