#!/bin/bash
# Spec 011 line O: two lanes on GPU 1.
cd "$(dirname "$0")/../.."
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
E=experiments/011-limits-and-models
L=$E/results/o
export PYTHONPATH=. PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $L
laneA() { for s in 40k 160k 640k 2.56M 10M; do $PY $E/o1_overfit.py $s scratch 20; done; }
laneB() { for s in 40k 160k 640k 2.56M 10M; do $PY $E/o1_overfit.py $s a1 10; done; $PY $E/o1_overfit.py 40M scratch 10; }
laneA > $L/laneA.log 2>&1 &
laneB > $L/laneB.log 2>&1 &
wait
echo O1_DONE
