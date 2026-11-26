#!/bin/bash
# Spec 016 GPU queue (GPU 1 only). Run from the repo root.
set -e
set -a && source .env && set +a
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:experiments/016-cross-matrix
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
D=experiments/016-cross-matrix
$PY $D/pairs.py gpu deepseek > $D/results/pairs_gpu_deepseek.log 2>&1
$PY $D/pairs.py gpu kimi > $D/results/pairs_gpu_kimi.log 2>&1
[ -f data/016/queries_kimi.npz ] || $PY $D/scan.py queries kimi > $D/results/queries_kimi.log 2>&1
$PY $D/scan.py gpu deepseek --workers 16 > $D/results/scan_gpu_deepseek.log 2>&1
$PY $D/scan.py gpu kimi --workers 16 --random-order --nq-vec 1024 --nq-patch 64 64 32 > $D/results/scan_gpu_kimi.log 2>&1
echo GPU_QUEUE_DONE
