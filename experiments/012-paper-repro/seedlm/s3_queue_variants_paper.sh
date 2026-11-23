#!/bin/bash
# S3 extension variants (paper exponent rule): same grid with the printed (paper) exponent rule; starts after the T2 queue.
# Starts after the T2 queue has finished. Run from the repo root (GPU 1).
set -e
R=experiments/012-paper-repro/results/s3
until grep -q S3_T2_DONE $R/queue_t2.log 2>/dev/null; do sleep 60; done
set -a; source .env; set +a
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:src OMP_NUM_THREADS=8
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
D=experiments/012-paper-repro/seedlm
for K in 8 12 16; do
  for C in 8 16 32; do
    for P in 1 2 3 4; do
      $PY $D/s3_run.py train8 $C $P $K paper
      $PY $D/s3b_lossless.py train8 $C $P $K paper
      $PY $D/s3_run.py T1 $C $P $K paper
      $PY $D/s3b_lossless.py T1 $C $P $K paper
    done
  done
done
echo S3_VARIANTS_PAPER_DONE
