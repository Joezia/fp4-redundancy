#!/bin/bash
# S3-a then S3-b for the two Table 1 configurations, both exponent rules (run from the repo root, GPU 1).
set -e
set -a; source .env; set +a
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:src OMP_NUM_THREADS=8
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
D=experiments/012-seedlm-bbans/seedlm
for cfg in "8 3 16" "12 4 16"; do
  for mode in range paper; do
    for set in train8 T1; do $PY $D/s3_run.py $set $cfg $mode; done
    $PY $D/s3b_lossless.py train8 $cfg $mode
    $PY $D/s3b_lossless.py T1 $cfg $mode
  done
done
echo S3_PAPER_DONE
