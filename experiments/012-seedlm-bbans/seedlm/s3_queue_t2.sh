#!/bin/bash
# S3 on T2 for the two Table 1 configurations (both exponent rules); starts after the variants queue.
set -e
R=experiments/012-seedlm-bbans/results/s3
until grep -q S3_VARIANTS_DONE $R/queue_variants.log 2>/dev/null; do sleep 60; done
set -a; source .env; set +a
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:src OMP_NUM_THREADS=8
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
D=experiments/012-seedlm-bbans/seedlm
for cfg in "8 3 16" "12 4 16"; do for mode in range paper; do
  $PY $D/s3_run.py T2 $cfg $mode
  $PY $D/s3b_lossless.py T2 $cfg $mode
done; done
echo S3_T2_DONE
