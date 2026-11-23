#!/bin/bash
# S3-c final check (user request, 2026-11-22): Table 1 configurations (8,3,16) and (12,4,16), range rule.
# stage 1: side-info entropy coding on train8 + all of T1 (no search); shared exponent chosen on train8;
#          sharing and code-length seed choice / bits-back bound on 4 T1 matrices (one per layer);
# stage 2: sharing and code-length seed choice on the remaining 20 T1 matrices (finished matrices are skipped).
# Run from the repo root (GPU 1).
set -e
set -a; source .env; set +a
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:src OMP_NUM_THREADS=8
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
Q=experiments/012-seedlm-bbans/seedlm/s3c_check.py
SUB=L0.E191.w1,L13.E127.w2,L26.E336.w3,L39.E88.w1
for cfg in "8 3 16" "12 4 16"; do
  $PY $Q side train8 $cfg
  $PY $Q side T1 $cfg
  $PY $Q e0 $cfg
  echo "S3C_STEP side+e0 $cfg $(date +%m-%d_%H:%M)"
done
for stage in 1 2; do
  for cfg in "8 3 16" "12 4 16"; do
    if [ $stage = 1 ]; then export S3_ONLY=$SUB; else unset S3_ONLY; fi
    $PY $Q share T1 $cfg
    $PY $Q bb T1 $cfg
    echo "S3C_STEP stage=$stage $cfg $(date +%m-%d_%H:%M)"
  done
  echo "S3C_STAGE${stage}_DONE $(date +%m-%d_%H:%M)"
done
