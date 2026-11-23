#!/bin/bash
# S3 extension: K in {18, 20}, same C in {8,16,32}, P = 4 only (user request, 2026-11-22), range exponent rule.
# Estimated ~11.5 h for all 32 matrices; a reduced first pass still runs first so a first answer comes early:
#   stage 1: 8 training matrices (needed to choose the sub-bin count) + 4 T1 matrices (one per layer);
#   stage 2: the remaining 20 T1 matrices (the same commands; finished matrices are skipped).
# Run from the repo root (GPU 1).
set -e
R=experiments/012-paper-repro/results/s3
set -a; source .env; set +a
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:src OMP_NUM_THREADS=8
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
D=experiments/012-paper-repro/seedlm
SUB=L0.E191.w1,L13.E127.w2,L26.E336.w3,L39.E88.w1
for stage in 1 2; do
  for K in 18 20; do
    for C in 8 16 32; do
      for P in 4; do
        if [ $stage = 1 ]; then
          $PY $D/s3_run.py train8 $C $P $K range
          $PY $D/s3b_lossless.py train8 $C $P $K range
          S3_ONLY=$SUB $PY $D/s3_run.py T1 $C $P $K range
          S3_ONLY=$SUB $PY $D/s3b_lossless.py T1 $C $P $K range
        else
          $PY $D/s3_run.py T1 $C $P $K range
          $PY $D/s3b_lossless.py T1 $C $P $K range
        fi
        echo "K1820_STEP stage=$stage K=$K C=$C P=$P $(date +%m-%d_%H:%M)"
      done
    done
  done
  echo "K1820_STAGE${stage}_DONE $(date +%m-%d_%H:%M)"
done
