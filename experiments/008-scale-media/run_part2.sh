#!/usr/bin/env bash
# Spec 008 Part 2 and V6: parameter selection on the dev splits, then the report runs.
# Run from the repository root after V1-V5 pass. Data in data/008 (see data_sources.md).
set -euo pipefail
PY=${PY:-/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python}
E=experiments/008-scale-media
R=$E/results
export PYTHONPATH=.

step=${1:-all}

if [[ $step == devsel || $step == all ]]; then
  # parameters only from the dev splits (Kodak 01-12, SQAM odd tracks, LibriSpeech dev-clean, FP4 pilot matrices)
  $PY -m redundancy.increment_run devsel --dataset kodak --out $R/kodak/dev --workers 12 &
  $PY -m redundancy.increment_run devsel --dataset sqam --out $R/sqam/dev --workers 12 --limit 12 --max-seconds 30 &
  $PY -m redundancy.increment_run devsel --dataset librispeech --out $R/librispeech/dev --workers 12 --limit 24 &
  $PY -m redundancy.increment_run fp4-devsel --out $R/fp4/increment_dev --workers 7 \
      --entries L0.E200.w1,L13.E147.w1,L26.E172.w1,L1.E353.w2,L23.E338.w2,L7.E322.w3,L38.E206.w3 &
  wait
fi
