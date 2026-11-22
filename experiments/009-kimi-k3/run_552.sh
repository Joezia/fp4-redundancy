#!/usr/bin/env bash
# Spec 009, enlarged sample: chain model only (default order, the 009 parameters from increment_dev/params.json)
# on 552 matrices (layers 1-92, rounds 1-2). The 24 matrices already in increment_36 are reused (--skip).
# Run from the repository root.
set -euo pipefail
PY=${PY:-/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python}
E=experiments/009-kimi-k3
R=$E/results
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -a; source .env; set +a
$PY -m redundancy.increment_run fp4 --snapshot-env KIMI_SNAPSHOT --samples data/009/kimi/matrices.json \
    --plan $R/plan_552.json --params $R/increment_dev/params.json --out $R/increment_552 --workers ${WORKERS:-48} \
    --orders default --skip $R/increment_36
echo CHAIN552_DONE
