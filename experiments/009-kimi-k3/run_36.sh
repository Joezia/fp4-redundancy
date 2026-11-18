#!/usr/bin/env bash
# Spec 009 first test: indicators and chain on the 36 Kimi-K3 matrices (run from the repository root).
set -euo pipefail
PY=${PY:-/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python}
E=experiments/009-kimi-k3
R=$E/results
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -a; source .env; set +a
# indicators (12 tasks = 12 experts) and parameter selection on the dev matrices run side by side
$PY -m redundancy.sweep run --config redundancy/configs/fp4_kimi.json --plan $R/plan_36.json \
    --out $R/indicators_run --workers 12 > $R/indicators_run.log 2>&1 &
$PY -m redundancy.increment_run fp4-devsel --snapshot-env KIMI_SNAPSHOT --samples data/009/kimi/matrices.json \
    --entries "$($PY -c "import json;print(','.join(json.load(open('$R/dev_entries.json'))['entries']))")" \
    --out $R/increment_dev --workers 7 > $R/increment_dev.log 2>&1
# chain on the 36 matrices, all three orders (as V6 of 008)
$PY -m redundancy.increment_run fp4 --snapshot-env KIMI_SNAPSHOT --samples data/009/kimi/matrices.json \
    --plan $R/plan_36.json --params $R/increment_dev/params.json --out $R/increment_36 --workers 12 \
    --orders default,alt,alt_after_R1 > $R/increment_36.log 2>&1
wait
echo done
