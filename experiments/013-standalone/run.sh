#!/usr/bin/env bash
# Spec 013: standalone chains on DeepSeek (008, 516 matrices) and Kimi-K3 (009, 552 matrices). Run from the repo root.
set -euo pipefail
PY=${PY:-/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python}
S=experiments/013-standalone/standalone.py
export PYTHONPATH=.
set -a; source .env; set +a
for m in deepseek kimi; do
  echo "START $m $(date '+%F %T')"
  $PY $S run $m --workers ${WORKERS:-48}
  $PY $S summarize $m
  echo "END $m $(date '+%F %T')"
done
echo STANDALONE_DONE
