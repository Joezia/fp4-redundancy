#!/usr/bin/env bash
# Spec 015 (ANS R1 + R2): full run on every MXFP4 tensor of both models. Run from the repo root.
set -euo pipefail
PY=${PY:-/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python}
S=experiments/015-ans-r1r2/ans_r1r2.py
export PYTHONPATH=.
set -a; source .env; set +a
W=${WORKERS:-64}
echo "START $(date '+%F %T')"
$PY $S code deepseek --workers $W          # pass 1 for DeepSeek already done (count)
echo "DS_CODE_DONE $(date '+%F %T')"
$PY $S count kimi --workers $W
echo "KIMI_COUNT_DONE $(date '+%F %T')"
$PY $S code kimi --workers $W
echo "KIMI_CODE_DONE $(date '+%F %T')"
for m in deepseek kimi; do $PY $S summarize $m > experiments/015-ans-r1r2/results/summarize_$m.log; done
for m in deepseek kimi; do $PY $S chain $m --workers $W > experiments/015-ans-r1r2/results/chain_$m.log; done
echo "ALL_DONE $(date '+%F %T')"
