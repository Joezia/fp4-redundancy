#!/usr/bin/env bash
# Spec 015 follow-up (user request): (1) scheme R1 (16 raw codes, one table + scale table) on all DeepSeek tensors;
# (2) Kimi-K3 on a larger stratified sample (171 experts x 3 roles per layer = 47,196 tensors), both schemes.
set -euo pipefail
PY=${PY:-/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python}
S=experiments/015-ans-r1r2/ans_r1r2.py
export PYTHONPATH=.
set -a; source .env; set +a
W=${WORKERS:-64}
echo "START $(date '+%F %T')"
$PY $S count deepseek --workers $W
echo "DS_COUNT_DONE $(date '+%F %T')"
$PY $S code deepseek --schemes r1 --workers $W
echo "DS_CODE_DONE $(date '+%F %T')"
$PY $S count kimi --big --workers $W
echo "KIMI_COUNT_DONE $(date '+%F %T')"
$PY $S code kimi --big --schemes r1r2,r1 --workers $W
echo "KIMI_CODE_DONE $(date '+%F %T')"
$PY $S summarize deepseek --scheme r1 > experiments/015-ans-r1r2/results/summarize_r1_deepseek.log
for sch in r1r2 r1; do $PY $S summarize kimi --big --scheme $sch > experiments/015-ans-r1r2/results/summarize_${sch}_kimi_big.log; done
$PY $S chain deepseek --scheme r1 --workers $W > experiments/015-ans-r1r2/results/chain_r1_deepseek.log
$PY $S chain kimi --big --scheme r1 --workers $W > experiments/015-ans-r1r2/results/chain_r1_kimi.log
echo "ALL_DONE $(date '+%F %T')"
