#!/bin/bash
# LLM.265 on FP4: QP sweep on train8 (chooses S) then T1. Run from the repo root.
set -e
set -a && source .env && set +a
export PYTHONPATH=.:experiments/010-transforms:experiments/010-transforms/tx:experiments/011-limits-and-models
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
T=${TARGETS:-raw8,qp10,qp16,qp22,qp28,qp34,qp40,qp46,qp51}
$PY experiments/012-seedlm-bbans/llm265/l265_run.py train8 $T
$PY experiments/012-seedlm-bbans/llm265/l265_run.py T1 $T
echo L265_DONE
