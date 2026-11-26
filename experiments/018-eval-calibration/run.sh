#!/usr/bin/env bash
# 018: calibrate the lossy evaluation framework on gpt-oss-20b with known perturbations.
# Run from the repo root. GPU 1 only. Logs and summaries go to results/.
set -euo pipefail
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
OUT=experiments/018-eval-calibration/results
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -z "${SKIP_CHECKS:-}" ]; then
$PY -m lossy_eval.synthcheck "${SYNTH_DIR:-/tmp/lossy_eval_synth}" 2>&1 | grep --line-buffered -v Warning | tee "$OUT/synthcheck.log"
$PY -m lossy_eval.selfcheck --hf-cpu --out "$OUT/selfcheck.json" 2>&1 | grep --line-buffered -v Warning | tee "$OUT/selfcheck.log"
fi
$PY -m lossy_eval.run --model gpt-oss-20b \
  --suite wikitext2 c4 arc_easy arc_challenge hellaswag piqa winogrande mmlu \
  --ppl-windows 64 --mc-limit 2000 \
  --methods identity white_noise:-40 white_noise:-20 white_noise:-10 white_noise:-2.4 \
            block_rtn:4 block_rtn:3 block_rtn:2 \
  --out "$OUT" 2>&1 | grep --line-buffered -v Warning | tee -a "$OUT/run.log"
