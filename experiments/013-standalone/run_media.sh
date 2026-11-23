#!/usr/bin/env bash
# Spec 013, media: standalone chains [R1] and [R1, Rk] on the 008 Part 2 report splits. Run from the repo root.
set -euo pipefail
PY=${PY:-/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python}
S=experiments/013-standalone/standalone.py
export PYTHONPATH=.
for d in kodak sqam librispeech; do
  echo "START $d $(date '+%F %T')"
  $PY $S media-run $d --workers ${WORKERS:-48}
  $PY $S media-summarize $d
  echo "END $d $(date '+%F %T')"
done
echo MEDIA_DONE
