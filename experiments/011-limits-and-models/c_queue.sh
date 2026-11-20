#!/bin/bash
# Spec 011 line C: GPU work on cuda:0 in two lanes.
#   synth lane : C3 and C4 synthetic checks (small; may share the GPU with a real run)
#   real lane  : C2, C3, C4 on the real data, each only after its synthetic check passed (one real run at a time)
cd "$(dirname "$0")/../.."
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
E=experiments/011-limits-and-models
L=$E/results/c
export PYTHONPATH=. PYTHONUNBUFFERED=1

passed() {   # wait until results/c/synthetic_<x>.json exists, then return 0 if all_pass
  while [ ! -f $L/synthetic_$1.json ]; do sleep 60; done
  $PY -c "import json,sys; sys.exit(0 if json.load(open('$L/synthetic_$1.json'))['all_pass'] else 1)"
}

synth_lane() {
  while pgrep -f "[c]2_coord.py synth" > /dev/null; do sleep 30; done
  echo "c3 synth $(date +%T)"; $PY $E/c3_scope.py synth > $L/c3_synth.log 2>&1
  echo "c4 synth $(date +%T)"; $PY $E/c4_hyperprior.py synth > $L/c4_synth.log 2>&1
  echo "synth lane done $(date +%T)"
}

real_lane() {
  if passed c2; then echo "c2 run $(date +%T)"; $PY $E/c2_coord.py run > $L/c2_run.log 2>&1; else echo "c2 synthetic failed: no real run"; fi
  if passed c3; then echo "c3 run $(date +%T)"; $PY $E/c3_scope.py run > $L/c3_run.log 2>&1; else echo "c3 synthetic failed: no real run"; fi
  if passed c4; then echo "c4 run $(date +%T)"; $PY $E/c4_hyperprior.py run > $L/c4_run.log 2>&1; else echo "c4 synthetic failed: no real run"; fi
  echo "real lane done $(date +%T)"
}

synth_lane > $L/queue_synth.log 2>&1 &
real_lane > $L/queue_real.log 2>&1 &
wait
echo "C_QUEUE_DONE $(date +%T)"
