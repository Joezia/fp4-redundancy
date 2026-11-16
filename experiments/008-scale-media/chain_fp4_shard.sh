#!/usr/bin/env bash
# wait until a Part 2 run ends (pattern), then start one shard of the FP4 chain on the B plan (rounds 1-4)
set -u
pat=$1; shard=$2; out=$3; workers=$4
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
R=experiments/008-scale-media/results/fp4
export PYTHONPATH=.
set -a; source .env; set +a
if [ -n "$pat" ]; then while pgrep -f -- "$pat" >/dev/null; do sleep 20; done; fi
mkdir -p $R/$out
STOP=$(date -d "04:00 tomorrow" +%s)
nice -n 10 $PY -m redundancy.increment_run fp4 --plan $R/B/plan.json --params $R/increment_dev/params.json \
  --out $R/$out --workers $workers --orders default --first 172 --shard $shard --stop-at $STOP \
  --skip $R/increment_B,$R/increment_B_s1,$R/increment_B_s2,$R/increment_B_s3 >> $R/$out/run.log 2>&1
