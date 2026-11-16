#!/usr/bin/env bash
# wait for a dataset's dev parameter file, then run the report split (and, for Kodak, all 24 images)
set -u
ds=$1; split=$2; workers=$3
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
R=experiments/008-scale-media/results
export PYTHONPATH=.
until [ -f $R/$ds/dev/params.json ]; do sleep 10; done
mkdir -p $R/$ds/$split
$PY -m redundancy.increment_run run --dataset $ds --split $split --params $R/$ds/dev/params.json --out $R/$ds/$split --workers $workers >> $R/$ds/$split/run.log 2>&1
