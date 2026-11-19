#!/usr/bin/env bash
# Spec 011 GPU queue: wait for A1, then A3 (first round) on the size with the best validation code length,
# then A2a (1M, 10M), A2b on the size with the better A2a validation, then A2c (online) on the best A2 model.
set -u
cd "$(dirname "$0")/../.."
PY=/home/zhouziang/.local/share/miniforge3/envs/lab/bin/python
E=experiments/011-limits-and-models
export PYTHONPATH=. PYTHONUNBUFFERED=1
set -a; source .env; set +a
while pgrep -f "a1.py run" >/dev/null; do sleep 30; done
SIZE=$($PY -c "
import json
rows=[json.loads(l) for l in open('$E/results/a/a1.jsonl') if json.loads(l)['N']==10**9]
print(min(rows,key=lambda r:r['train']['best_val'])['size'])")
echo "A3 size (best validation): $SIZE  $(date +%T)"
$PY $E/a3.py $SIZE 160000000 > $E/results/a/a3.log 2>&1
echo "A3 done $(date +%T)"
$PY $E/a2.py train a 1M > $E/results/a/a2a_1M.log 2>&1
$PY $E/a2.py train a 10M > $E/results/a/a2a_10M.log 2>&1
BEST=$($PY -c "
import json
rows=[json.loads(l) for l in open('$E/results/a/a2.jsonl') if json.loads(l)['variant']=='a']
print(min(rows,key=lambda r:r['train']['best_val'])['size'])")
echo "A2b size (best A2a validation): $BEST  $(date +%T)"
$PY $E/a2.py train b $BEST > $E/results/a/a2b.log 2>&1
TAG=$($PY -c "
import json
rows=[json.loads(l) for l in open('$E/results/a/a2.jsonl')]
print(min(rows,key=lambda r:r['train']['best_val'])['tag'])")
echo "A2c model (best A2 validation): $TAG  $(date +%T)"
$PY $E/a2.py online $TAG offline > $E/results/a/a2c_offline.log 2>&1
$PY $E/a2.py online $TAG random > $E/results/a/a2c_random.log 2>&1
echo "QUEUE_DONE $(date +%T)"
