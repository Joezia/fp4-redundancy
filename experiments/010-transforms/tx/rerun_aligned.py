"""Rerun of the aligned variants of A6 (expert axis) and C3 expert order after the layout fix (each expert coded in
its own layout). The earlier aligned rows (residual / codes coded in the permuted layout, which breaks w2's 32-blocks)
were removed from the result files.

    PYTHONPATH=. python experiments/010-transforms/tx/rerun_aligned.py <A6_klt|C3_expert_order>
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import txcore as T  # noqa: E402
import tx_global as G  # noqa: E402
import tx_stage2 as S  # noqa: E402

meth = sys.argv[1]
vs = [v for v in G.GROUP_VARIANTS[meth] if v.get("aligned")]
out = S.OUT / (f"{meth}_expert_axis_aligned.jsonl" if meth == "A6_klt" else f"{meth}_aligned.jsonl")
T.run_jobs(G.group_job, S.group_jobs(meth, vs, 8), out, workers=5, done_key="gkey")
