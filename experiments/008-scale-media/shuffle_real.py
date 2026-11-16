"""Shuffle baseline on real audio (spec 008 expectation: every pattern but R1 about 0 after shuffling).
One SQAM report track (first 30 s, samples of each channel shuffled) and one LibriSpeech test-clean file."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json  # noqa: E402
import sys

import numpy as np

from redundancy import increment as I
from redundancy.adapters import media

out = {}
for ds, path, secs in [("sqam", "data/008/sqam/flac/16.flac", 30), ("librispeech", media.items("librispeech", "report")[0], None)]:
    x, meta = media.load(ds, path, secs)
    P = dict(I.AUDIO_DEFAULT)
    P.update(json.load(open(f"experiments/008-scale-media/results/{ds}/dev/params.json")))
    rng = np.random.default_rng(8100)
    xs = np.stack([rng.permutation(c) for c in x])
    res = {}
    for name, arr in [("original", x), ("shuffled", xs)]:
        r = I.run_chain(lambda f: I.audio_codelen(arr, f, P), ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"], 16 * arr.size)
        res[name] = {"increment": {k: 100 * v for k, v in r["increment"].items()},
                     "if_forced": {k: 100 * v for k, v in r["if_forced"].items()}, "total": 100 * r["total"]}
    out[ds] = {"item": path.split("/")[-1], "seconds": meta["seconds_used"], **res}
    print(ds, json.dumps(res["shuffled"]["increment"]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
