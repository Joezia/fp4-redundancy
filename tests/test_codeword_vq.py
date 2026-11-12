"""Categorical VQ mechanisms, paid metadata, parameter plumbing and exact bytes."""
import base64
import json
import os
import random
import sqlite3
import struct
import subprocess
import sys
import unittest

import test_checkpoint
from test_shared_patterns import Reader, target
from fp4_redundancy.catalog import builtin_registry
from fp4_redundancy.experiment import run_experiment
from fp4_redundancy.methods.codeword_vq import NativeCodewordVQ, ROOT, build_metadata
from fp4_redundancy.types import EncodedArtifact


def state(artifact, component=0):
    header = json.loads(artifact.side_information)
    part = header["components"][component]
    r = Reader(base64.b64decode(part["streams"][0]["model"]["table_b64"]))
    version, alphabet = r.get("<II")
    assert version == 1
    control = r.take(r.get("<I")[0])
    bits = r.get("<Q")[0]
    assert bits == part["streams"][0]["bytes"] * 8 and r.p == len(r.b)
    r = Reader(control)
    version, dim, requested, k, mode, objective, nt, nv = r.get("<IHHHBBII")
    assert version == 1
    book = r.take(r.get("<I")[0])
    itable = r.take(r.get("<I")[0]); ni = r.get("<Q")[0]
    emodel = r.take(r.get("<I")[0]); assert r.p == len(r.b)
    e = Reader(emodel); eversion, ea, ebits = e.get("<IIQ")
    global_model = e.take(e.get("<I")[0]); contexts = e.get("<I")[0]; active = 0
    for _ in range(contexts):
        flag = e.get("<B")[0]
        if flag:
            e.take(e.get("<I")[0]); active += 1
    assert eversion == 1 and ea == alphabet and e.p == len(e.b)
    return dict(dim=dim, requested=requested, k=k, mode=mode, objective=objective, nt=nt, nv=nv,
                book=book, index_bytes=ni, correction_bytes=part["streams"][0]["bytes"]-ni,
                index_model=itable, correction_model=global_model, correction_bits=ebits,
                contexts=contexts, active=active)


@unittest.skipUnless((ROOT / "build/libfp4_codeword_vq.so").is_file(), "Build codeword VQ first")
class CodewordVQTests(unittest.TestCase):
    def test_observed_codeword_centers_packed_width_and_both_nibble_orders(self):
        raw = bytes(range(256)) * 4
        for fmt in ("fp4_packed_low_first", "fp4_packed_high_first"):
            t = target(raw, fmt)
            for dim in (2, 4, 8):
                a = NativeCodewordVQ().encode(t, {"weight": raw}, {"vector_length": dim, "centers": 16})
                q = state(a); self.assertEqual(len(q["book"]), q["k"] * dim // 2)
                codes = [((byte >> shift) & 15) for byte in raw for shift in ((0, 4) if fmt.endswith("low_first") else (4, 0))]
                centers = [(byte >> shift) & 15 for byte in q["book"] for shift in (4, 0)]
                observed = {tuple(codes[i:i+dim]) for i in range(0, len(codes), dim)}
                self.assertTrue(all(tuple(centers[i:i+dim]) in observed for i in range(0, len(centers), dim)))
                self.assertEqual(NativeCodewordVQ().decode(a), {"weight": raw})
        # Categorical matching needs no inferred E2M1 numerical interpretation.
        self.assertTrue(NativeCodewordVQ().check_support(target(raw, "fp4_packed_low_first"), {}).applicable)

    def test_auto_selects_actual_smallest_complete_artifact(self):
        rng = random.Random(7)
        for raw in (bytes([0x10, 0x98]) * 256, bytes(rng.randrange(256) for _ in range(1027))):
            t = target(raw, "fp4_packed_low_first")
            sizes = {}
            for mode in ("shared", "conditioned", "auto"):
                a = NativeCodewordVQ().encode(t, {"weight": raw}, {"correction_model": mode})
                sizes[mode] = a.total_bytes
                self.assertEqual(NativeCodewordVQ().decode(a), {"weight": raw})
            self.assertEqual(sizes["auto"], min(sizes["shared"], sizes["conditioned"]))

    def test_repeated_codeword_vectors_are_actually_encoded_as_indices(self):
        # Sixteen complete, recurring two-codeword vectors, all exactly representable.
        raw = bytes(v | ((v ^ 1) << 4) for _ in range(512) for v in range(16))
        a = NativeCodewordVQ().encode(target(raw, "fp4_packed_low_first"), {"weight": raw},
                                      {"vector_length": 2, "centers": 16, "objective": "hamming"})
        q = state(a)
        self.assertEqual(q["k"], 16)
        self.assertEqual(q["correction_bits"], 2)
        self.assertLess(a.total_bytes, len(raw))
        self.assertEqual(NativeCodewordVQ().decode(a), {"weight": raw})

    def test_fresh_process_parameter_independence_special_bytes_and_corruption(self):
        raw = bytes.fromhex("0080ff7f1080fe") * 19
        a = NativeCodewordVQ().encode(target(raw, "F8_E8M0"), {"weight": raw},
                                      {"vector_length": 4, "centers": 64, "correction_model": "conditioned"})
        script = """import json,sys
from fp4_redundancy.methods.codeword_vq import NativeCodewordVQ
from fp4_redundancy.types import EncodedArtifact
r=json.load(sys.stdin)
print(NativeCodewordVQ().decode(EncodedArtifact(bytes.fromhex(r['p']),bytes.fromhex(r['s'])))['weight'].hex())
"""
        result = subprocess.run([sys.executable, "-c", script], input=json.dumps({"p": a.payload.hex(), "s": a.side_information.hex()}),
                                text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT/"src")})
        self.assertEqual(result.returncode, 0, result.stderr); self.assertEqual(bytes.fromhex(result.stdout), raw)
        for payload in (a.payload[:-1], a.payload+b"extra", bytes([a.payload[0]^255])+a.payload[1:]):
            with self.assertRaises(ValueError):
                NativeCodewordVQ().decode(EncodedArtifact(payload, a.side_information))
        for options in ({"centers": 5}, {"vector_length": True}, {"objective": "numeric"}, {"unknown": 1}):
            self.assertFalse(NativeCodewordVQ().check_support(target(raw), options).applicable)
        self.assertFalse(NativeCodewordVQ().check_support(target(bytes(4), "BF16"), {}).applicable)

    def test_parallel_selected_run_records_parameters_costs_and_resume(self):
        fixture = test_checkpoint.CheckpointTests(); fixture.setUp()
        try:
            output = fixture.root / "vq"
            params = {"vq_codeword": {"vector_length": 2, "centers": 4, "objective": "rate"}}
            result = run_experiment(fixture.snapshot, output, workers=2, chunk_bytes=2050,
                                    methods=["vq_codeword"], method_parameters=params)
            q = result["methods"]["vq_codeword"]
            self.assertEqual(set(q["statuses"]), {"verified", "inapplicable"})
            self.assertEqual(q["total_bytes"], q["payload_bytes"]+q["side_information_bytes"])
            self.assertEqual(q["baseline_comparison"]["kind"], "complete_scheme")
            config = json.loads((output/"run.json").read_text())["configuration"]
            self.assertEqual(config["method_parameters"], params)
            self.assertEqual(config["codeword_vq_build"]["library_sha256"], build_metadata()["library_sha256"])
            same = run_experiment(fixture.snapshot, output, chunk_bytes=2050, methods=["vq_codeword"],
                                  method_parameters=params, resume=True)
            self.assertEqual(result["methods"], same["methods"])
            with self.assertRaises(ValueError):
                run_experiment(fixture.snapshot, output, chunk_bytes=2050, methods=["vq_codeword"],
                               method_parameters={"vq_codeword": {"centers": 16}}, resume=True)
        finally:
            fixture.tearDown()
