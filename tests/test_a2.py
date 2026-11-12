"""A2 whole-codeword methods: real streams, model costs and matched comparisons."""
import base64
from collections import Counter
import json
import sqlite3
import struct
import unittest
from unittest.mock import patch
import zlib

import test_checkpoint
from fp4_redundancy.catalog import builtin_registry
from fp4_redundancy.experiment import run_experiment
from fp4_redundancy.methods.native import NativeA
from fp4_redundancy.runner import evaluate_target
from fp4_redundancy.types import Component, EncodedArtifact, Target

A2 = ("ppm", "ctw", "paq", "cabac", "cavlc", "neural_entropy")


def unit(fmt, raw):
    return Target("synthetic", (len(raw),), (Component("weight", "packed_fp4" if fmt.startswith("fp4") else "other",
                  "unused", 0, len(raw), fmt),))


class A2Tests(unittest.TestCase):
    def test_ctw_matches_closed_form_kt_for_constant_history(self):
        # For 32 zero symbols with zero initial history, all CTW nodes have
        # identical KT counts. Independently encoding CDF(0)=0 and
        # CDF(s)=(2*n+s)*2^24/(2*n+16), s>0, gives this 29-bit stream.
        raw = bytes(16)
        artifact = NativeA("ctw").encode(unit("fp4_packed_low_first",raw),{"weight":raw},{})
        self.assertEqual(artifact.payload.hex(),"00000008")
        model=json.loads(artifact.side_information)["components"][0]["streams"][0]["model"]
        self.assertEqual(model["payload_bits"],29)

    def test_whole_codewords_only_and_one_shared_zero_order_baseline(self):
        registry, cache = builtin_registry(), {}
        raw = bytes.fromhex("10325476") * 256
        target = unit("fp4_packed_low_first", raw)
        encode, calls = NativeA.encode, []
        def counted(instance, *args):
            calls.append(instance.spec.id)
            return encode(instance, *args)
        with patch.object(NativeA, "encode", new=counted):
            for technique in A2:
                self.assertNotIn(technique + "_fp8_bf16_exp", registry.ids())
                with self.assertRaises(ValueError):
                    NativeA(technique, True)
                method = registry.create(technique)
                self.assertEqual(method.spec.baseline_method_id, "arithmetic")
                result = evaluate_target(registry, technique, target, components={"weight": raw}, _baseline_cache=cache)
                self.assertEqual(result["status"], "verified", result["reason"])
                self.assertEqual(result["saving_bytes"], len(raw)-result["total_bytes"])
                paired = result["baseline_comparison"]
                self.assertEqual(paired["status"], "verified")
                self.assertEqual(paired["kind"], "complete_scheme" if technique == "cavlc" else "probability_model")
                self.assertEqual(paired["net_saving_bytes"], paired["total_bytes"]-result["total_bytes"])
                artifact = method.encode(target, {"weight": raw}, {})
                part = json.loads(artifact.side_information)["components"][0]
                self.assertFalse(part["split"])
                self.assertEqual(len(part["streams"]), 1)
                self.assertEqual(part["streams"][0]["count"], 2*len(raw))
        self.assertEqual(Counter(calls)["arithmetic"], 1)
        self.assertEqual(set(calls), set(A2) | {"arithmetic"})

    def test_cavlc_standard_vlc_vectors_extended_levels_and_runs(self):
        # coeff_token + trailing-one sign + total_zeros, from OpenH264 tables.
        for first, payload, bits in [(0, "80", 1), (1, "50", 4), (2, "17", 8)]:
            raw = bytes([first]) + bytes(7)
            artifact = NativeA("cavlc").encode(unit("fp4_packed_low_first", raw), {"weight": raw}, {})
            self.assertEqual(artifact.payload.hex(), payload)
            model = json.loads(artifact.side_information)["components"][0]["streams"][0]["model"]
            self.assertEqual(model["payload_bits"], bits)
        cases = [[0]*15+[65535], [65535]+[0]*14+[65534], [0]+[1,2]*7+[1],
                 [1,2]*8, [65535,0,32768,0,1,2,3,4,0,16,255,256,32767,0,1,0,65535]]
        for values in cases:
            raw = b"".join(v.to_bytes(2,"little") for v in values)
            target = unit("BF16",raw)
            artifact = NativeA("cavlc").encode(target,{"weight":raw},{})
            self.assertEqual(NativeA("cavlc").decode(artifact),{"weight":raw})

    def test_neural_weights_are_in_artifact_and_configuration_is_checked(self):
        raw = bytes(range(256))*2
        target = unit("F8_E4M3",raw)
        artifact = NativeA("neural_entropy").encode(target,{"weight":raw},{})
        header = json.loads(artifact.side_information)
        stream = header["components"][0]["streams"][0]
        binary = base64.b64decode(stream["model"]["table_b64"])
        self.assertEqual(struct.unpack_from("<8I",binary),(1,256,8,8,12,2048,4,81))
        self.assertEqual(len(binary),32+81*2+8)
        self.assertGreaterEqual(len(artifact.side_information),len(stream["model"]["table_b64"]))
        self.assertEqual(NativeA("neural_entropy").decode(artifact),{"weight":raw})
        damaged = bytearray(binary)
        damaged[28:32] = (80).to_bytes(4,"little")
        stream["model"]["table_b64"] = base64.b64encode(damaged).decode()
        with self.assertRaises(ValueError):
            NativeA("neural_entropy").decode(EncodedArtifact(artifact.payload,json.dumps(header).encode()))
        damaged = bytearray(binary)
        damaged[4:8] = (1).to_bytes(4,"little")
        stream["model"]["table_b64"] = base64.b64encode(damaged).decode()
        with self.assertRaises(ValueError):
            NativeA("neural_entropy").decode(EncodedArtifact(artifact.payload,json.dumps(header).encode()))

    def test_parallel_a2_run_keeps_pairing_scope_costs_and_baseline_states(self):
        fixture = test_checkpoint.CheckpointTests();fixture.setUp()
        try:
            output = fixture.root / "a2"
            summary = run_experiment(fixture.snapshot,output,workers=2,chunk_bytes=2050,methods=list(A2))
            self.assertEqual(summary["status"],"completed")
            self.assertEqual(set(summary["methods"]),set(A2))
            for name, method in summary["methods"].items():
                paired=method["baseline_comparison"]
                self.assertEqual(paired["paired_original_bytes"],method["verified_original_bytes"])
                self.assertEqual(paired["kind"],"complete_scheme" if name=="cavlc" else "probability_model")
                self.assertEqual(paired["net_saving_bytes"],paired["baseline_total_bytes"]-method["total_bytes"])
            db=sqlite3.connect(output/"results.sqlite3")
            try:
                for raw,blob in db.execute("SELECT result_json,state_zlib FROM chunks"):
                    record,state=json.loads(raw),json.loads(zlib.decompress(blob))
                    self.assertEqual(set(state),set(A2)|{"arithmetic"})
                    self.assertTrue(all(r["bitwise_verified"] for r in record["results"]))
                    self.assertTrue(all(r["baseline_comparison"]["status"]=="verified" for r in record["results"]))
            finally:db.close()
        finally:fixture.tearDown()


if __name__=="__main__":unittest.main()
