"""PPM reference behavior and matched arithmetic-baseline accounting; synthetic only."""
import base64
import csv
import hashlib
import json
import sqlite3
import unittest
import zlib
from unittest.mock import patch

import test_checkpoint
from fp4_redundancy.catalog import Registry, builtin_registry
from fp4_redundancy.experiment import run_experiment
from fp4_redundancy.methods.native import NativeA
from fp4_redundancy.runner import evaluate_target
from fp4_redundancy.types import Component, EncodedArtifact, Target


def target(fmt, raw):
    return Target("synthetic", (len(raw),), (Component("weight", "packed_fp4" if fmt.startswith("fp4") else "other",
                  "unused", 0, len(raw), fmt),))


class PpmTests(unittest.TestCase):
    def test_reference_model_vector_and_unchanged_a1_bitstreams(self):
        # Generated independently with unmodified upstream Nayuki PpmModel.cpp
        # ab6ee50afec04d235a4b82d17f407f0fd2b42e9a, its dense FrequencyTable,
        # alphabet=16 + escape, order=3, the same YAECL interval normalization,
        # and our documented length-delimited uniform order-minus-one (no EOF).
        raw = bytes.fromhex("10325476") * 16
        artifact = NativeA("ppm").encode(target("fp4_packed_low_first", raw), {"weight": raw}, {})
        self.assertEqual(artifact.payload.hex(), "08dcd3da01e7a1e765c004")
        model = json.loads(artifact.side_information)["components"][0]["streams"][0]["model"]
        self.assertEqual(model["payload_bits"], 86)
        self.assertEqual(model["alphabet"], 16)
        # Captured before extending YAECL's interface. The existing coder's
        # interval equations, normalization and termination must stay identical.
        bf16 = b"".join(i.to_bytes(2, "little") for i in range(65536))
        cases = [("fp4_packed_low_first", bytes(range(256)), False, "964512a27340939173cf2cae6d101af566a04fd36e17a12885f9b58a1f917339"),
                 ("F8_E4M3", bytes(range(256))*2, False, "1a06f294f4576b9b2031ac24f642b3d26594e77026885c2438714117948bb647"),
                 ("F8_E4M3", bytes(range(256))*2, True, "a0665c259f92387ff308573d037738971d6f12429233d7e6d748ab510f3f2019"),
                 ("BF16", bf16, False, "09d0231be95a29f0327cb9c149fa2d0f4a1b58b1fd3fbd90b7f8aa2f9a24e2a3"),
                 ("BF16", bf16, True, "62ad8d39d6b468ccc8756f80aa67d2a22c6fc0d37cb13d71156dd7f3cea125b4")]
        for fmt, raw, split, expected in cases:
            with self.subTest(format=fmt, split=split):
                a = NativeA("arithmetic", split).encode(target(fmt, raw), {"weight": raw}, {})
                self.assertEqual(hashlib.sha256(a.payload).hexdigest(), expected)

    def test_absolute_and_net_savings_and_missing_baseline(self):
        raw = bytes.fromhex("10325476") * 1024
        unit = target("fp4_packed_low_first", raw)
        registry, cache, source = builtin_registry(), {}, {"weight": raw}
        encode, calls = NativeA.encode, []
        def counted(instance, *args):
            calls.append(instance.spec.id)
            return encode(instance, *args)
        with patch.object(NativeA, "encode", new=counted):
            baseline = evaluate_target(registry, "arithmetic", unit, components=source, _baseline_cache=cache)
            result = evaluate_target(registry, "ppm", unit, components=source, _baseline_cache=cache)
        self.assertEqual(calls.count("arithmetic"), 1)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["saving_bytes"], len(raw) - result["total_bytes"])
        comparison = result["baseline_comparison"]
        self.assertEqual(comparison["status"], "verified")
        self.assertEqual(comparison["total_bytes"], baseline["total_bytes"])
        self.assertEqual(comparison["net_saving_bytes"], baseline["total_bytes"] - result["total_bytes"])
        self.assertEqual(comparison["net_saving_fraction"], 1 - result["total_bytes"] / baseline["total_bytes"])
        only_ppm = Registry()
        only_ppm.register(lambda: NativeA("ppm"))
        result = evaluate_target(only_ppm, "ppm", unit, components=source)
        self.assertEqual(result["status"], "verified")
        self.assertIsNotNone(result["saving_bytes"])
        self.assertEqual(result["baseline_comparison"]["status"], "not_implemented")
        self.assertIsNone(result["baseline_comparison"]["net_saving_bytes"])

    def test_count_rescaling_independent_decode_and_invalid_configuration(self):
        # Slightly more than 2^20 FP4 codewords exercises count rescaling, in
        # addition to the full-BF16 test's context-cap resets (existing suite).
        raw = bytes(524300)
        unit = target("fp4_packed_low_first", raw)
        codec = NativeA("ppm")
        artifact = codec.encode(unit, {"weight": raw}, {})
        self.assertEqual(NativeA("ppm").decode(artifact), {"weight": raw})
        self.assertLess(artifact.total_bytes, len(raw))
        header = json.loads(artifact.side_information)
        model = header["components"][0]["streams"][0]["model"]
        binary = bytearray(base64.b64decode(model["table_b64"]))
        binary[8:12] = (4).to_bytes(4, "little")  # unimplemented order
        model["table_b64"] = base64.b64encode(binary).decode()
        with self.assertRaises(ValueError):
            codec.decode(EncodedArtifact(artifact.payload, json.dumps(header).encode()))

    def test_ppm_only_parallel_run_saves_both_states_and_both_benefits(self):
        fixture = test_checkpoint.CheckpointTests()
        fixture.setUp()
        try:
            output = fixture.root / "ppm"
            methods = ["ppm"]
            summary = run_experiment(fixture.snapshot, output, workers=2, chunk_bytes=2050, methods=methods)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(set(summary["methods"]), set(methods))
            db = sqlite3.connect(output / "results.sqlite3")
            net, paired_raw = dict.fromkeys(methods, 0), dict.fromkeys(methods, 0)
            try:
                for raw, blob in db.execute("SELECT result_json,state_zlib FROM chunks"):
                    record, state = json.loads(raw), json.loads(zlib.decompress(blob))
                    expected_states = set()
                    for result in record["results"]:
                        if result["status"] != "verified":
                            self.assertIsNone(result["baseline_comparison"])
                            continue
                        comparison = result["baseline_comparison"]
                        self.assertEqual(comparison["status"], "verified")
                        expected_states.update((result["method_id"], comparison["method_id"]))
                        self.assertEqual(result["saving_bytes"], result["original_bytes"] - result["total_bytes"])
                        self.assertEqual(comparison["net_saving_bytes"], comparison["total_bytes"] - result["total_bytes"])
                        net[result["method_id"]] += comparison["net_saving_bytes"]
                        paired_raw[result["method_id"]] += result["original_bytes"]
                    self.assertEqual(set(state), expected_states)
            finally:
                db.close()
            for mid, method in summary["methods"].items():
                self.assertEqual(method["saving_bytes"], method["verified_original_bytes"] - method["total_bytes"])
                self.assertEqual(method["baseline_comparison"]["net_saving_bytes"], net[mid])
                self.assertEqual(method["baseline_comparison"]["paired_original_bytes"], paired_raw[mid])
            with (output / "sizes.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    method = summary["methods"][row["method_id"]]
                    self.assertEqual(int(row["saving_bytes"]), method["saving_bytes"])
                    self.assertEqual(int(row["net_saving_bytes"]), method["baseline_comparison"]["net_saving_bytes"])
            resumed = run_experiment(fixture.snapshot, output, workers=1, chunk_bytes=2050, methods=methods, resume=True)
            self.assertEqual(resumed["methods"], summary["methods"])
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
