"""Native upstream codecs: all codewords, separate-process decode, and real runner plumbing."""

import base64
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch
import zlib

import test_checkpoint
import fp4_redundancy.experiment as experiment
from fp4_redundancy.catalog import builtin_registry
from fp4_redundancy.experiment import run_experiment
from fp4_redundancy.methods.native import NativeA, ROOT, build_metadata
from fp4_redundancy.types import Component, EncodedArtifact, Target


@unittest.skipUnless((ROOT / "build/libfp4_entropy.so").is_file(), "Run python3 scripts/build_native.py")
class NativeTests(unittest.TestCase):
    def test_complete_codewords_and_independent_scale_models(self):
        bf16 = b"".join(i.to_bytes(2, "little") for i in range(65536))
        for fmt, raw in (("fp4_packed_low_first", bytes(range(256))),
                         ("fp4_packed_high_first", bytes(range(256))),
                         ("F8_E4M3", bytes(range(256))), ("F8_E5M2", bytes(range(256))),
                         ("BF16", bf16), ("F32", bytes(range(256)))):
            target = Target("synthetic", (len(raw),), (
                Component("weight", "packed_fp4" if fmt.startswith("fp4") else "other", "missing", 0, len(raw), fmt),
                Component("scale", "scale", "missing", 0, 19, "F8_E8M0")), {"fp4_value_format": "E2M1FN"})
            source = {"weight": raw, "scale": bytes([255]) * 19}
            for method_id in builtin_registry().ids():
                with self.subTest(format=fmt, method=method_id):
                    codec = builtin_registry().create(method_id)
                    if codec.exponent_split and fmt not in ("F8_E4M3", "F8_E5M2", "BF16"):
                        self.assertFalse(codec.check_support(target, {}).applicable)
                        continue
                    if method_id == "vq_codeword" and fmt == "BF16":
                        self.assertFalse(codec.check_support(target, {}).applicable)
                        with self.assertRaises(ValueError):
                            codec.encode(target, source, {})
                        continue
                    artifact = codec.encode(target, source, {})
                    self.assertEqual(builtin_registry().create(method_id).decode(artifact), source)
                    header = json.loads(artifact.side_information)
                    weight, scale = header["components"]
                    self.assertEqual(len(weight["streams"]), 2 if codec.exponent_split else 1)
                    self.assertEqual(len(scale["streams"]), 1)
                    self.assertNotEqual(weight["streams"][0]["model"], scale["streams"][0]["model"])
                    self.assertEqual(sum(s["bytes"] for p in header["components"] for s in p["streams"]), len(artifact.payload))

    def test_rans_known_stream(self):
        raw = bytes([0, 1])
        target = Target("synthetic", (2,), (Component("weight", "other", "missing", 0, 2, "F8_E4M3"),))
        artifact = NativeA("rans").encode(target, {"weight": raw}, {})
        self.assertEqual(artifact.payload, bytes([0, 16, 0, 2]))
        self.assertEqual(NativeA("rans").decode(artifact), {"weight": raw})

    def test_rice_parameter_and_reference_bitstream(self):
        # Independent scalar definition: quotient zeroes, stop one, k-bit remainder.
        # Include k=0, a unary sequence spanning many words, and the uint16 limit.
        for values in ([0] * 17, list(range(6)), [65535] * 3,
                       [0] * 32768 + [65535], [0, 15, 16, 255, 256, 32768, 65535]):
            with self.subTest(count=len(values), maximum=max(values)):
                raw = b"".join(v.to_bytes(2, "little") for v in values)
                target = Target("synthetic", (len(values),), (
                    Component("weight", "other", "missing", 0, len(raw), "BF16"),))
                artifact = NativeA("golomb_rice").encode(target, {"weight": raw}, {})
                model = json.loads(artifact.side_information)["components"][0]["streams"][0]["model"]
                k, bits = struct.unpack("<BQ", base64.b64decode(model["table_b64"]))
                costs = [sum(v // (2 ** candidate) + 1 + candidate for v in values)
                         for candidate in range(17)]
                self.assertEqual((k, bits), (costs.index(min(costs)), min(costs)))
                code = "".join("0" * (v // (2 ** k)) + "1" +
                               (format(v % (2 ** k), f"0{k}b") if k else "") for v in values)
                code += "0" * (-len(code) % 8)
                expected = bytes(int(code[i:i + 8], 2) for i in range(0, len(code), 8))
                self.assertEqual(artifact.payload, expected)
                self.assertEqual(NativeA("golomb_rice").decode(artifact), {"weight": raw})
                self.assertEqual(artifact.total_bytes, len(expected) + len(artifact.side_information))
                if values == list(range(6)):
                    self.assertEqual((k, bits, artifact.payload), (1, 18, bytes.fromhex("b4c8c0")))

    def test_rice_rfc_vector_and_invalid_models(self):
        self.assertNotIn("golomb_rice_fp8_bf16_exp", builtin_registry().ids())
        with self.assertRaises(ValueError):
            NativeA("golomb_rice", True)
        # RFC 9639 9.2.7: unsigned value 38 with k=3 is 00001110.
        raw = bytes([38])
        target = Target("synthetic", (1,), (Component("weight", "other", "missing", 0, 1, "F8_E4M3"),))
        codec = NativeA("golomb_rice")
        artifact = codec.encode(target, {"weight": raw}, {})
        header = json.loads(artifact.side_information)
        stream = header["components"][0]["streams"][0]
        stream["bytes"] = 1
        stream["model"].update(table_b64=base64.b64encode(struct.pack("<BQ", 3, 8)).decode(),
                               rice_parameter=3, payload_bits=8)
        self.assertEqual(codec.decode(EncodedArtifact(b"\x0e", json.dumps(header).encode())), {"weight": raw})
        for model in (struct.pack("<BQ", 17, 8), struct.pack("<BQ", 3, 0),
                      struct.pack("<BQ", 3, 8) + b"extra", b"\x03"):
            with self.subTest(model=model):
                stream["model"]["table_b64"] = base64.b64encode(model).decode()
                with self.assertRaises(ValueError):
                    codec.decode(EncodedArtifact(b"\x0e", json.dumps(header).encode()))
        # Nonzero trailing padding must also be rejected.
        raw = bytes(17)
        target = Target("synthetic", (17,), (Component("weight", "other", "missing", 0, 17, "F8_E8M0"),))
        artifact = codec.encode(target, {"weight": raw}, {})
        with self.assertRaises(ValueError):
            codec.decode(EncodedArtifact(artifact.payload[:-1] + bytes([artifact.payload[-1] | 1]),
                                         artifact.side_information))

    def test_separate_process_and_corrupt_artifacts(self):
        for technique in ("huffman", "arithmetic", "rans", "golomb_rice", "ppm", "ctw", "paq", "cabac", "cavlc", "neural_entropy"):
            target = Target("synthetic", (128,), (Component("weight", "other", "missing", 0, 256, "BF16"),))
            split = technique in ("huffman", "arithmetic", "rans")
            codec = NativeA(technique, split)
            source = {"weight": bytes(range(256))}
            artifact = codec.encode(target, source, {})
            script = """import json,sys
from fp4_redundancy.methods.native import NativeA
from fp4_redundancy.types import EncodedArtifact
r=json.load(sys.stdin)
a=EncodedArtifact(bytes.fromhex(r['payload']),bytes.fromhex(r['side']))
print(json.dumps({k:v.hex() for k,v in NativeA(r['technique'],r['split']).decode(a).items()}))
"""
            run = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                                 input=json.dumps({"technique": technique, "split": split, "payload": artifact.payload.hex(),
                                                   "side": artifact.side_information.hex()}),
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"weight": source["weight"].hex()})
            for payload in (artifact.payload[:-1], artifact.payload + b"extra"):
                with self.assertRaises(ValueError):
                    codec.decode(EncodedArtifact(payload, artifact.side_information))
            damaged = bytes([artifact.payload[0] ^ 255]) + artifact.payload[1:]
            with self.assertRaises(ValueError):
                codec.decode(EncodedArtifact(damaged, artifact.side_information))

    def test_parallel_native_run_saves_states_and_reports_nonapplicable_variants(self):
        fixture = test_checkpoint.CheckpointTests()
        fixture.setUp()
        try:
            output = fixture.root / "native_run"
            summary = run_experiment(fixture.snapshot, output, workers=2, chunk_bytes=2050)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["processed_fraction"], 1)
            for method_id, result in summary["methods"].items():
                if method_id == "vq_codeword":
                    self.assertEqual(result["statuses"]["inapplicable"], 1)
                    self.assertEqual(result["status_original_bytes"]["inapplicable"], 7*9*2)
                    self.assertEqual(result["verified_coverage_fraction"], 1-126/summary["inventory_original_bytes"])
                elif "_exp" in method_id:
                    self.assertGreater(result["statuses"]["inapplicable"], 0)
                    self.assertLess(result["verified_coverage_fraction"], 1)
                else:
                    self.assertEqual(set(result["statuses"]), {"verified"})
                    self.assertEqual(result["verified_coverage_fraction"], 1)
            run = json.loads((output / "run.json").read_text())["configuration"]
            self.assertEqual(run["backend"], "native")
            self.assertEqual(run["native_build"]["library_sha256"], build_metadata()["library_sha256"])
            database = sqlite3.connect(output / "results.sqlite3")
            try:
                for row, blob in database.execute("SELECT result_json,state_zlib FROM chunks"):
                    record, state = json.loads(row), json.loads(zlib.decompress(blob))
                    for result in record["results"]:
                        if result["status"] == "verified":
                            self.assertIn(result["method_id"], state)
                            header = state[result["method_id"]]["side_information"]
                            self.assertEqual(header["backend"], builtin_registry().create(result["method_id"]).backend)
                            if result["method_id"].startswith("golomb_rice"):
                                for component in header["components"]:
                                    for stream in component["streams"]:
                                        model = stream["model"]
                                        self.assertEqual(struct.unpack("<BQ", base64.b64decode(model["table_b64"])),
                                                         (model["rice_parameter"], model["payload_bits"]))
                        else:
                            self.assertIsNone(result["total_bytes"])
                            self.assertIsNone(result["saving_fraction"])
            finally:
                database.close()
        finally:
            fixture.tearDown()

    def test_stop_marker_and_resume(self):
        fixture = test_checkpoint.CheckpointTests()
        fixture.setUp()
        try:
            output = fixture.root / "stoppable"
            write = experiment._atomic_json

            def request_stop(path, value):
                write(path, value)
                if path.name == "progress.json" and value["status"] == "running":
                    (output / "STOP").touch()

            with patch.object(experiment, "_atomic_json", side_effect=request_stop):
                stopped = run_experiment(fixture.snapshot, output, workers=2, methods=["rans"])
            self.assertEqual(stopped["status"], "stopped")
            self.assertEqual(stopped["completed_chunks"], 0)
            resumed = run_experiment(fixture.snapshot, output, workers=2, methods=["rans"], resume=True)
            self.assertEqual(resumed["status"], "completed")
            self.assertFalse((output / "STOP").exists())
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
