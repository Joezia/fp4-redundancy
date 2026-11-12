"""Resumable execution on tiny synthetic checkpoint files only."""

from contextlib import redirect_stdout
import csv
import io
import json
import os
import sqlite3
import unittest
import zlib

import test_checkpoint
from fp4_redundancy.catalog import builtin_registry
from fp4_redundancy.cli import main
from fp4_redundancy.checkpoint import load_checkpoint
from fp4_redundancy.experiment import run_experiment, select_chunks


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_checkpoint.CheckpointTests()
        self.fixture.setUp()
        self.output = self.fixture.root / "run"

    def tearDown(self):
        self.fixture.tearDown()

    def test_full_synthetic_run_saves_states_and_exact_accounting(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["run", str(self.fixture.snapshot), "--output", str(self.output),
                                   "--workers", "2", "--chunk-bytes", "2050"]), 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["processed_fraction"], 1)
        original = sum(map(len, self.fixture.raw.values()))
        self.assertEqual(summary["processed_original_bytes"], original)
        self.assertEqual(summary["chunk_bytes_limit"], 2050)
        self.assertEqual(set(summary["methods"]), set(builtin_registry().ids()))
        for method in summary["methods"].values():
            self.assertFalse(set(method["statuses"]) - {"verified", "inapplicable"})
            self.assertEqual(method["verified_original_bytes"] + method["status_original_bytes"].get("inapplicable", 0), original)
            direct = sum(g["direct_total_bytes"] for g in method["by_component_format_layout"].values())
            self.assertEqual(direct + method["shared_side_information_bytes"], method["total_bytes"])
            self.assertEqual(method["saving_fraction_on_verified_bytes"], 1 - method["total_bytes"] / method["verified_original_bytes"])
        database = sqlite3.connect(self.output / "results.sqlite3")
        try:
            rows = list(database.execute("SELECT result_json,state_zlib FROM chunks"))
            self.assertEqual(len(rows), summary["completed_chunks"])
            chunk_sizes = [json.loads(raw)["original_bytes"] for raw, _ in rows]
            self.assertEqual(summary["chunk_original_bytes_min"], min(chunk_sizes))
            self.assertEqual(summary["chunk_original_bytes_max"], max(chunk_sizes))
            for raw, state in rows:
                record = json.loads(raw)
                states = json.loads(zlib.decompress(state))
                self.assertEqual(set(states), {r["method_id"] for r in record["results"] if r["status"] == "verified"})
                for result in record["results"]:
                    if result["status"] == "inapplicable":
                        self.assertIsNone(result["saving_fraction"])
                        continue
                    self.assertTrue(result["bitwise_verified"])
                    saved = states[result["method_id"]]
                    self.assertEqual(len(saved["payload_sha256"]), 64)
                    self.assertNotIn("payload", saved)
                    self.assertIn("chunk", saved["side_information"])
                    self.assertTrue(all(c["streams"] for c in saved["side_information"]["components"]))
        finally:
            database.close()
        config = json.loads((self.output / "run.json").read_text())["configuration"]
        self.assertFalse(config["payload_persisted"])
        with (self.output / "sizes.csv").open(newline="") as stream:
            sizes = list(csv.DictReader(stream))
        self.assertEqual({row["method_id"] for row in sizes}, set(summary["methods"]))
        for row in sizes:
            method = summary["methods"][row["method_id"]]
            self.assertEqual(int(row["chunk_bytes_limit"]), 2050)
            self.assertEqual(int(row["chunk_original_bytes_min"]), min(chunk_sizes))
            self.assertEqual(int(row["chunk_original_bytes_max"]), max(chunk_sizes))
            for key in ("payload_bytes", "side_information_bytes", "total_bytes"):
                self.assertEqual(int(row[key]), method[key])
            self.assertEqual(int(row["payload_bytes"]) + int(row["side_information_bytes"]), int(row["total_bytes"]))

    def test_resume_skips_committed_chunks_and_rejects_parameter_drift(self):
        first = run_experiment(self.fixture.snapshot, self.output, workers=2, chunk_bytes=2050,
                               methods=["rans"])
        second = run_experiment(self.fixture.snapshot, self.output, workers=1, chunk_bytes=2050,
                                methods=["rans"], resume=True)
        self.assertEqual(first["completed_chunks"], second["completed_chunks"])
        self.assertEqual(first["methods"], second["methods"])
        with self.assertRaises(ValueError):
            run_experiment(self.fixture.snapshot, self.output, chunk_bytes=4096, methods=["rans"], resume=True)

    def test_subset_and_results_only_are_explicit_and_source_changes_rejected(self):
        summary = run_experiment(self.fixture.snapshot, self.output, chunk_bytes=2050,
                                 methods=["huffman"], max_chunks=1, keep_states=False)
        self.assertEqual(summary["status"], "completed_subset")
        self.assertLess(summary["processed_fraction"], 1)
        database = sqlite3.connect(self.output / "results.sqlite3")
        try:
            self.assertEqual(database.execute("SELECT COUNT(*) FROM chunks WHERE state_zlib IS NULL").fetchone()[0], 1)
        finally:
            database.close()
        source = next(self.fixture.snapshot.glob("*.safetensors"))
        before = source.stat()
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        with self.assertRaises(ValueError):
            run_experiment(self.fixture.snapshot, self.output, chunk_bytes=2050, methods=["huffman"],
                           max_chunks=1, keep_states=False, resume=True)

    def test_explicit_sample_selects_middle_tail_and_preserves_coverage_resume(self):
        model = load_checkpoint(self.fixture.snapshot)
        name = "layers.0.ffn.experts.1.w1.weight"
        selection = {"model_id": model.model_id, "revision": model.revision, "chunk_bytes": 68,
                     "chunks": [{"target_id": name, "index": 1}, {"target_id": name, "index": 2}]}
        chunks = select_chunks(model, 68, selection)
        self.assertEqual([c.structure["chunk"]["rows"] for c in chunks], [[2, 4], [4, 5]])
        plan = self.fixture.root / "selection.json"
        plan.write_text(json.dumps(selection))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["run", str(self.fixture.snapshot), "--output", str(self.output),
                                   "--chunk-bytes", "68", "--methods", "rans", "--selection", str(plan)]), 0)
        first = json.loads(output.getvalue())
        self.assertEqual(first["status"], "completed_subset")
        self.assertEqual(first["coverage"], "subset")
        self.assertEqual(first["selection_processed_fraction"], 1)
        self.assertLess(first["processed_fraction"], 1)
        self.assertEqual(first["processed_original_bytes"], 102)
        self.assertEqual(first["methods"]["rans"]["statuses"], {"verified": 2})
        with sqlite3.connect(self.output / "results.sqlite3") as db:
            records = [json.loads(r[0]) for r in db.execute("SELECT result_json FROM chunks")]
        self.assertEqual({r["chunk_id"] for r in records}, {c.id for c in chunks})
        for record in records:
            part = record["structure"]["chunk"]["ranges"]["weight"]
            import hashlib
            raw = self.fixture.raw[name][part["offset"]:part["offset"] + part["length"]]
            self.assertEqual(record["results"][0]["source_sha256"]["weight"], hashlib.sha256(raw).hexdigest())
        second = run_experiment(self.fixture.snapshot, self.output, chunk_bytes=68, methods=["rans"],
                                selection=selection, resume=True)
        self.assertEqual(first["methods"], second["methods"])
        with self.assertRaises(ValueError):
            run_experiment(self.fixture.snapshot, self.output, chunk_bytes=68, methods=["rans"],
                           selection={**selection, "chunks": selection["chunks"][:1]}, resume=True)
        for bad in (None, [], {**selection, "revision": "other"}, {**selection, "chunks": []},
                    {**selection, "chunks": selection["chunks"] * 2},
                    {**selection, "chunks": [{"target_id": "missing", "index": 0}]},
                    {**selection, "chunks": [{"target_id": name, "index": 100}]}):
            with self.assertRaises(ValueError):
                select_chunks(model, 68, bad)


if __name__ == "__main__":
    unittest.main()
