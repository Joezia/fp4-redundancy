"""Tiny sharded safetensors fixtures; no model payloads or model experiments."""

from contextlib import redirect_stdout
import copy
import io
import itertools
import json
from math import prod
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from fp4_redundancy.catalog import builtin_registry, iter_chunk_plan
from fp4_redundancy.checkpoint import DTYPE_BYTES, REVISION, load_checkpoint, read_header
from fp4_redundancy.chunks import DEFAULT_CHUNK_BYTES, iter_target_chunks
from fp4_redundancy.cli import main
from fp4_redundancy.manifest import load_manifest, save_manifest
from fp4_redundancy.runner import evaluate_target, evaluate_target_chunks
from fp4_redundancy.types import Component, Target


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.blobs = self.root / "blobs"
        self.blobs.mkdir()
        self.config = {"model_type": "deepseek_v41", "quantization_config": {
            "quant_method": "fp8", "expert_dtype": "fp4", "weight_block_size": [32, 32], "scale_fmt": "ue8m0"}}
        self.specs = {
            "layers.0.ffn.experts.1.w1.weight": ("I8", [5, 32]),
            "layers.0.ffn.experts.1.w1.scale": ("F8_E8M0", [5, 2]),
            "layers.0.attn.wq_a.weight": ("F8_E4M3", [65, 64]),
            "layers.0.attn.wq_a.scale": ("F8_E8M0", [3, 2]),
            "layers.1.engram.embed.weight": ("F8_E4M3", [9, 64]),
            "layers.1.engram.embed.scale": ("F8_E8M0", [9, 2]),
            "mtp.0.ffn.experts.0.w2.weight": ("I8", [4, 16]),
            "mtp.0.ffn.experts.0.w2.scale": ("F8_E8M0", [4, 1]),
            "vision.blocks.0.attn.wo.weight": ("BF16", [7, 9]),
            "layers.0.hc_attn_scale": ("F32", [17]),
            "scalar": ("F32", [])}
        self.write_snapshot()

    def tearDown(self):
        self.temp.cleanup()

    def write_snapshot(self):
        self.raw, self.data_starts, self.shard_headers = {}, {}, {}
        groups = [{}, {}]
        weight_map = {}
        for index, (name, (dtype, shape)) in enumerate(self.specs.items()):
            size = prod(shape) * DTYPE_BYTES.get(dtype, 1)
            seed = sum(map(ord, name))
            data = bytes((seed + i) % 256 for i in range(size))
            self.raw[name] = data
            groups[index % 2][name] = (dtype, shape, data)
        for index, group in enumerate(groups):
            shard = f"model-{index + 1:05}-of-00002.safetensors"
            header, payload = {}, bytearray()
            for name, (dtype, shape, data) in group.items():
                header[name] = {"dtype": dtype, "shape": shape,
                                "data_offsets": [len(payload), len(payload) + len(data)]}
                payload.extend(data)
                weight_map[name] = shard
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            blob = self.blobs / f"blob{index}"
            blob.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            link = self.snapshot / shard
            if not link.exists():
                link.symlink_to(blob)
            self.data_starts[link] = 8 + len(encoded)
            self.shard_headers[link] = header
        (self.snapshot / "config.json").write_text(json.dumps(self.config))
        (self.snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    def test_inventory_reads_only_headers_and_pairs_cross_shard_scales(self):
        original_open = Path.open
        boundaries = self.data_starts

        class Guard:
            def __init__(self, stream, limit):
                self.stream, self.limit = stream, limit
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def read(self, size):
                if size < 0 or self.stream.tell() + size > self.limit:
                    raise AssertionError("Adapter tried to read a tensor payload")
                return self.stream.read(size)

        def guarded_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return Guard(stream, boundaries[path]) if path in boundaries else stream

        with patch.object(Path, "open", guarded_open):
            model = load_checkpoint(self.snapshot)
        self.assertEqual(model.metadata["tensor_count"], len(self.specs))
        self.assertEqual(model.metadata["tensor_bytes"], sum(map(len, self.raw.values())))
        self.assertEqual(sum(t.original_bytes for t in model.targets), model.metadata["tensor_bytes"])
        self.assertFalse(model.metadata["payload_hashed"])
        expert = next(t for t in model.targets if t.id == "layers.0.ffn.experts.1.w1.weight")
        self.assertEqual(expert.logical_shape, (5, 64))
        self.assertNotEqual(expert.components[0].path, expert.components[1].path)
        self.assertEqual(expert.components[0].storage_format, "fp4_packed_low_first")
        self.assertEqual(expert.structure["expert"], 1)
        self.assertEqual(expert.inference["access_unit"], "unknown")
        layouts = {t.id: t.structure["scale_block_shape"] for t in model.targets}
        self.assertEqual(layouts["layers.0.attn.wq_a.weight"], [32, 32])
        self.assertEqual(layouts["layers.1.engram.embed.weight"], [1, 32])
        self.assertEqual(next(t for t in model.targets if t.id.startswith("mtp.")).structure["domain"], "mtp")

    def test_manifest_and_cli_roundtrip(self):
        path = self.root / "manifest.json"
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["checkpoint", str(self.snapshot), "--output", str(path)]), 0)
        summary = json.loads(output.getvalue())
        self.assertFalse(summary["measurements"])
        self.assertTrue(summary["inventory_verified"])
        model = load_manifest(path, check_files=True)
        self.assertEqual(model.metadata["tensor_count"], len(self.specs))
        self.assertEqual(next(t for t in model.targets if t.id == "scalar").logical_shape, ())
        with self.assertRaises(FileExistsError):
            save_manifest(model, path)
        name = "layers.1.engram.embed.weight"
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["plan", str(path), "--categories", "A", "--chunk-bytes", "132",
                                   "--target", name]), 0)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(records), 5)
        self.assertTrue(all(r["coverage"] == "subset" and not r["measurements"] for r in records))
        self.assertTrue(all(r["execution_status"] == "not_run" for r in records))

    def test_chunks_cover_each_component_once_and_restore_all_methods(self):
        model = load_checkpoint(self.snapshot)
        for target in model.targets:
            block = target.structure["scale_block_shape"]
            limit = 2050 if block == [32, 32] else 132 if ".engram.embed." in target.id else 68 if block else 20
            chunks = list(iter_target_chunks(target, limit))
            cursors = {c.id: 0 for c in target.components}
            for chunk in chunks:
                self.assertLessEqual(chunk.original_bytes, limit)
                self.assertEqual(chunk.structure["layer"], target.structure["layer"])
                for part in chunk.components:
                    parent = next(c for c in target.components if c.id == part.id)
                    record = chunk.structure["chunk"]["ranges"][part.id]
                    self.assertEqual(record["offset"], cursors[part.id])
                    self.assertEqual(part.offset, parent.offset + record["offset"])
                    cursors[part.id] += part.length
            self.assertEqual(cursors, {c.id: c.length for c in target.components})
            if block == [32, 32]:
                self.assertEqual([c.structure["chunk"]["rows"] for c in chunks], [[0, 32], [32, 64], [64, 65]])
            for method_id in builtin_registry().ids():
                if not builtin_registry().create(method_id).check_support(chunks[0], {}).applicable:
                    continue
                restored = {c.id: bytearray() for c in target.components}
                for chunk in chunks:
                    raw = {}
                    for part in chunk.components:
                        name = target.structure["tensor_names"][part.id]
                        offset = chunk.structure["chunk"]["ranges"][part.id]["offset"]
                        raw[part.id] = self.raw[name][offset:offset + part.length]
                    codec = builtin_registry().create(method_id)
                    artifact = codec.encode(chunk, raw, {})
                    header = json.loads(artifact.side_information)
                    self.assertEqual(header["chunk"], chunk.structure["chunk"])
                    self.assertEqual(header["tensor_names"], target.structure["tensor_names"])
                    decoded = builtin_registry().create(method_id).decode(artifact)
                    for key, data in decoded.items():
                        restored[key].extend(data)
                self.assertEqual(restored, {key: self.raw[name] for key, name in target.structure["tensor_names"].items()})

    def test_lazy_chunk_evaluation_and_byte_limit(self):
        model = load_checkpoint(self.snapshot)
        target = next(t for t in model.targets if t.id == "layers.0.attn.wq_a.weight")
        for method_id in builtin_registry().ids():
            records = list(evaluate_target_chunks(builtin_registry(), method_id, target, chunk_bytes=2050))
            self.assertTrue(all(r["status"] == "verified" for r in records))
            self.assertEqual(sum(r["original_bytes"] for r in records), target.original_bytes)
            self.assertTrue(all(r["total_bytes"] == r["payload_bytes"] + r["side_information_bytes"] for r in records))
        huge = Target("huge", (10**12,), (Component("weight", "other", "does-not-exist", 0, 10**12, "BF16"),))
        chunks = iter_target_chunks(huge)
        first_two = list(itertools.islice(chunks, 2))
        self.assertEqual([c.components[0].offset for c in first_two], [0, DEFAULT_CHUNK_BYTES])
        rejected = evaluate_target(builtin_registry(), "huffman", huge)
        self.assertEqual(rejected["status"], "inapplicable")
        self.assertIn("limit", rejected["reason"])
        # A changed/truncated shard becomes a failure with no measured savings.
        source = Path(target.components[0].path)
        source.write_bytes(b"short")
        result = next(evaluate_target_chunks(builtin_registry(), "huffman", target, chunk_bytes=2050))
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["saving_fraction"])

    def test_layout_and_index_errors_are_rejected(self):
        original = copy.deepcopy(self.specs)
        changes = [
            ("layers.0.ffn.experts.1.w1.scale", ("F8_E8M0", [5, 1])),
            ("layers.1.engram.embed.scale", ("F8_E8M0", [1, 2])),
            ("layers.0.attn.wq_a.scale", ("F8_E8M0", [65, 2])),
            ("scalar", ("F16", [])),
            ("unrecognized.weight", ("I8", [1, 16])),
            ("orphan.scale", ("F8_E8M0", [1]))]
        for name, spec in changes:
            self.specs = {**original, name: spec}
            self.write_snapshot()
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_checkpoint(self.snapshot)
        self.specs = original
        self.write_snapshot()
        with self.assertRaises(ValueError):
            load_checkpoint(self.snapshot, revision="0" * 40)
        index = self.snapshot / "model.safetensors.index.json"
        raw = json.loads(index.read_text())
        raw["weight_map"]["absent.weight"] = next(iter(raw["weight_map"].values()))
        index.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            load_checkpoint(self.snapshot)

    def test_bad_headers_and_impossible_chunk_sizes(self):
        path = self.root / "invalid.safetensors"
        for data in (b"short", struct.pack("<Q", 10**12) + b"{}"):
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                read_header(path)
        header = b'{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"x":{}}'
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"ab")
        with self.assertRaises(ValueError):
            read_header(path)
        model = load_checkpoint(self.snapshot)
        target = next(t for t in model.targets if t.structure["scale_block_shape"] == [32, 32])
        for limit in (0, -1, 2049):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                next(iter_target_chunks(target, limit))
        with self.assertRaises(ValueError):
            next(iter_chunk_plan(model, ["A"], 4096, "missing"))
        self.assertEqual(list(iter_chunk_plan(model, list("EFG"), 4096)), [])


if __name__ == "__main__":
    unittest.main()
