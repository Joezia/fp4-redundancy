"""Small synthetic checks; no model data or compression findings."""

import copy
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

from fp4_redundancy.catalog import CATALOG, Registry, builtin_registry, catalog_output, plan
from fp4_redundancy.cli import main
from fp4_redundancy.types import Component, EncodedArtifact, MethodSpec, Support, Target
from fp4_redundancy.manifest import load_manifest
from fp4_redundancy.runner import evaluate_target

ROOT = Path(__file__).resolve().parents[1]


class CopyFixture:
    """Test-only byte storage, never a production compression method."""

    spec = MethodSpec("test_copy", "test fixture", "G", "native", "test fixture")

    def check_support(self, target, parameters):
        return Support(not parameters, "Test fixture has no parameters")

    def encode(self, target, components, parameters):
        layout = [[key, len(value)] for key, value in components.items()]
        return EncodedArtifact(b"".join(components.values()), json.dumps(layout).encode())

    def decode(self, artifact):
        output, offset = {}, 0
        for key, length in json.loads(artifact.side_information):
            output[key] = artifact.payload[offset:offset + length]
            offset += length
        return output


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.manifest = json.loads((ROOT / "examples/model-manifest.json").read_text())

    def tearDown(self):
        self.directory.cleanup()

    def load(self, raw, check_files=False):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(raw))
        return load_manifest(path, check_files=check_files)

    def target(self):
        path = self.root / "components.bin"
        path.write_bytes(bytes([1, 143, 0, 255, 0, 128]))
        return Target("tiny", (8,), (
            Component("weight", "packed_fp4", str(path), 0, 4, "synthetic:packed"),
            Component("scale", "scale", str(path), 4, 2, "synthetic:scale")),
            {"layer": 0, "expert": 0}, {"access_unit": "unknown"})

    def evaluate(self, factory, parameters=None):
        registry = Registry()
        registry.register(factory)
        return evaluate_target(registry, factory.spec.id, self.target(), parameters)

    def test_manifest_rejects_invalid_or_overlapping_ranges(self):
        for offset, length in ((-1, 4), (0, 0)):
            raw = copy.deepcopy(self.manifest)
            raw["targets"][0]["components"][0].update(offset=offset, length=length)
            with self.subTest(offset=offset, length=length), self.assertRaises(ValueError):
                self.load(raw)
        raw = copy.deepcopy(self.manifest)
        duplicate = copy.deepcopy(raw["targets"][0])
        duplicate["id"] = "another"
        raw["targets"].append(duplicate)
        with self.assertRaises(ValueError):
            self.load(raw)

    def test_file_checks(self):
        with self.assertRaises(ValueError):
            self.load(self.manifest, check_files=True)
        target = self.target()
        self.manifest["targets"][0]["logical_shape"] = list(target.logical_shape)
        self.manifest["targets"][0]["components"] = [asdict(c) for c in target.components]
        self.assertEqual(self.load(self.manifest, check_files=True).targets[0].original_bytes, 6)
        self.manifest["targets"][0]["components"][-1]["length"] = 3
        with self.assertRaises(ValueError):
            self.load(self.manifest, check_files=True)

    def test_cli_and_plan(self):
        model = self.load(self.manifest)
        output = plan(model, ["A", "B", "C", "D"])
        self.assertFalse(output["measurements"])
        self.assertEqual({job["method_id"] for job in output["jobs"]}, {e.id for e in CATALOG})
        self.assertTrue(all(job["execution_status"] == "not_run" for job in output["jobs"]))
        self.assertEqual(plan(model, list("EFG"))["jobs"], [])
        for args in (["methods"], ["validate", str(ROOT / "examples/model-manifest.json")],
                     ["plan", str(ROOT / "examples/model-manifest.json")]):
            with self.subTest(args=args), redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(args), 0)
            data = json.loads(stream.getvalue())
            if args[0] == "methods":
                self.assertEqual({m["id"] for m in data["methods"] if m["implemented"]},
                                 set(builtin_registry().ids()))
            elif args[0] == "validate":
                self.assertTrue(data["valid"])
            else:
                self.assertEqual(data, {**output, "source_file_ranges_checked": False,
                                        "storage_format_parsed": False})

    def test_registration_needs_no_research_approval(self):
        class Fixture(CopyFixture):
            # Only test registration under a listed id; this fixture does not implement ADPCM.
            spec = MethodSpec("adpcm_like_fp4", "test fixture", "B", "adapted", "test fixture")
        registry = Registry()
        registry.register(Fixture)
        entry = next(e for e in catalog_output(registry)["methods"] if e["id"] == Fixture.spec.id)
        self.assertTrue(entry["implemented"])
        with self.assertRaises(ValueError):
            registry.register(Fixture)

    def test_roundtrip_counts_all_components_and_metadata(self):
        result = self.evaluate(CopyFixture)
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["bitwise_verified"])
        self.assertEqual(result["original_bytes"], 6)
        self.assertGreater(result["side_information_bytes"], 0)
        self.assertEqual(result["total_bytes"], result["payload_bytes"] + result["side_information_bytes"])
        self.assertLess(result["saving_fraction"], 0)
        self.assertEqual(set(result["source_sha256"]), {"weight", "scale"})
        self.assertEqual(result["structure"], {"layer": 0, "expert": 0})
        self.assertEqual(result["inference"]["access_unit"], "unknown")

    def test_unavailable_methods_have_no_savings(self):
        missing = evaluate_target(Registry(), "missing", self.target())
        self.assertEqual(missing["status"], "not_implemented")
        self.assertIsNone(missing["saving_fraction"])
        unsupported = self.evaluate(CopyFixture, {"unsupported": True})
        self.assertEqual(unsupported["status"], "inapplicable")
        self.assertIsNone(unsupported["saving_fraction"])

    def test_bad_decode_and_exceptions_have_no_savings(self):
        class Corrupt(CopyFixture):
            def decode(self, artifact):
                output = super().decode(artifact)
                output["weight"] = b"wrong"
                return output
        class Missing(CopyFixture):
            def decode(self, artifact):
                return {"weight": bytes([1, 143, 0, 255])}
        class Broken(CopyFixture):
            def encode(self, target, components, parameters):
                raise RuntimeError("test error")
        for fixture, status in ((Corrupt, "verification_failed"), (Missing, "verification_failed"),
                                (Broken, "failed")):
            with self.subTest(fixture=fixture.__name__):
                result = self.evaluate(fixture)
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["total_bytes"])
                self.assertIsNone(result["saving_fraction"])

    def test_decode_uses_new_instance(self):
        class Stateful(CopyFixture):
            def encode(self, target, components, parameters):
                self.cached = dict(components)
                return EncodedArtifact(b"")
            def decode(self, artifact):
                return self.cached
        result = self.evaluate(Stateful)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["saving_fraction"])


if __name__ == "__main__":
    unittest.main()
