"""Section 7.5: the same config run twice gives byte-identical result files (array_file config)."""
import tempfile
import unittest
from pathlib import Path

from redundancy.tests.acceptance import compare_dirs, make_arrays, portable_config, run_cmd


class Reproducible(unittest.TestCase):
    def test_twice(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            files = make_arrays(d)
            cfg = portable_config(d, files)
            for out in ("a", "b"):
                code, log = run_cmd(["redundancy.run", "--config", str(cfg), "--out", str(d / out)])
                self.assertEqual(code, 0, log)
            r = compare_dirs(d / "a", d / "b")
        self.assertTrue(r["pass"], r)


if __name__ == "__main__":
    unittest.main()
