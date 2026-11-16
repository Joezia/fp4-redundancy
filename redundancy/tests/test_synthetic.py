"""Section 7.1: verdicts on the synthetic pairs. Uses REDUNDANCY_RESULTS (a results directory with
synthetic panels), default experiments/007-figures/results."""
import os
import unittest
from pathlib import Path

from redundancy.tests.acceptance import ROOT, check_synthetic

RESULTS = Path(os.environ.get("REDUNDANCY_RESULTS", ROOT / "experiments/007-figures/results"))


@unittest.skipUnless((RESULTS / "manifest.json").exists(), "no results directory")
class Synthetic(unittest.TestCase):
    def test_verdicts(self):
        r = check_synthetic(RESULTS)
        self.assertTrue(r["pass"], [x for x in r["rows"] if not x["pass"]])


if __name__ == "__main__":
    unittest.main()
