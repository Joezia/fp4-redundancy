"""Section 7.3: two temporary .npy arrays through the array_file adapter (run + plot)."""
import tempfile
import unittest

from redundancy.tests.acceptance import check_portable


class Portable(unittest.TestCase):
    def test_portable(self):
        with tempfile.TemporaryDirectory() as d:
            r = check_portable(d)
        self.assertTrue(r["datasets"]["uniform"]["pass"], r["datasets"]["uniform"])
        self.assertTrue(r["datasets"]["smooth"]["pass"], r["datasets"]["smooth"])


if __name__ == "__main__":
    unittest.main()
