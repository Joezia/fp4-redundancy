"""Section 7.4: generic layers contain no FP4 words; plot.py imports no adapter or indicator."""
import unittest

from redundancy.tests.acceptance import check_isolation


class Isolation(unittest.TestCase):
    def test_isolation(self):
        r = check_isolation()
        self.assertEqual(r["hits"], [])
        self.assertEqual(r["plot_bad_imports"], [])


if __name__ == "__main__":
    unittest.main()
