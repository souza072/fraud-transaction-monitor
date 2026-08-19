import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fraud_monitor.constants import risk_level


class RiskLevelTest(unittest.TestCase):
    def test_levels_are_relative_to_threshold(self):
        threshold = 0.01
        self.assertEqual(risk_level(0.01, threshold), "watch")
        self.assertEqual(risk_level(0.02, threshold), "moderate")
        self.assertEqual(risk_level(0.04, threshold), "high")
        self.assertEqual(risk_level(0.08, threshold), "critical")


if __name__ == "__main__":
    unittest.main()
