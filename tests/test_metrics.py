import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fraud_monitor.metrics import average_precision, choose_threshold, classification_metrics


class MetricsTest(unittest.TestCase):
    def test_perfect_ranking_has_unit_average_precision(self):
        y = np.array([0, 1, 0, 1])
        probability = np.array([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(average_precision(y, probability), 1.0)

    def test_confusion_matrix(self):
        result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.9, 0.2]), 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 1))

    def test_threshold_respects_target_recall(self):
        result = choose_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.9]), 1.0)
        self.assertEqual(result["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
