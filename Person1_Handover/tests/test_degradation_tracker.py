import unittest

import numpy as np
from src.drift.degradation import DegradationTracker


class DegradationTrackerTests(unittest.TestCase):
    def test_missing_f05_reference_keeps_residual_degradation_available(self):
        tracker = DegradationTracker(reference_f05=None, reference_mse=0.1)
        result = tracker.evaluate_f05_degradation(current_f05=0.5, step=1)
        self.assertIsNone(result["reference_f05"])
        self.assertFalse(result["is_significant_degradation"])

        residuals = np.array([0.1, 0.2, 0.15])
        residual_result = tracker.evaluate_residual_degradation(residuals, step=2)
        self.assertIn("relative_degradation", residual_result)


if __name__ == "__main__":
    unittest.main()
