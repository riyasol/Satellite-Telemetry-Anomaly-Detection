import numpy as np
from src.drift.degradation import DegradationTracker

def test_degradation_tracker_missing_f05():
    tracker = DegradationTracker(reference_f05=None, reference_mse=0.1)
    rec = tracker.evaluate_f05_degradation(current_f05=0.5, step=1)
    assert rec["reference_f05"] is None
    assert rec["is_significant_degradation"] is False
    # residual degradation should work
    residuals = np.array([0.1, 0.2, 0.15])
    rec2 = tracker.evaluate_residual_degradation(residuals, step=2)
    assert "relative_degradation" in rec2
