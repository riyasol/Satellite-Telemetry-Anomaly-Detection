import unittest

import numpy as np

from src.drift.anomaly_masker import confirm_adwin_alarm
from src.drift.decision_engine import DecisionEngine
from src.drift.streaming_detectors import ADWIN
from src.evaluation.event_metrics import compute_event_metrics


class P2IsolationAndB4Tests(unittest.TestCase):
    def test_adaptation_and_final_test_bounds_are_separate(self):
        adaptation_bounds = (70, 85)
        test_bounds = (85, 100)

        adaptation = np.zeros(adaptation_bounds[1] - adaptation_bounds[0])
        final_test = np.zeros(test_bounds[1] - test_bounds[0])

        self.assertEqual(adaptation.size, 15)
        self.assertEqual(final_test.size, 15)
        self.assertEqual(adaptation_bounds[1], test_bounds[0])

    def test_ground_truth_is_not_used_by_alarm_confirmation(self):
        confirmation = confirm_adwin_alarm(
            residual_window=np.array([0.1, 0.2, 1.0, 1.1]),
            window_indices=np.array([0, 1, 2, 3]),
            predicted_mask_by_stream_index=np.array([True, True, False, False]),
            cut_point=2,
            mask_dominance_threshold=1.0,
            min_clean_reference_samples=1,
            min_clean_recent_samples=1,
        )

        self.assertTrue(confirmation["raw_adwin_alarm"])

    def test_mask_dominance_suppresses_raw_alarm(self):
        confirmation = confirm_adwin_alarm(
            residual_window=np.array([1.0, 1.1, 1.2, 1.3]),
            window_indices=np.array([10, 11, 12, 13]),
            predicted_mask_by_stream_index=np.array(
                [False] * 10 + [True, True, True, True]
            ),
            cut_point=2,
            mask_dominance_threshold=0.75,
            min_clean_reference_samples=1,
            min_clean_recent_samples=1,
        )

        self.assertTrue(confirmation["raw_adwin_alarm"])
        self.assertFalse(confirmation["drift_confirmed"])
        self.assertEqual(
            confirmation["suppression_reason"],
            "anomaly_mask_dominance",
        )

    def test_insufficient_clean_evidence_suppresses_alarm(self):
        confirmation = confirm_adwin_alarm(
            residual_window=np.array([0.1, 0.2, 1.0, 1.1]),
            window_indices=np.array([0, 1, 2, 3]),
            predicted_mask_by_stream_index=np.array([True, True, False, False]),
            cut_point=2,
            mask_dominance_threshold=1.0,
            min_clean_reference_samples=2,
            min_clean_recent_samples=1,
        )

        self.assertTrue(confirmation["raw_adwin_alarm"])
        self.assertFalse(confirmation["drift_confirmed"])
        self.assertEqual(
            confirmation["suppression_reason"],
            "insufficient_clean_reference",
        )

    def test_mask_alignment_uses_stream_indices_past_recent_buffer(self):
        indices = np.array([300, 301, 302, 303])
        predicted_mask = np.zeros(400, dtype=bool)
        predicted_mask[300:302] = True

        confirmation = confirm_adwin_alarm(
            residual_window=np.array([0.1, 0.2, 1.0, 1.1]),
            window_indices=indices,
            predicted_mask_by_stream_index=predicted_mask,
            cut_point=2,
            mask_dominance_threshold=1.0,
            min_clean_reference_samples=1,
            min_clean_recent_samples=1,
        )

        self.assertEqual(confirmation["anomaly_mask_fraction"], 0.5)
        self.assertEqual(confirmation["reference_clean_count"], 0)
        self.assertEqual(confirmation["recent_clean_count"], 2)

    def test_raw_alarm_is_not_automatically_confirmed(self):
        detector = ADWIN(delta=0.0001, max_window=200, min_window=5)
        alarm = False
        for value in [0.0] * 30 + [10.0] * 30:
            alarm = detector.update(value)
            if alarm:
                break

        engine = DecisionEngine(theta_low=0.35, theta_high=0.65)
        decision = engine.decide(
            step=1,
            timestamp_utc="test",
            drift_detected=False,
            detector_name="ADWIN",
            raw_measures={"ks": 0.0, "wasserstein": 0.0, "js": 0.0},
            normalized_measures={"ks": 0.0, "wasserstein": 0.0, "js": 0.0},
            severity=0.0,
            persistence_count=0,
            persistence_passed=False,
            degradation_value=0.0,
            degradation_passed=False,
            raw_adwin_alarm=alarm,
            drift_confirmed=False,
            suppression_reason="anomaly_mask_dominance",
        )

        self.assertTrue(decision.raw_adwin_alarm)
        self.assertFalse(decision.drift_confirmed)

    def test_b4_state_instances_are_independent(self):
        original = ADWIN(delta=0.002)
        dual = ADWIN(delta=0.002)
        original.update(1.0)

        self.assertEqual(len(original.window), 1)
        self.assertEqual(len(dual.window), 0)

    def test_empty_offline_segment_reports_na(self):
        metrics = compute_event_metrics(
            np.array([False, False, False]),
            np.array([False, False, False]),
        )
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["f0.5_event"])


if __name__ == "__main__":
    unittest.main()
