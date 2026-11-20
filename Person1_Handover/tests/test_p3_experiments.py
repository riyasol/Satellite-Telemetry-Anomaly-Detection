import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.evaluation.benchmark import (
    SustainedElevationTracker,
    prepare_scenario_data,
    run_adaptive_simulation,
)
from src.continual.replay_buffer import ReplayBuffer
from src.evaluation.event_metrics import compute_event_metrics
from src.models.lstm_forecaster import LSTMForecaster
from src.scenarios.drift_generator import (
    generate_scenario,
    inject_anomaly_pulses,
)


class ScenarioTests(unittest.TestCase):
    def test_controlled_drift_is_reproducible_with_fixed_seed(self):
        values = np.zeros((2000, 3), dtype=np.float32)
        first, first_config = generate_scenario(values, 3, seed=19)
        second, second_config = generate_scenario(values, 3, seed=19)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_config.to_dict(), second_config.to_dict())

    def test_scenario_parameters_control_the_generated_drift(self):
        values = np.zeros((200, 3), dtype=np.float32)
        changed, scenario = generate_scenario(
            values,
            2,
            seed=19,
            scenario_parameters={
                "start_step_cap": 20,
                "start_fraction_denominator": 10,
                "duration_steps": 12,
                "affected_channels": [2],
                "magnitude": 4.0,
            },
        )

        self.assertEqual(scenario.start_step, 20)
        self.assertEqual(scenario.end_step, 32)
        self.assertEqual(scenario.affected_channels, [2])
        self.assertEqual(float(changed[20:32, 2].min()), 4.0)
        self.assertEqual(float(changed[:, :2].max()), 0.0)

    def test_anomaly_pulses_are_labelled_and_reproducible(self):
        values = np.zeros((100, 3), dtype=np.float32)
        changed, labels = inject_anomaly_pulses(
            values,
            drift_start_step=40,
            stride=5,
            channels=[0, 2],
            magnitude=9.0,
            pulse_length=2,
        )

        self.assertEqual(int(labels.sum()), 6)
        self.assertEqual(int(np.count_nonzero(changed[:, 1])), 0)
        self.assertGreater(float(changed[:, [0, 2]].max()), 0.0)

    def test_sustained_elevation_requires_consecutive_high_windows(self):
        tracker = SustainedElevationTracker(minimum_windows=3)

        self.assertFalse(tracker.update(2.0, 1.0))
        self.assertFalse(tracker.update(2.0, 1.0))
        self.assertTrue(tracker.update(2.0, 1.0))
        self.assertFalse(tracker.update(0.5, 1.0))
        self.assertFalse(tracker.update(2.0, 1.0))


class BenchmarkProtocolTests(unittest.TestCase):
    def test_b4_candidate_attempts_respect_recent_window_cooldown(self):
        windows_x = np.zeros((10, 2, 1), dtype=np.float32)
        windows_y = np.zeros((10, 1, 1), dtype=np.float32)
        scenario_data = {
            "gate_x": windows_x[:2],
            "gate_y": windows_y[:2],
            "adaptation_validation_x": windows_x[:2],
            "adaptation_validation_y": windows_y[:2],
            "adaptation_x": windows_x,
            "adaptation_y": windows_y,
            "normal_validation_raw_scores": np.array(
                [0.1, 0.2, 0.3, 0.4], dtype=np.float32
            ),
            "threshold": 1.0,
            "adaptation_drift_start_window": 0,
            "pre_margin_windows": 0,
            "post_margin_windows": 0,
            "evaluation_x": windows_x[:2],
            "evaluation_y": windows_y[:2],
        }
        cfg = {
            "seed": 3,
            "model": {"batch_size": 2},
            "anomaly": {"ema_alpha": 1.0},
            "drift": {
                "adwin_delta": 0.002,
                "persistent_elevation_min_windows": 1,
                "reference_window_size": 4,
                "recent_window_size": 4,
                "persistence_window_k": 25,
                "persistence_min": 1,
                "severity_weights": [0.4, 0.3, 0.3],
                "theta_low": 0.35,
                "theta_high": 0.65,
                "degradation_threshold_delta_q": 0.05,
                "mask_dominance_threshold": 0.75,
                "min_clean_reference_samples": 1,
                "min_clean_recent_samples": 1,
            },
            "continual": {
                "validation_tolerance": 0.02,
                "replay_capacity": 10,
                "replay_ratio": 0.3,
                "finetune_epochs": 1,
                "finetune_lr": 0.0003,
                "strong_epochs": 1,
                "strong_lr": 0.001,
            },
            "experiments": {
                "minimum_candidate_windows": 2,
                "minimum_distribution_samples": 2,
            },
            "output": {"model_dir": "unused"},
        }
        model = LSTMForecaster(
            input_size=1,
            hidden_size=2,
            num_layers=1,
            dropout=0.0,
            horizon=1,
        )

        class NoAlarmDetector:
            def __init__(self, delta):
                pass

            def update(self, score, stream_index=None):
                return False

        class AlwaysSevere:
            def __init__(self, weights):
                pass

            def calibrate_from_normal_validation(self, scores):
                pass

            def compute_severity(self, measures, step=None):
                return {
                    "severity": 1.0,
                    "normalized": {"ks": 1.0, "wasserstein": 1.0, "js": 1.0},
                }

        class AlwaysDegraded:
            def __init__(self, degradation_threshold_delta_q, reference_mse):
                pass

            def evaluate_residual_degradation(self, residuals, step=None):
                return {
                    "relative_degradation": 1.0,
                    "is_significant_degradation": True,
                }

        class RejectingGate:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate_candidate(self, *args, **kwargs):
                return SimpleNamespace(accepted=False)

            def get_summary(self):
                return {
                    "accepted_count": 0,
                    "rejected_count": 3,
                    "rejection_counts": {
                        "retention": 0,
                        "adaptation_validation": 3,
                        "other": 0,
                    },
                }

        class RejectingTrainer:
            attempts = 0

            def __init__(self, *args, **kwargs):
                pass

            def create_candidate(self, current_model, *args, **kwargs):
                type(self).attempts += 1
                return current_model, {}, Path("unused.pt")

        def fixed_scores(model, x_arr, y_arr, device, batch_size):
            return np.full(len(x_arr), 2.0, dtype=np.float32), 0.0

        with (
            patch("src.evaluation.benchmark.ADWIN", NoAlarmDetector),
            patch("src.evaluation.benchmark.SeverityCalculator", AlwaysSevere),
            patch("src.evaluation.benchmark.DegradationTracker", AlwaysDegraded),
            patch("src.evaluation.benchmark.ValidationGate", RejectingGate),
            patch("src.evaluation.benchmark.CandidateTrainer", RejectingTrainer),
            patch("src.evaluation.benchmark.score_model_on_windows", fixed_scores),
            patch(
                "src.evaluation.benchmark._evaluate_frozen_model",
                return_value=({}, np.zeros(2, dtype=bool), 0.0),
            ),
        ):
            run_adaptive_simulation(
                "B4",
                3,
                scenario_data,
                cfg,
                torch.device("cpu"),
                model,
                ReplayBuffer(capacity=10, seed=3),
            )

        self.assertEqual(RejectingTrainer.attempts, 3)

    def test_scenario_adapts_on_validation_and_evaluates_on_test(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=(4000, 6)).astype(np.float32)
        labels = np.zeros(4000, dtype=np.uint8)
        timestamps = np.arange(4000, dtype=np.int64) * 30_000_000_000
        splits = {
            "values": values,
            "labels": labels,
            "exclusion_mask": np.zeros(4000, dtype=np.uint8),
            "evaluation_ignore_mask": np.zeros(4000, dtype=np.uint8),
            "timestamps_ns": timestamps,
            "train_bounds": np.array([0, 2000]),
            "validation_bounds": np.array([2000, 3400]),
            "test_bounds": np.array([3400, 4000]),
        }
        cfg = {
            "seed": 11,
            "model": {
                "input_window": 2,
                "forecast_horizon": 1,
                "window_stride": 1,
                "batch_size": 32,
            },
            "anomaly": {"ema_alpha": 0.2, "threshold_quantile": 0.995},
            "continual": {
                "adaptation_fraction_of_validation": 0.5,
                "b2_periodic_interval": 2,
            },
            "drift": {
                "pre_anomaly_margin_seconds": 30,
                "post_anomaly_margin_seconds": 30,
            },
        }
        model = LSTMForecaster(
            input_size=6,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            horizon=1,
        )

        prepared = prepare_scenario_data(
            scenario_id=3,
            cfg=cfg,
            clean_splits=splits,
            base_model=model,
            device=torch.device("cpu"),
        )

        adaptation_bounds = prepared["scenario"]["adaptation_bounds"]
        adaptation_validation_bounds = prepared["scenario"][
            "adaptation_validation_bounds"
        ]
        self.assertGreaterEqual(adaptation_bounds[0], 2000)
        self.assertEqual(adaptation_bounds[1], adaptation_validation_bounds[0])
        self.assertLessEqual(adaptation_validation_bounds[1], 3400)
        self.assertGreater(len(prepared["adaptation_validation_x"]), 0)
        self.assertTrue(np.all(
            prepared["adaptation_starts"] < adaptation_bounds[1] - adaptation_bounds[0]
        ))
        self.assertTrue(np.all(prepared["evaluation_starts"] >= 3400))
        self.assertEqual(len(prepared["evaluation_true"]), len(prepared["evaluation_valid"]))

    def test_event_metrics_exclude_invalid_segments_without_joining_events(self):
        truth = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
        pred = np.array([0, 1, 0, 0, 0, 0], dtype=bool)
        valid = np.array([1, 1, 0, 0, 1, 1], dtype=bool)

        result = compute_event_metrics(
            truth, pred, tolerance_steps=0, valid_mask=valid
        )

        self.assertEqual(result["tp_events"], 1)
        self.assertEqual(result["fn_events"], 1)
        self.assertEqual(result["evaluated_windows"], 4)

    def test_one_predicted_alarm_cannot_match_multiple_true_events(self):
        truth = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
        pred = np.ones(6, dtype=bool)

        result = compute_event_metrics(truth, pred, tolerance_steps=0)

        self.assertEqual(result["tp_events"], 1)
        self.assertEqual(result["fn_events"], 1)
        self.assertEqual(result["fp_events"], 0)
        self.assertEqual(result["recall"], 0.5)

    def test_duplicate_predicted_alarms_cannot_match_one_true_event_twice(self):
        truth = np.array([0, 1, 0], dtype=bool)
        pred = np.array([1, 0, 1], dtype=bool)

        result = compute_event_metrics(truth, pred, tolerance_steps=1)

        self.assertEqual(result["tp_events"], 1)
        self.assertEqual(result["fn_events"], 0)
        self.assertEqual(result["fp_events"], 1)


if __name__ == "__main__":
    unittest.main()
