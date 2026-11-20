import unittest

import numpy as np
import torch

from src.evaluation.benchmark import prepare_scenario_data
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


class BenchmarkProtocolTests(unittest.TestCase):
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
        self.assertGreaterEqual(adaptation_bounds[0], 2000)
        self.assertLessEqual(adaptation_bounds[1], 3400)
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


if __name__ == "__main__":
    unittest.main()
