import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.continual.candidate_trainer import CandidateTrainer
from src.continual.replay_buffer import ReplayBuffer
from src.continual.validation_gate import ValidationGate
from src.models.lstm_forecaster import LSTMForecaster


class ConstantForecaster(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))

    def forward(self, inputs):
        return self.value.expand(inputs.shape[0], 1, inputs.shape[2])


class ReplayBufferTests(unittest.TestCase):
    def test_masked_samples_are_rejected_and_reservoir_is_bounded(self):
        buffer = ReplayBuffer(capacity=3, seed=7)
        x = np.zeros((2, 1), dtype=np.float32)
        y = np.zeros((1, 1), dtype=np.float32)

        self.assertFalse(buffer.add(x, y, is_anomaly_masked=True))
        self.assertEqual(buffer.total_added, 0)

        for value in range(20):
            self.assertTrue(
                buffer.add(
                    x + value,
                    y + value,
                    meta={"value": value},
                )
            )

        self.assertEqual(buffer.size(), 3)
        self.assertEqual(buffer.total_added, 20)
        self.assertEqual(len(buffer.metadata), 3)

    def test_seed_population_checks_complete_input_and_target_span(self):
        values = np.arange(10, dtype=np.float32).reshape(-1, 1)
        exclusion = np.zeros(10, dtype=np.uint8)
        exclusion[2] = 1
        buffer = ReplayBuffer(capacity=5, seed=1)

        buffer.populate_from_training_split(
            values,
            exclusion,
            train_start=0,
            train_end=10,
            input_window=3,
            forecast_horizon=2,
            stride=2,
        )

        self.assertEqual(
            [entry["start_idx"] for entry in buffer.metadata],
            [4],
        )

    def test_buffer_round_trip_preserves_metadata(self):
        buffer = ReplayBuffer(capacity=2, seed=5)
        buffer.add(
            np.ones((2, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            meta={"source": "test"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.npz"
            buffer.save(path)
            restored = ReplayBuffer(capacity=1)
            restored.load(path)

        self.assertEqual(restored.capacity, 2)
        self.assertEqual(restored.metadata, [{"source": "test"}])
        self.assertEqual(restored.total_added, 1)

    def test_invalid_capacity_and_nonfinite_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=0)

        buffer = ReplayBuffer(capacity=2)
        with self.assertRaisesRegex(ValueError, "finite"):
            buffer.add(
                np.array([[np.nan]], dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
            )


class ValidationGateTests(unittest.TestCase):
    def test_gate_accepts_safe_candidate_and_rejects_regression(self):
        val_x = np.zeros((3, 2, 1), dtype=np.float32)
        val_y = np.zeros((3, 1, 1), dtype=np.float32)
        gate = ValidationGate(val_x, val_y, torch.device("cpu"), tolerance=0.02)
        current = ConstantForecaster(1.0)

        accepted = gate.evaluate_candidate(
            current,
            ConstantForecaster(0.0),
            {"candidate_id": "better"},
        )
        rejected = gate.evaluate_candidate(
            current,
            ConstantForecaster(2.0),
            {"candidate_id": "worse"},
        )

        self.assertTrue(accepted.accepted)
        self.assertFalse(rejected.accepted)
        self.assertEqual(gate.get_summary()["accepted_count"], 1)
        self.assertEqual(gate.get_summary()["rejected_count"], 1)

    def test_gate_rejects_empty_or_nonfinite_validation_data(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            ValidationGate(
                np.empty((0, 2, 1), dtype=np.float32),
                np.empty((0, 1, 1), dtype=np.float32),
                torch.device("cpu"),
            )

        invalid_x = np.zeros((1, 2, 1), dtype=np.float32)
        invalid_x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            ValidationGate(
                invalid_x,
                np.zeros((1, 1, 1), dtype=np.float32),
                torch.device("cpu"),
            )


class CandidateTrainerTests(unittest.TestCase):
    def test_candidate_uses_unique_namespaced_checkpoint_and_configured_epochs(self):
        model = LSTMForecaster(
            input_size=1,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            horizon=1,
        )
        replay = ReplayBuffer(capacity=2, seed=2)
        replay.add(
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
        recent_x = np.random.default_rng(1).normal(size=(2, 3, 1)).astype(np.float32)
        recent_y = np.zeros((2, 1, 1), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            trainer = CandidateTrainer(
                model,
                torch.device("cpu"),
                replay,
                Path(tmp),
                candidate_namespace="mission1_s3_b4",
            )
            _, first, first_path = trainer.create_candidate(
                model,
                recent_x,
                recent_y,
                action_type="FINE_TUNE",
                epochs=1,
                lr=0.001,
                replay_ratio=0.3,
                batch_size=2,
            )
            _, second, second_path = trainer.create_candidate(
                model,
                recent_x,
                recent_y,
                action_type="FINE_TUNE",
                epochs=1,
                lr=0.001,
                replay_ratio=0.3,
                batch_size=2,
            )
            first_exists = first_path.exists()
            second_exists = second_path.exists()

        self.assertEqual(first["epochs"], 1)
        self.assertEqual(second["epochs"], 1)
        self.assertNotEqual(first_path, second_path)
        self.assertIn("mission1_s3_b4", first_path.name)
        self.assertTrue(first_exists)
        self.assertTrue(second_exists)

    def test_candidate_rejects_empty_training_data(self):
        model = LSTMForecaster(
            input_size=1,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            horizon=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            trainer = CandidateTrainer(
                model, torch.device("cpu"), ReplayBuffer(2), Path(tmp)
            )
            with self.assertRaisesRegex(ValueError, "non-zero"):
                trainer.create_candidate(
                    model,
                    np.empty((0, 3, 1), dtype=np.float32),
                    np.empty((0, 1, 1), dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main()
