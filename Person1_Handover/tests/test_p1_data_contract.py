import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.esa_labels import build_point_labels
from src.data.esa_loader import _timestamps_to_ns, load_channel_zip, mission_channels
from src.data.validate_processed import validate_mission
from src.drift.window_eligibility import (
    eligible_window_starts,
    score_mask_to_observation_mask,
)
from src.models.evaluate_b1 import operational_test_ema
from src.utils.config import load_config


class ESALabelsTests(unittest.TestCase):
    def test_datetime_index_conversion_always_returns_nanoseconds(self):
        timestamps = pd.date_range("2026-01-01", periods=2, freq="30s", tz="UTC")
        timestamps_ns = _timestamps_to_ns(timestamps)
        self.assertEqual(int(timestamps_ns[1] - timestamps_ns[0]), 30_000_000_000)

    def test_mission_channel_lists_are_stable(self):
        self.assertEqual(mission_channels("Mission1"), [
            f"channel_{i}" for i in range(41, 47)
        ])
        self.assertEqual(mission_channels("Mission2"), [
            f"channel_{i}" for i in range(18, 29)
        ])

    def test_channel_zip_loader_rejects_wrong_column_name(self):
        timestamps = pd.date_range("2026-01-01", periods=4, freq="30s", tz="UTC")
        frame = pd.DataFrame({"wrong_name": [1.0, 2.0, 3.0, 4.0]}, index=timestamps)
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "channel.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                buffer = BytesIO()
                frame.to_pickle(buffer)
                archive.writestr("channel.pkl", buffer.getvalue())
            with self.assertRaisesRegex(ValueError, "expected column"):
                load_channel_zip(zip_path, "channel_41")

    def test_exact_labels_and_expanded_mask_are_separate(self):
        timestamps = pd.date_range("2026-01-01", periods=20, freq="30s", tz="UTC")
        timestamps_ns = timestamps.to_numpy(
            dtype="datetime64[ns]"
        ).view(np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "labels.csv"
            pd.DataFrame({
                "Channel": ["channel_41"],
                "StartTime": ["2026-01-01T00:02:00Z"],
                "EndTime": ["2026-01-01T00:03:00Z"],
            }).to_csv(csv_path, index=False)

            exact, excluded, info = build_point_labels(
                csv_path,
                ["channel_41"],
                timestamps_ns,
                pre_margin_seconds=60,
                post_margin_seconds=60,
            )

        self.assertGreater(excluded.sum(), exact.sum())
        self.assertNotEqual(exact.sum(), excluded.sum())
        self.assertEqual(info["label_semantics"].split(";")[0], "exact interval ground truth")

    def test_invalid_annotation_interval_is_rejected(self):
        timestamps = pd.date_range("2026-01-01", periods=10, freq="30s", tz="UTC").asi8
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "labels.csv"
            pd.DataFrame({
                "Channel": ["channel_41"],
                "StartTime": ["2026-01-01T00:03:00Z"],
                "EndTime": ["2026-01-01T00:02:00Z"],
            }).to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "before StartTime"):
                build_point_labels(csv_path, ["channel_41"], timestamps)


class WindowEligibilityTests(unittest.TestCase):
    def test_complete_input_and_target_span_is_checked(self):
        excluded = np.zeros(100, dtype=bool)
        excluded[20:23] = True
        starts, eligible = eligible_window_starts(
            excluded, 0, 100, input_window=5, horizon=3, stride=1
        )
        bad_starts = starts[~eligible].tolist()
        self.assertEqual(bad_starts, list(range(13, 23)))

    def test_score_mask_maps_only_forecast_horizon(self):
        score_df = pd.DataFrame({
            "forecast_start_index": [0, 10, 20],
            "excluded_from_drift": [0, 1, 0],
        })
        mask = score_mask_to_observation_mask(
            score_df, 100, input_window=5, horizon=3
        )
        self.assertEqual(np.flatnonzero(mask).tolist(), [15, 16, 17])

    def test_invalid_window_config_is_rejected(self):
        with self.assertRaises(ValueError):
            eligible_window_starts(
                np.zeros(10, dtype=bool),
                0,
                10,
                input_window=0,
                horizon=3,
            )


class OperationalIsolationTests(unittest.TestCase):
    def test_operational_ema_is_causal_and_does_not_reset(self):
        actual = operational_test_ema(
            np.array([0.0, 10.0, 0.0, 0.0]), alpha=0.2
        )
        np.testing.assert_allclose(actual, [0.0, 2.0, 1.6, 1.28])

    def test_missing_processed_data_is_reported_as_blocked(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["data"]["esa_work_dir"] = str(Path(tmp) / "missing")
            with self.assertRaises(FileNotFoundError):
                validate_mission(cfg, "Mission1")


if __name__ == "__main__":
    unittest.main()
