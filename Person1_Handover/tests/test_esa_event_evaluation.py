import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.esa_event_evaluation import (
    build_mission_report,
    compute_esa_event_metrics,
    extract_events,
)


class ESAEventEvaluationTests(unittest.TestCase):
    def test_extract_events_uses_half_open_intervals(self):
        events = extract_events(np.array([0, 1, 1, 0, 1], dtype=bool))
        self.assertEqual(events, [(1, 3), (4, 5)])

    def test_matching_is_one_to_one_when_prediction_overlaps_two_events(self):
        truth = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
        prediction = np.array([0, 1, 1, 1, 1, 0], dtype=bool)

        result = compute_esa_event_metrics(
            truth,
            prediction,
            np.ones(len(truth), dtype=bool),
            tolerance_steps=0,
            matching_rule="one_to_one_max_cardinality",
        )

        self.assertEqual(result["tp_events"], 1)
        self.assertEqual(result["fp_events"], 0)
        self.assertEqual(result["fn_events"], 1)
        self.assertEqual(len(result["matched_pairs"]), 1)

    def test_boundary_tolerance_is_in_scored_window_steps(self):
        truth = np.array([0, 0, 0, 1, 0, 0], dtype=bool)
        prediction = np.array([0, 1, 0, 0, 0, 0], dtype=bool)

        within_tolerance = compute_esa_event_metrics(
            truth,
            prediction,
            np.ones(len(truth), dtype=bool),
            tolerance_steps=2,
            matching_rule="one_to_one_max_cardinality",
        )
        outside_tolerance = compute_esa_event_metrics(
            truth,
            prediction,
            np.ones(len(truth), dtype=bool),
            tolerance_steps=1,
            matching_rule="one_to_one_max_cardinality",
        )

        self.assertEqual(within_tolerance["tp_events"], 1)
        self.assertEqual(outside_tolerance["tp_events"], 0)

    def test_invalid_windows_break_events_and_undefined_metrics_are_null(self):
        truth = np.array([1, 0, 1], dtype=bool)
        prediction = np.zeros(3, dtype=bool)
        valid = np.array([True, False, True])

        result = compute_esa_event_metrics(
            truth,
            prediction,
            valid,
            tolerance_steps=0,
            matching_rule="one_to_one_max_cardinality",
        )

        self.assertEqual(result["total_true_events"], 2)
        self.assertEqual(result["invalid_or_boundary_ignored_windows"], 1)
        self.assertIsNone(result["precision_event"])
        self.assertIsNone(result["f0.5_event"])

    def test_event_metrics_remain_pending_without_approval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            mission_dir = results_dir / "mission1"
            mission_dir.mkdir()
            (mission_dir / "b1_metrics.json").write_text(
                json.dumps({"mission": "Mission1"}),
                encoding="utf-8",
            )
            (mission_dir / "b1_scores.csv").write_text(
                "forecast_start_index,predicted_anomaly,"
                "true_target_window_anomaly\n0,1,1\n",
                encoding="utf-8",
            )
            cfg = {
                "data": {"missions": ["Mission1"]},
                "evaluation": {
                    "event_tolerance_steps": 2,
                    "event_tolerance_is_provisional": True,
                    "event_evaluation": {
                        "approval_status": "pending_team_approval",
                        "approval_reference": "",
                        "source_verification_status": "verified",
                        "source_verification_reference": "verified source report",
                        "matching_rule": "one_to_one_max_cardinality",
                        "matching_rule_is_provisional": True,
                        "undefined_metric_policy": None,
                    },
                },
                "output": {"results_dir": str(results_dir)},
            }

            report = build_mission_report(cfg, "Mission1")

        self.assertEqual(report["status"], "pending")
        self.assertIsNone(report["metrics"])
        self.assertIn(
            "event tolerance is still provisional",
            report["pending_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
