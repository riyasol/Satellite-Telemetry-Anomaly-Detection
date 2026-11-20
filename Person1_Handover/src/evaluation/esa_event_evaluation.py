"""Approval-gated event evaluation for ESA B1 score files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path


def extract_events(sequence: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True intervals as half-open scored-window indices."""
    values = np.asarray(sequence, dtype=bool)
    if values.ndim != 1:
        raise ValueError("event sequence must be one-dimensional")
    padded = np.r_[False, values, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _event_distance(
    true_event: tuple[int, int],
    predicted_event: tuple[int, int],
) -> int:
    true_start, true_end = true_event
    predicted_start, predicted_end = predicted_event
    if predicted_end <= true_start:
        return true_start - predicted_end + 1
    if true_end <= predicted_start:
        return predicted_start - true_end + 1
    return 0


def compute_esa_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid: np.ndarray,
    tolerance_steps: int,
    matching_rule: str,
    undefined_metric_policy: str | None = None,
) -> dict[str, Any]:
    """Calculate event counts and one-to-one maximum-cardinality matches."""
    truth = np.asarray(y_true, dtype=bool)
    prediction = np.asarray(y_pred, dtype=bool)
    valid_rows = np.asarray(valid, dtype=bool)
    if truth.ndim != 1 or prediction.ndim != 1 or valid_rows.ndim != 1:
        raise ValueError("truth, prediction, and valid arrays must be one-dimensional")
    if not (len(truth) == len(prediction) == len(valid_rows)):
        raise ValueError("truth, prediction, and valid arrays must have equal length")
    if tolerance_steps < 0:
        raise ValueError("tolerance_steps must be non-negative")
    if matching_rule != "one_to_one_max_cardinality":
        raise ValueError(f"Unsupported event matching rule: {matching_rule}")
    if undefined_metric_policy not in (None, "zero"):
        raise ValueError("undefined_metric_policy must be null or 'zero'")

    true_events = extract_events(truth & valid_rows)
    predicted_events = extract_events(prediction & valid_rows)
    candidates: list[list[int]] = []
    for true_start, true_end in true_events:
        expanded_start = max(0, true_start - tolerance_steps)
        expanded_end = min(len(truth), true_end + tolerance_steps)
        candidates.append([
            pred_index
            for pred_index, (pred_start, pred_end) in enumerate(predicted_events)
            if pred_start < expanded_end and pred_end > expanded_start
        ])

    predicted_to_true: dict[int, int] = {}

    def assign(true_index: int, seen_predictions: set[int]) -> bool:
        for predicted_index in candidates[true_index]:
            if predicted_index in seen_predictions:
                continue
            seen_predictions.add(predicted_index)
            previous_true = predicted_to_true.get(predicted_index)
            if previous_true is None or assign(previous_true, seen_predictions):
                predicted_to_true[predicted_index] = true_index
                return True
        return False

    for true_index in range(len(true_events)):
        assign(true_index, set())

    true_to_predicted = {
        true_index: predicted_index
        for predicted_index, true_index in predicted_to_true.items()
    }
    matched_pairs = []
    for true_index, predicted_index in sorted(true_to_predicted.items()):
        true_event = true_events[true_index]
        predicted_event = predicted_events[predicted_index]
        matched_pairs.append({
            "true_event_index": true_index,
            "predicted_event_index": predicted_index,
            "true_start_window": true_event[0],
            "true_end_window_exclusive": true_event[1],
            "predicted_start_window": predicted_event[0],
            "predicted_end_window_exclusive": predicted_event[1],
            "distance_windows": _event_distance(true_event, predicted_event),
        })

    unmatched_true = [
        {"event_index": index, "start_window": start, "end_window_exclusive": end}
        for index, (start, end) in enumerate(true_events)
        if index not in true_to_predicted
    ]
    unmatched_predicted = [
        {"event_index": index, "start_window": start, "end_window_exclusive": end}
        for index, (start, end) in enumerate(predicted_events)
        if index not in predicted_to_true
    ]

    tp = len(matched_pairs)
    fp = len(unmatched_predicted)
    fn = len(unmatched_true)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        if undefined_metric_policy == "zero":
            precision = 0.0 if precision is None else precision
            recall = 0.0 if recall is None else recall
            f05 = 0.0
        else:
            f05 = None
    else:
        denominator = 0.25 * precision + recall
        f05 = 1.25 * precision * recall / denominator if denominator else None

    return {
        "tp_events": tp,
        "fp_events": fp,
        "fn_events": fn,
        "total_true_events": len(true_events),
        "total_predicted_events": len(predicted_events),
        "precision_event": precision,
        "recall_event": recall,
        "f0.5_event": f05,
        "tolerance_steps": int(tolerance_steps),
        "tolerance_unit": "scored windows",
        "matching_rule": matching_rule,
        "matched_pairs": matched_pairs,
        "unmatched_true_events": unmatched_true,
        "unmatched_predicted_events": unmatched_predicted,
        "invalid_or_boundary_ignored_windows": int((~valid_rows).sum()),
        "undefined_metric_policy": undefined_metric_policy,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_score_arrays(
    scores_path: Path,
    metrics_path: Path,
    mission: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metadata.get("mission") != mission:
        raise ValueError(
            f"{metrics_path}: expected mission {mission}, "
            f"got {metadata.get('mission')!r}"
        )
    frame = pd.read_csv(scores_path)
    required = {
        "forecast_start_index",
        "predicted_anomaly",
        "true_target_window_anomaly",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{scores_path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{scores_path}: contains no scored windows")

    starts = frame["forecast_start_index"].to_numpy(dtype=np.int64)
    if len(starts) > 1 and np.any(np.diff(starts) <= 0):
        raise ValueError(f"{scores_path}: forecast starts are not strictly increasing")

    valid = (
        frame["valid"].astype(bool).to_numpy()
        if "valid" in frame
        else np.ones(len(frame), dtype=bool)
    )
    if "raw_mse" in frame:
        valid &= np.isfinite(frame["raw_mse"].to_numpy(dtype=float))

    if "evaluation_ignore" in frame:
        valid &= ~frame["evaluation_ignore"].astype(bool).to_numpy()
    elif "mask_reason" in frame:
        valid &= ~frame["mask_reason"].astype(str).str.contains(
            "boundary", case=False, regex=False
        ).to_numpy()
    elif int(metadata.get("test_windows_excluded_for_boundary_crossing", 0)) > 0:
        raise ValueError(
            f"{scores_path}: boundary-excluded windows are not identified"
        )

    for column in ("predicted_anomaly", "true_target_window_anomaly"):
        if not frame[column].isin([0, 1, False, True]).all():
            raise ValueError(f"{scores_path}: {column} must contain only 0/1")

    return (
        frame["true_target_window_anomaly"].astype(bool).to_numpy(),
        frame["predicted_anomaly"].astype(bool).to_numpy(),
        valid,
    )


def build_mission_report(cfg: dict, mission: str) -> dict[str, Any]:
    evaluation_cfg = cfg["evaluation"]
    configured_protocol = evaluation_cfg.get("event_evaluation", {})
    if not isinstance(configured_protocol, dict):
        raise ValueError("evaluation.event_evaluation must be a mapping")
    protocol_cfg = {
        "approval_status": "pending_team_approval",
        "approval_reference": "",
        "source_verification_status": "pending_raw_source_verification",
        "source_verification_reference": "",
        "matching_rule": "one_to_one_max_cardinality",
        "matching_rule_is_provisional": True,
        "undefined_metric_policy": None,
    }
    protocol_cfg.update(configured_protocol)
    output_root = resolve_path(cfg["output"]["results_dir"])
    mission_dir = output_root / mission.lower()
    scores_path = mission_dir / "b1_scores.csv"
    metrics_path = mission_dir / "b1_metrics.json"

    report: dict[str, Any] = {
        "mission": mission,
        "status": "pending",
        "metrics": None,
        "protocol": {
            "tolerance_steps": int(evaluation_cfg.get("event_tolerance_steps", 2)),
            "tolerance_unit": "scored windows",
            "tolerance_is_provisional": bool(
                evaluation_cfg.get("event_tolerance_is_provisional", True)
            ),
            "matching_rule": protocol_cfg["matching_rule"],
            "matching_rule_is_provisional": bool(
                protocol_cfg["matching_rule_is_provisional"]
            ),
            "undefined_metric_policy": protocol_cfg["undefined_metric_policy"],
        },
        "approval": {
            "status": protocol_cfg["approval_status"],
            "reference": protocol_cfg["approval_reference"] or None,
            "source_verification_status":
                protocol_cfg["source_verification_status"],
            "source_verification_reference":
                protocol_cfg["source_verification_reference"] or None,
        },
        "input_artifacts": {
            "scores_path": str(scores_path),
            "scores_sha256": _sha256(scores_path) if scores_path.is_file() else None,
            "metrics_path": str(metrics_path),
            "metrics_sha256":
                _sha256(metrics_path) if metrics_path.is_file() else None,
        },
        "pending_reasons": [],
    }

    if protocol_cfg["approval_status"] != "approved" or not protocol_cfg[
        "approval_reference"
    ]:
        report["pending_reasons"].append("event protocol guide/team approval missing")
    if evaluation_cfg.get("event_tolerance_is_provisional", True):
        report["pending_reasons"].append("event tolerance is still provisional")
    if protocol_cfg["matching_rule_is_provisional"]:
        report["pending_reasons"].append("event matching rule is still provisional")
    if protocol_cfg["source_verification_status"] != "verified" or not protocol_cfg[
        "source_verification_reference"
    ]:
        report["pending_reasons"].append("raw-source verification missing")
    if not scores_path.is_file():
        report["pending_reasons"].append("mission B1 score file missing")
    if not metrics_path.is_file():
        report["pending_reasons"].append("mission B1 metrics sidecar missing")

    if report["pending_reasons"]:
        return report

    truth, prediction, valid = _read_score_arrays(
        scores_path,
        metrics_path,
        mission,
    )
    report["metrics"] = compute_esa_event_metrics(
        truth,
        prediction,
        valid,
        tolerance_steps=report["protocol"]["tolerance_steps"],
        matching_rule=protocol_cfg["matching_rule"],
        undefined_metric_policy=protocol_cfg["undefined_metric_policy"],
    )
    report["status"] = "final"
    report["pending_reasons"] = []
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


def main() -> None:
    cfg = load_config()
    evaluation_cfg = cfg["evaluation"]
    protocol_cfg = evaluation_cfg.get("event_evaluation", {})
    if not isinstance(protocol_cfg, dict):
        raise ValueError("evaluation.event_evaluation must be a mapping")
    output_dir = resolve_path(
        protocol_cfg.get("output_dir", "results/event_evaluation")
    )
    for mission in cfg["data"].get("missions", ["Mission1"]):
        report = build_mission_report(cfg, mission)
        output_path = output_dir / f"{mission.lower()}_b1_event_metrics.json"
        write_report(output_path, report)
        print(f"{mission}: {report['status']} -> {output_path}")
        if report["pending_reasons"]:
            for reason in report["pending_reasons"]:
                print(f"  pending: {reason}")


if __name__ == "__main__":
    main()
