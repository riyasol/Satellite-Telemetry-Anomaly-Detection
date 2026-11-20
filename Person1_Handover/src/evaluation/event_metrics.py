"""Event-wise anomaly detection evaluation metrics.

Calculates event-wise F0.5 (primary metric), precision, recall, and false alarm rate,
consistent with Kuhn et al. (2026) satellite housekeeping telemetry benchmark standards.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np


def extract_events(binary_sequence: np.ndarray) -> List[Tuple[int, int]]:
    """Extract contiguous [start, end) index intervals where binary_sequence is True."""
    arr = np.asarray(binary_sequence, dtype=bool)
    if not np.any(arr):
        return []

    diff = np.diff(arr.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if arr[0]:
        starts = np.r_[0, starts]
    if arr[-1]:
        ends = np.r_[ends, len(arr)]

    return list(zip(starts.tolist(), ends.tolist()))


def _maximum_event_matches(
    true_events: List[Tuple[int, int]],
    pred_events: List[Tuple[int, int]],
    tolerance_steps: int,
    sequence_length: int,
) -> int:
    """Match predicted and true events one-to-one, maximizing detected events."""
    adjacency = []
    for true_start, true_end in true_events:
        expanded_start = max(0, true_start - tolerance_steps)
        expanded_end = min(sequence_length, true_end + tolerance_steps)
        adjacency.append([
            pred_index
            for pred_index, (pred_start, pred_end) in enumerate(pred_events)
            if pred_end > expanded_start and pred_start < expanded_end
        ])

    true_match = [-1] * len(true_events)
    pred_match = [-1] * len(pred_events)
    match_count = 0
    for root in range(len(true_events)):
        queue = [root]
        visited_true = {root}
        visited_pred = set()
        parent_pred = {}
        free_pred = -1
        cursor = 0
        while cursor < len(queue) and free_pred < 0:
            true_index = queue[cursor]
            cursor += 1
            for pred_index in adjacency[true_index]:
                if pred_index in visited_pred:
                    continue
                visited_pred.add(pred_index)
                parent_pred[pred_index] = true_index
                matched_true = pred_match[pred_index]
                if matched_true < 0:
                    free_pred = pred_index
                    break
                if matched_true not in visited_true:
                    visited_true.add(matched_true)
                    queue.append(matched_true)

        if free_pred >= 0:
            pred_index = free_pred
            while pred_index >= 0:
                true_index = parent_pred[pred_index]
                previous_pred = true_match[true_index]
                true_match[true_index] = pred_index
                pred_match[pred_index] = true_index
                pred_index = previous_pred
            match_count += 1

    return match_count


def compute_event_metrics(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    tolerance_steps: int = 2,
    valid_mask: np.ndarray = None,
) -> Dict[str, Any]:
    """Compute event-wise evaluation metrics.

    Events are matched one-to-one when their intervals overlap within the
    configured tolerance.
    """
    y_true = np.asarray(y_true_binary, dtype=bool)
    y_pred = np.asarray(y_pred_binary, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted event sequences must have the same shape.")
    if tolerance_steps < 0:
        raise ValueError("tolerance_steps must be non-negative.")

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != y_true.shape:
            raise ValueError("valid_mask must match the event sequence shape.")
        padded = np.r_[False, valid, False].astype(np.int8)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        segments = [
            compute_event_metrics(
                y_true[start:end],
                y_pred[start:end],
                tolerance_steps=tolerance_steps,
            )
            for start, end in zip(starts, ends)
        ]
        tp = sum(segment["tp_events"] for segment in segments)
        fp = sum(segment["fp_events"] for segment in segments)
        fn = sum(segment["fn_events"] for segment in segments)
        true_count = sum(segment["total_true_events"] for segment in segments)
        pred_count = sum(
            len(extract_events(y_pred[start:end]))
            for start, end in zip(starts, ends)
        )
        if true_count == 0 and pred_count == 0:
            return {
                "tp_events": 0,
                "fp_events": 0,
                "fn_events": 0,
                "total_true_events": 0,
                "total_pred_events": 0,
                "precision": None,
                "recall": None,
                "f0.5_event": None,
                "false_alarm_rate": 0.0,
                "evaluated_windows": int(valid.sum()),
                "excluded_windows": int((~valid).sum()),
            }
        if true_count == 0:
            return {
                "tp_events": 0,
                "fp_events": pred_count,
                "fn_events": 0,
                "total_true_events": 0,
                "total_pred_events": pred_count,
                "precision": None,
                "recall": None,
                "f0.5_event": None,
                "false_alarm_rate": float(pred_count / max(1, int(valid.sum()))),
                "evaluated_windows": int(valid.sum()),
                "excluded_windows": int((~valid).sum()),
            }
        precision = tp / (tp + fp) if tp + fp else (1.0 if not true_count else 0.0)
        recall = tp / (tp + fn) if tp + fn else (1.0 if not pred_count else 0.0)
        beta2 = 0.25
        f05 = (
            (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
            if precision + recall else 0.0
        )
        return {
            "tp_events": tp,
            "fp_events": fp,
            "fn_events": fn,
            "total_true_events": true_count,
            "total_pred_events": pred_count,
            "precision": float(precision),
            "recall": float(recall),
            "f0.5_event": float(f05),
            "false_alarm_rate": float(fp / max(1, int(valid.sum()))),
            "evaluated_windows": int(valid.sum()),
            "excluded_windows": int((~valid).sum()),
        }

    true_events = extract_events(y_true)
    pred_events = extract_events(y_pred)

    if not true_events and not pred_events:
        return {
            "tp_events": 0, "fp_events": 0, "fn_events": 0, "total_true_events": 0,
            "total_pred_events": 0,
            "precision": None, "recall": None, "f0.5_event": None, "false_alarm_rate": 0.0
        }

    if not true_events and pred_events:
        return {
            "tp_events": 0,
            "fp_events": len(pred_events),
            "fn_events": 0,
            "total_true_events": 0,
            "total_pred_events": len(pred_events),
            "precision": None,
            "recall": None,
            "f0.5_event": None,
            "false_alarm_rate": float(len(pred_events) / max(1, len(y_pred)))
        }

    tp_events = _maximum_event_matches(
        true_events, pred_events, tolerance_steps, len(y_pred)
    )
    fn_events = len(true_events) - tp_events
    fp_events = len(pred_events) - tp_events

    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
    recall = tp_events / (tp_events + fn_events) if (tp_events + fn_events) > 0 else 0.0

    beta2 = 0.5 ** 2 # 0.25
    f05 = (
        (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    far = float(fp_events / max(1, len(y_pred)))

    return {
        "tp_events": tp_events,
        "fp_events": fp_events,
        "fn_events": fn_events,
        "total_true_events": len(true_events),
        "total_pred_events": len(pred_events),
        "precision": float(precision),
        "recall": float(recall),
        "f0.5_event": float(f05),
        "false_alarm_rate": float(far),
    }


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute standard point-window classification metrics."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    tn = int(np.sum(~yt & ~yp))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = 0.25
    f05 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision), "recall": float(recall),
        "f0.5_point": float(f05), "false_alarm_rate": float(far)
    }

