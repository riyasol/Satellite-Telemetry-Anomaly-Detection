"""Anomaly masking and surrounding-window exclusion.

Masks anomalous windows and configurable surrounding margins so fault-related
residual spikes do not masquerade as normal distribution drift or enter model
update data.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np


class AnomalyMasker:
    """Manages anomaly masks and surrounding exclusion margins."""

    def __init__(
        self,
        pre_margin_seconds: float = 60.0,
        post_margin_seconds: float = 60.0,
        nominal_step_seconds: float = 30.0,
    ):
        self.pre_margin_seconds = float(pre_margin_seconds)
        self.post_margin_seconds = float(post_margin_seconds)
        self.nominal_step_seconds = float(nominal_step_seconds)

        # Margin in number of window strides / steps
        self.pre_margin_steps = int(np.ceil(self.pre_margin_seconds / max(1.0, self.nominal_step_seconds)))
        self.post_margin_steps = int(np.ceil(self.post_margin_seconds / max(1.0, self.nominal_step_seconds)))

        self.masked_intervals_log: List[Dict[str, Any]] = []

    def build_exclusion_mask(
        self, anomaly_flags: np.ndarray, timestamps_ns: np.ndarray = None
    ) -> np.ndarray:
        """Create a boolean exclusion mask (True = excluded/masked, False = eligible).

        Expands each True anomaly flag by pre_margin and post_margin.
        """
        n = len(anomaly_flags)
        exclusion = np.zeros(n, dtype=bool)
        raw_flags = np.asarray(anomaly_flags, dtype=bool)

        if not np.any(raw_flags):
            return exclusion

        anomaly_indices = np.where(raw_flags)[0]
        # Group contiguous anomaly indices into intervals
        intervals = []
        if len(anomaly_indices) > 0:
            start = anomaly_indices[0]
            prev = start
            for idx in anomaly_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    intervals.append((start, prev))
                    start = idx
                    prev = idx
            intervals.append((start, prev))

        for start_idx, end_idx in intervals:
            expanded_start = max(0, start_idx - self.pre_margin_steps)
            expanded_end = min(n, end_idx + self.post_margin_steps + 1)
            exclusion[expanded_start:expanded_end] = True

            log_entry = {
                "anomaly_start_index": int(start_idx),
                "anomaly_end_index": int(end_idx),
                "expanded_start_index": int(expanded_start),
                "expanded_end_index": int(expanded_end),
                "pre_margin_steps": self.pre_margin_steps,
                "post_margin_steps": self.post_margin_steps,
                "reason": "Anomaly spike and surrounding safety margin exclusion",
            }
            if timestamps_ns is not None:
                log_entry["start_utc"] = str(np.datetime64(timestamps_ns[start_idx], "ns"))
                log_entry["end_utc"] = str(np.datetime64(timestamps_ns[end_idx], "ns"))

            self.masked_intervals_log.append(log_entry)

        return exclusion

    def filter_eligible_samples(
        self, samples: np.ndarray, exclusion_mask: np.ndarray
    ) -> np.ndarray:
        """Filter out masked/excluded rows, returning only eligible normal samples."""
        samples_arr = np.asarray(samples)
        mask = np.asarray(exclusion_mask, dtype=bool)
        if len(mask) != len(samples_arr):
            raise ValueError(f"Mask length {len(mask)} != sample length {len(samples_arr)}")
        return samples_arr[~mask]

    def get_mask_log(self) -> List[Dict[str, Any]]:
        return self.masked_intervals_log

    @staticmethod
    def mask_dominance(predicted_flags: np.ndarray) -> float:
        """Return the fraction of predicted-anomaly entries in an aligned window."""
        flags = np.asarray(predicted_flags, dtype=bool).reshape(-1)
        if flags.size == 0:
            return 0.0
        finite = np.isfinite(flags.astype(np.float64))
        if not np.all(finite):
            return 0.0
        return float(np.mean(flags))

    @staticmethod
    def clean_residual_window(
        residuals: np.ndarray,
        predicted_flags: np.ndarray,
    ) -> np.ndarray:
        """Remove predicted anomalous residuals by aligned position."""
        values = np.asarray(residuals, dtype=np.float64).reshape(-1)
        flags = np.asarray(predicted_flags, dtype=bool).reshape(-1)
        if values.size != flags.size:
            raise ValueError(
                f"Residual and predicted-mask lengths differ: {values.size} != {flags.size}"
            )
        return values[np.isfinite(values) & ~flags]

    @staticmethod
    def split_adwin_window(
        residuals: np.ndarray,
        predicted_flags: np.ndarray,
        cut_point: int,
        min_clean_samples: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Split by local ADWIN cut position, then remove predicted anomalies."""
        values = np.asarray(residuals, dtype=np.float64).reshape(-1)
        flags = np.asarray(predicted_flags, dtype=bool).reshape(-1)
        if values.size != flags.size:
            raise ValueError(
                f"ADWIN residual and predicted-mask lengths differ: {values.size} != {flags.size}"
            )
        if cut_point <= 0 or cut_point >= values.size:
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                {
                    "suppression_reason": "invalid_cut_position",
                    "reference_clean_count": 0,
                    "recent_clean_count": 0,
                },
            )

        reference = AnomalyMasker.clean_residual_window(
            values[:cut_point], flags[:cut_point]
        )
        recent = AnomalyMasker.clean_residual_window(
            values[cut_point:], flags[cut_point:]
        )
        reference_count = int(reference.size)
        recent_count = int(recent.size)
        reason = "none"
        if reference_count < int(min_clean_samples):
            reason = "insufficient_clean_reference"
        elif recent_count < int(min_clean_samples):
            reason = "insufficient_clean_recent"

        return reference, recent, {
            "suppression_reason": reason,
            "reference_clean_count": reference_count,
            "recent_clean_count": recent_count,
        }


def confirm_adwin_alarm(
    residual_window: np.ndarray,
    window_indices: np.ndarray,
    predicted_mask_by_stream_index: np.ndarray,
    cut_point: int,
    mask_dominance_threshold: float,
    min_clean_reference_samples: int,
    min_clean_recent_samples: int,
) -> Dict[str, Any]:
    """Confirm a raw ADWIN alarm using anomaly-mask evidence and clean samples."""
    mask_dominance_threshold = float(mask_dominance_threshold)
    min_clean_reference_samples = int(min_clean_reference_samples)
    min_clean_recent_samples = int(min_clean_recent_samples)
    if not 0.0 <= mask_dominance_threshold <= 1.0:
        raise ValueError("mask_dominance_threshold must be between 0 and 1")
    if min_clean_reference_samples <= 0 or min_clean_recent_samples <= 0:
        raise ValueError("minimum clean sample counts must be positive")

    residuals = np.asarray(residual_window, dtype=np.float64).reshape(-1)
    indices = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    mask = np.asarray(predicted_mask_by_stream_index, dtype=bool).reshape(-1)

    if residuals.size != indices.size:
        raise ValueError(
            f"ADWIN residual/window-index lengths differ: {residuals.size} != {indices.size}"
        )
    if indices.size == 0:
        return {
            "raw_adwin_alarm": True,
            "drift_confirmed": False,
            "suppression_reason": "insufficient_window",
            "anomaly_mask_fraction": 0.0,
            "reference_clean_count": 0,
            "recent_clean_count": 0,
            "cut_point": int(cut_point),
        }

    valid_indices = (indices >= 0) & (indices < mask.size)
    if not np.all(valid_indices):
        raise ValueError("ADWIN window index is outside the predicted-mask stream")

    aligned_flags = mask[indices]
    mask_fraction = float(np.mean(aligned_flags)) if aligned_flags.size else 0.0
    if cut_point <= 0 or cut_point >= residuals.size:
        return {
            "raw_adwin_alarm": True,
            "drift_confirmed": False,
            "suppression_reason": "invalid_cut_position",
            "anomaly_mask_fraction": mask_fraction,
            "reference_clean_count": 0,
            "recent_clean_count": 0,
            "cut_point": int(cut_point),
        }

    reference_flags = aligned_flags[:cut_point]
    recent_flags = aligned_flags[cut_point:]
    reference = residuals[:cut_point][
        np.isfinite(residuals[:cut_point]) & ~reference_flags
    ]
    recent = residuals[cut_point:][
        np.isfinite(residuals[cut_point:]) & ~recent_flags
    ]

    reference_count = int(reference.size)
    recent_count = int(recent.size)
    if mask_fraction > mask_dominance_threshold:
        reason = "anomaly_mask_dominance"
    elif reference_count < min_clean_reference_samples:
        reason = "insufficient_clean_reference"
    elif recent_count < min_clean_recent_samples:
        reason = "insufficient_clean_recent"
    else:
        reason = "none"

    return {
        "raw_adwin_alarm": True,
        "drift_confirmed": reason == "none",
        "suppression_reason": reason,
        "anomaly_mask_fraction": mask_fraction,
        "reference_clean_count": reference_count,
        "recent_clean_count": recent_count,
        "cut_point": int(cut_point),
        "reference": reference,
        "recent": recent,
        "reference": reference,
        "recent": recent,
    }

