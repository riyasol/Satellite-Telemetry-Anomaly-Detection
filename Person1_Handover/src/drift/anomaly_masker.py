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

