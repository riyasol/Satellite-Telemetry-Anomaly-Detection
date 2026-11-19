"""P1/P2/P3 observation-mask and complete-window eligibility helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_mask_to_observation_mask(
    score_df,
    n_observations,
    input_window,
    horizon,
    excluded_column="excluded_from_drift",
):
    required = {"forecast_start_index", excluded_column}
    missing = required - set(score_df.columns)
    if missing:
        raise ValueError(f"score dataframe missing columns: {sorted(missing)}")

    n_observations = int(n_observations)
    input_window = int(input_window)
    horizon = int(horizon)
    if n_observations <= 0 or input_window <= 0 or horizon <= 0:
        raise ValueError("observation count, input_window and horizon must be positive")

    mask = np.zeros(n_observations, dtype=bool)
    excluded_rows = score_df.loc[
        score_df[excluded_column].astype(bool), "forecast_start_index"
    ].to_numpy(dtype=np.int64)

    for start in excluded_rows:
        left = max(0, int(start) + input_window)
        right = min(n_observations, left + horizon)
        if right > left:
            mask[left:right] = True
    return mask


def eligible_window_starts(
    observation_exclusion_mask,
    split_start,
    split_end,
    input_window,
    horizon,
    stride=1,
):
    """Return starts and eligibility by complete input-plus-target span."""
    excluded = np.asarray(observation_exclusion_mask, dtype=bool)
    split_start = int(split_start)
    split_end = int(split_end)
    input_window = int(input_window)
    horizon = int(horizon)
    stride = int(stride)

    if excluded.ndim != 1:
        raise ValueError("observation_exclusion_mask must be one-dimensional")
    if not 0 <= split_start < split_end <= len(excluded):
        raise ValueError("invalid mask or split bounds")
    if input_window <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("window sizes and stride must be positive")

    total_span = input_window + horizon
    last_exclusive = split_end - total_span + 1
    if last_exclusive <= split_start:
        raise ValueError("split too short for configured input and horizon")

    starts = np.arange(split_start, last_exclusive, stride, dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(excluded.astype(np.int64))))
    excluded_counts = prefix[starts + total_span] - prefix[starts]
    return starts, excluded_counts == 0


def summarize_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return [
        {
            "start_observation": int(start),
            "end_observation_inclusive": int(end),
            "length_observations": int(end - start + 1),
        }
        for start, end in zip(starts, ends)
    ]
