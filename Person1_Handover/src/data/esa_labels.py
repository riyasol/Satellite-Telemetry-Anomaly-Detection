"""Build exact ESA-ADB annotations and separate margin-expanded masks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_point_labels(
    labels_csv,
    channels,
    timestamps_ns,
    pre_margin_seconds=0,
    post_margin_seconds=0,
):
    if pre_margin_seconds < 0 or post_margin_seconds < 0:
        raise ValueError("Margins must be non-negative")

    df = pd.read_csv(labels_csv)
    required = {"Channel", "StartTime", "EndTime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels file missing columns: {missing}")

    df = df[df["Channel"].astype(str).isin([str(channel) for channel in channels])].copy()
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True, errors="coerce")
    df["EndTime"] = pd.to_datetime(df["EndTime"], utc=True, errors="coerce")
    if df[["StartTime", "EndTime"]].isna().any().any():
        raise ValueError("Invalid annotation timestamps in labels.csv")
    if (df["EndTime"] < df["StartTime"]).any():
        raise ValueError("Found anomaly interval with EndTime before StartTime")

    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    exact = np.zeros(len(timestamps_ns), dtype=np.uint8)
    excluded = np.zeros(len(timestamps_ns), dtype=np.uint8)
    annotated_points_by_channel = {}

    pre_ns = int(pre_margin_seconds * 1e9)
    post_ns = int(post_margin_seconds * 1e9)

    for channel in channels:
        channel_subset = df[df["Channel"].astype(str) == str(channel)]
        channel_count = 0
        for row in channel_subset.itertuples(index=False):
            exact_left = int(row.StartTime.value)
            exact_right = int(row.EndTime.value)
            left = int(np.searchsorted(timestamps_ns, exact_left, side="left"))
            right = int(np.searchsorted(timestamps_ns, exact_right, side="right"))
            if right > left:
                exact[left:right] = 1
                channel_count += right - left

            excluded_left = int(np.searchsorted(timestamps_ns, exact_left - pre_ns, side="left"))
            excluded_right = int(np.searchsorted(timestamps_ns, exact_right + post_ns, side="right"))
            excluded[excluded_left:excluded_right] = 1

        annotated_points_by_channel[str(channel)] = int(channel_count)

    return exact, excluded, {
        "selected_channel_annotation_rows": int(len(df)),
        "ground_truth_point_count_union": int(exact.sum()),
        "exclusion_point_count_union": int(excluded.sum()),
        "pre_margin_seconds": int(pre_margin_seconds),
        "post_margin_seconds": int(post_margin_seconds),
        "annotated_points_by_channel": annotated_points_by_channel,
        "label_semantics":
            "exact interval ground truth; exclusion mask expands intervals by configured margins",
    }
