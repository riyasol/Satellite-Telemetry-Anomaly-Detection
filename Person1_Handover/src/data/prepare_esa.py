"""Prepare ESA-ADB Mission 1 and Mission 2 blocks and chronological splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.esa_labels import build_point_labels
from src.data.esa_loader import load_mission_matrix
from src.utils.config import load_config, resolve_path


def prepare_one(cfg: dict, mission: str) -> dict:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    raw_root = resolve_path(data_cfg["esa_raw_root"])
    work = resolve_path(data_cfg["esa_work_dir"]) / mission.lower()
    work.mkdir(parents=True, exist_ok=True)

    info = load_mission_matrix(raw_root, mission, work)
    full_values = np.load(info["matrix_path"], mmap_mode="r")
    full_timestamps = np.load(info["timestamps_path"], mmap_mode="r")

    start_cfg = data_cfg.get("block_start_utc_by_mission", {}).get(
        mission, data_cfg.get("block_start_utc")
    )
    if start_cfg:
        start_timestamp = pd.Timestamp(start_cfg)
        if start_timestamp.tz is None:
            start_timestamp = start_timestamp.tz_localize("UTC")
        else:
            start_timestamp = start_timestamp.tz_convert("UTC")
        start_ns = int(start_timestamp.value)
    else:
        start_ns = int(full_timestamps[0])

    duration_days = float(data_cfg.get("block_duration_days", 112))
    end_ns = start_ns + int(duration_days * 86400 * 1e9)
    left = int(np.searchsorted(full_timestamps, start_ns, side="left"))
    right = int(np.searchsorted(full_timestamps, end_ns, side="left"))

    minimum_rows = int(model_cfg["input_window"]) + int(model_cfg["forecast_horizon"]) + 100
    if right - left < minimum_rows:
        raise ValueError(
            f"{mission}: selected block too short ({right - left} rows); "
            "check block_start_utc/duration"
        )

    timestamps = np.asarray(full_timestamps[left:right], dtype=np.int64)
    values = np.asarray(full_values[left:right], dtype=np.float32)
    channels = info["channels"]

    labels, exclusion_mask, label_info = build_point_labels(
        raw_root / f"ESA-{mission}" / "labels.csv",
        channels,
        timestamps,
        data_cfg.get("pre_anomaly_margin_seconds", 0),
        data_cfg.get("post_anomaly_margin_seconds", 0),
    )

    train_fraction = float(data_cfg["train_fraction"])
    validation_fraction = float(data_cfg["validation_fraction"])
    test_fraction = float(data_cfg["test_fraction"])
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Train/validation/test fractions must sum to 1")

    row_count = len(values)
    train_end = int(row_count * train_fraction)
    validation_end = train_end + int(row_count * validation_fraction)

    labels_csv = pd.read_csv(raw_root / f"ESA-{mission}" / "labels.csv")
    labels_csv = labels_csv[
        labels_csv["Channel"].astype(str).isin([str(channel) for channel in channels])
    ].copy()
    labels_csv["StartTime"] = pd.to_datetime(labels_csv["StartTime"], utc=True)
    labels_csv["EndTime"] = pd.to_datetime(labels_csv["EndTime"], utc=True)

    crossing_ids = set()
    for boundary_ns in (int(timestamps[train_end]), int(timestamps[validation_end])):
        boundary = pd.Timestamp(boundary_ns, tz="UTC")
        crossing = labels_csv[
            (labels_csv["StartTime"] < boundary)
            & (labels_csv["EndTime"] >= boundary)
        ]
        crossing_ids.update(crossing["ID"].astype(str))

    evaluation_ignore = np.zeros(row_count, dtype=np.uint8)
    if crossing_ids:
        for row in labels_csv[
            labels_csv["ID"].astype(str).isin(crossing_ids)
        ].itertuples(index=False):
            left_index = int(
                np.searchsorted(timestamps, int(row.StartTime.value), side="left")
            )
            right_index = int(
                np.searchsorted(timestamps, int(row.EndTime.value), side="right")
            )
            evaluation_ignore[left_index:right_index] = 1

    normal_training_rows = values[:train_end][exclusion_mask[:train_end] == 0]
    if len(normal_training_rows) == 0:
        raise ValueError(f"{mission}: no normal training rows remain")

    mean = normal_training_rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = normal_training_rows.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    normalized = ((values - mean) / std).astype(np.float32)

    bounds = {
        "train": [0, train_end],
        "validation": [train_end, validation_end],
        "test": [validation_end, row_count],
    }

    split_path = work / "splits.npz"
    np.savez_compressed(
        split_path,
        values=normalized,
        labels=labels,
        exclusion_mask=exclusion_mask,
        evaluation_ignore_mask=evaluation_ignore,
        has_labels=np.array(True),
        train_bounds=np.array(bounds["train"]),
        validation_bounds=np.array(bounds["validation"]),
        test_bounds=np.array(bounds["test"]),
        channel_names=np.array(channels),
        timestamps_ns=timestamps,
    )

    metadata = {
        **info,
        "selected_block_start_utc":
            pd.Timestamp(timestamps[0], tz="UTC").isoformat(),
        "selected_block_end_utc":
            pd.Timestamp(timestamps[-1], tz="UTC").isoformat(),
        "requested_duration_days": duration_days,
        "actual_rows": row_count,
        "actual_elapsed_days":
            float((timestamps[-1] - timestamps[0]) / 1e9 / 86400),
        "split_bounds": bounds,
        "split_anomaly_points": {
            "train": int(labels[:train_end].sum()),
            "validation": int(labels[train_end:validation_end].sum()),
            "test": int(labels[validation_end:].sum()),
        },
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "label_info": label_info,
        "cross_split_anomaly_ids_excluded_from_validation_test_metrics":
            sorted(crossing_ids),
        "event_metric_protocol":
            "No event-level metric is evaluated during P1 split preparation.",
        "sampling_note":
            "Original irregular timestamps retained; LSTM steps are observation indices.",
        "alignment_note":
            "All selected channel timestamp indexes required exact identity.",
        "test_label_usage":
            "Exact test labels may be used only after prediction generation for offline evaluation.",
        "warnings": [
            "Confirm professor-approved pre/post anomaly margins.",
            "Unlabelled timestamps are normal only under the supplied interval annotation interpretation.",
        ],
    }
    (work / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return {
        "mission": mission,
        "splits_path": str(split_path),
        "metadata_path": str(work / "preprocessing_metadata.json"),
        "metadata": metadata,
    }


def main() -> None:
    cfg = load_config()
    for mission in cfg["data"].get("missions", ["Mission1"]):
        result = prepare_one(cfg, mission)
        print(
            f"{mission}: prepared {result['metadata']['actual_rows']} rows"
        )


if __name__ == "__main__":
    main()
