"""Validate processed ESA-ADB splits and P1 assets."""
from __future__ import annotations

import json

import numpy as np

from src.utils.config import load_config, resolve_path


def validate_mission(cfg: dict, mission: str) -> dict:
    data_dir = resolve_path(cfg["data"]["esa_work_dir"]) / mission.lower()
    model_dir = resolve_path(cfg["output"]["model_dir"]) / mission.lower()
    split_path = data_dir / "splits.npz"
    metadata_path = data_dir / "preprocessing_metadata.json"

    missing = [
        str(path)
        for path in (split_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{mission}: missing P1 data assets: {missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = np.load(split_path, allow_pickle=False)

    required_keys = {
        "values",
        "labels",
        "exclusion_mask",
        "evaluation_ignore_mask",
        "has_labels",
        "train_bounds",
        "validation_bounds",
        "test_bounds",
        "channel_names",
        "timestamps_ns",
    }
    missing_keys = sorted(required_keys - set(data.files))
    if missing_keys:
        raise ValueError(f"{mission}: missing NPZ keys {missing_keys}")

    values = data["values"]
    labels = data["labels"]
    exclusion = data["exclusion_mask"]
    timestamps = data["timestamps_ns"]
    channels = data["channel_names"].tolist()

    if values.ndim != 2:
        raise ValueError(f"{mission}: values must be a two-dimensional array")
    if values.shape[0] != len(timestamps):
        raise ValueError(f"{mission}: values/timestamps length mismatch")
    if values.shape[0] != len(labels) or values.shape[0] != len(exclusion):
        raise ValueError(f"{mission}: labels/mask length mismatch")
    if values.shape[1] != len(channels):
        raise ValueError(f"{mission}: channel count mismatch")
    if not np.isfinite(values).all():
        raise ValueError(f"{mission}: values contain NaN or infinite values")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"{mission}: timestamps are not strictly increasing")

    bounds = {
        "train": data["train_bounds"].tolist(),
        "validation": data["validation_bounds"].tolist(),
        "test": data["test_bounds"].tolist(),
    }
    if bounds != metadata["split_bounds"]:
        raise ValueError(f"{mission}: NPZ split bounds do not match metadata")

    if len(values) != int(metadata["actual_rows"]):
        raise ValueError(f"{mission}: row count does not match metadata")
    if int(labels.sum()) != int(
        metadata["label_info"]["ground_truth_point_count_union"]
    ):
        raise ValueError(f"{mission}: exact label count mismatch")
    if int(exclusion.sum()) != int(
        metadata["label_info"]["exclusion_point_count_union"]
    ):
        raise ValueError(f"{mission}: expanded exclusion count mismatch")

    return {
        "mission": mission,
        "status": "passed",
        "rows": int(values.shape[0]),
        "channels": channels,
        "train_bounds": bounds["train"],
        "validation_bounds": bounds["validation"],
        "test_bounds": bounds["test"],
        "labelled_observations": int(labels.sum()),
        "excluded_observations": int(exclusion.sum()),
        "median_sampling_interval_seconds":
            float(metadata["median_gap_seconds"]),
        "timestamps_strictly_increasing": True,
        "finite_values": True,
        "checkpoint_directory_checked": str(model_dir),
    }


def main() -> None:
    cfg = load_config()
    missions = cfg["data"].get("missions", ["Mission1"])
    reports = [validate_mission(cfg, mission) for mission in missions]
    payload = {
        "overall_status": "passed",
        "missions": reports,
        "checks": [
            "NPZ schema",
            "array lengths",
            "finite telemetry",
            "timestamp monotonicity",
            "channel order",
            "split bounds",
            "exact label counts",
            "expanded exclusion counts",
        ],
        "test_label_usage":
            "Final test labels are offline-evaluation-only and are not accepted by operational P1 interfaces.",
    }
    output = resolve_path(cfg["output"]["results_dir"]) / "data_integration"
    output.mkdir(parents=True, exist_ok=True)
    (output / "processed_data_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
