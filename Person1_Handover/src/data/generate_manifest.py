"""Generate a P1 data and window manifest without copying large arrays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.drift.window_eligibility import summarize_mask
from src.utils.config import load_config, resolve_path


def build_manifest(cfg: dict) -> dict:
    mission_reports = []
    window_stride = int(cfg["model"].get("window_stride", 1))
    input_window = int(cfg["model"]["input_window"])
    horizon = int(cfg["model"]["forecast_horizon"])

    for mission in cfg["data"].get("missions", ["Mission1"]):
        work_dir = resolve_path(cfg["data"]["esa_work_dir"]) / mission.lower()
        split_path = work_dir / "splits.npz"
        metadata_path = work_dir / "preprocessing_metadata.json"
        report = {
            "mission": mission,
            "split_path": str(split_path),
            "metadata_path": str(metadata_path),
            "available": split_path.exists() and metadata_path.exists(),
        }

        if report["available"]:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            data = np.load(split_path, allow_pickle=False)
            train_start, train_end = data["train_bounds"].tolist()
            validation_start, validation_end = data["validation_bounds"].tolist()
            test_start, test_end = data["test_bounds"].tolist()

            report.update({
                "actual_rows": int(len(data["timestamps_ns"])),
                "channels": data["channel_names"].tolist(),
                "train_bounds": [int(train_start), int(train_end)],
                "validation_bounds":
                    [int(validation_start), int(validation_end)],
                "test_bounds": [int(test_start), int(test_end)],
                "labelled_observations":
                    int(data["labels"].sum()),
                "expanded_excluded_observations":
                    int(data["exclusion_mask"].sum()),
                "evaluation_ignore_observations":
                    int(data["evaluation_ignore_mask"].sum()),
                "timestamps_start_ns":
                    int(data["timestamps_ns"][0]),
                "timestamps_end_ns":
                    int(data["timestamps_ns"][-1]),
                "window_config": {
                    "input_window": input_window,
                    "forecast_horizon": horizon,
                    "stride": window_stride,
                },
                "exclusion_intervals":
                    summarize_mask(data["exclusion_mask"]),
                "preprocessing_metadata": metadata,
            })
            data.close()

        mission_reports.append(report)

    return {
        "manifest_type": "p1_esa_derived_data_manifest",
        "missions": mission_reports,
        "held_out_test_rule":
            "Test labels are offline-evaluation-only and are not P2/P3 update inputs.",
        "event_tolerance":
            cfg.get("evaluation", {}).get("event_tolerance_steps", None),
        "event_tolerance_status":
            "provisional_pending_team_approval"
            if cfg.get("evaluation", {}).get("event_tolerance_is_provisional")
            else "configured",
        "large_artifacts_policy":
            "NPZ arrays and PT checkpoints remain local and must not be committed.",
    }


def main() -> None:
    cfg = load_config()
    payload = build_manifest(cfg)
    output = resolve_path(cfg["output"]["results_dir"]) / "data_integration"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "p1_data_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
