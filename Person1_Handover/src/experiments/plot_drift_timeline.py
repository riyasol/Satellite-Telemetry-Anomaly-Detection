"""Plot the P3 adaptation stream without using test labels for online decisions."""
from __future__ import annotations

import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.continual.replay_buffer import ReplayBuffer
from src.evaluation.benchmark import (
    prepare_scenario_data,
    run_adaptive_simulation,
)
from src.models.anomaly_scoring import load_forecaster
from src.utils.config import load_config, resolve_path


def main() -> None:
    cfg = load_config()
    device_name = cfg.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    work_dir = resolve_path(cfg["data"]["esa_work_dir"]) / "mission1"
    split_path = work_dir / "splits.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Run prepare_esa first.")
    with np.load(split_path, allow_pickle=False) as archive:
        clean_splits = {key: archive[key] for key in archive.files}

    model_dir = resolve_path(cfg["output"]["model_dir"]) / "mission1"
    checkpoint = model_dir / "lstm_b1.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing base checkpoint {checkpoint}.")
    model = load_forecaster(checkpoint, device)
    scenario_data = prepare_scenario_data(
        scenario_id=3,
        cfg=cfg,
        clean_splits=clean_splits,
        base_model=model,
        device=device,
        inject_anomalies=True,
    )

    train_start, train_end = map(int, clean_splits["train_bounds"])
    replay = ReplayBuffer(
        capacity=int(cfg["continual"]["replay_capacity"]),
        seed=int(cfg.get("seed", 42)),
    )
    replay.populate_from_training_split(
        clean_splits["values"],
        clean_splits["exclusion_mask"],
        train_start,
        train_end,
        int(cfg["model"]["input_window"]),
        int(cfg["model"]["forecast_horizon"]),
        stride=max(1, int(cfg["model"].get("window_stride", 1)) * 5),
        max_samples=min(300, int(cfg["continual"]["replay_capacity"])),
    )
    result = run_adaptive_simulation(
        baseline_name="B4",
        scenario_id=3,
        scenario_data=scenario_data,
        cfg=cfg,
        device=device,
        base_model=model,
        replay_seed_buffer=replay,
    )
    trace = result["trace"]
    steps = np.arange(len(trace["raw_mse"]))
    drift_start = scenario_data["adaptation_drift_start_window"]
    output_dir = resolve_path(cfg["output"]["plot_dir"]) / "mission1"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    telemetry = scenario_data["adaptation_x"][:, 0, 0]
    axes[0].plot(steps, telemetry, color="#2b5c8f", lw=1.0, label="Adaptation telemetry")
    if drift_start is not None:
        axes[0].axvline(
            drift_start, color="#d62728", linestyle="--", label="Injected drift start"
        )
    axes[0].set_ylabel("Channel 1")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(steps, trace["raw_mse"], color="#333333", lw=1.0, label="Forecast MSE")
    axes[1].axhline(
        scenario_data["threshold"],
        color="green",
        linestyle="--",
        label="Normal calibration threshold",
    )
    predicted = np.flatnonzero(trace["predicted_anomaly"])
    if len(predicted):
        axes[1].scatter(
            predicted,
            trace["raw_mse"][predicted],
            color="orange",
            marker="x",
            label="Predicted anomaly",
        )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("MSE")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(steps, trace["severity"], color="#d95f02", label="Severity")
    axes[2].axhline(cfg["drift"]["theta_low"], color="#e6ab02", linestyle=":")
    axes[2].axhline(cfg["drift"]["theta_high"], color="#e7298a", linestyle=":")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Severity")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    actions = trace["action"]
    action_values = np.array([
        2 if action == "STRONG_UPDATE" else 1 if action == "FINE_TUNE" else 0
        for action in actions
    ])
    axes[3].step(steps, action_values, where="mid", color="#7570b3", label="Requested update")
    masked = np.flatnonzero(trace["masked"])
    if len(masked):
        axes[3].scatter(
            masked, np.zeros(len(masked)), color="orange", marker="|", label="Masked windows"
        )
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_yticklabels(["NO_UPDATE", "FINE_TUNE", "STRONG_UPDATE"])
    axes[3].set_ylabel("Decision")
    axes[3].set_xlabel("Adaptation-stream window")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("B4 adaptation stream: predicted-anomaly masking and drift decisions")
    fig.tight_layout()
    output_path = output_dir / "drift_severity_decision_timeline.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved diagnostic timeline: {output_path}")
    print(
        f"Held-out event F0.5={result['event_metrics']['f0.5_event']:.4f}; "
        f"updates={result['updates']}; "
        f"candidates rejected={result['candidates_rejected']}"
    )


if __name__ == "__main__":
    main()
