"""Run reproducible B1-B4 scenarios and P3 component ablations."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from src.continual.replay_buffer import ReplayBuffer
from src.evaluation.ablation import ABLATION_VARIANTS, run_ablation_simulation
from src.evaluation.benchmark import (
    prepare_scenario_data,
    run_benchmark_for_scenario,
)
from src.models.anomaly_scoring import load_forecaster
from src.utils.config import load_config, resolve_path


def _experiment_parameters(cfg: Dict[str, Any]) -> str:
    relevant = {
        "seed": cfg.get("seed", 42),
        "model": cfg["model"],
        "anomaly": cfg["anomaly"],
        "evaluation": cfg.get("evaluation", {}),
        "drift": cfg["drift"],
        "continual": cfg["continual"],
        "experiments": cfg.get("experiments", {}),
    }
    return json.dumps(relevant, sort_keys=True)


def _result_row(
    scenario_id: int,
    scenario_name: str,
    baseline: str,
    result: Dict[str, Any],
    parameters_json: str,
    scenario: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    metrics = result["event_metrics"]
    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "baseline": baseline,
        "baseline_name": result["name"],
        "seed": seed,
        "f0.5_event": metrics["f0.5_event"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "tp_events": metrics["tp_events"],
        "fp_events": metrics["fp_events"],
        "fn_events": metrics["fn_events"],
        "evaluated_windows": metrics.get("evaluated_windows"),
        "total_updates": result["updates"],
        "fine_tune_updates": result.get("fine_tune_updates", 0),
        "strong_updates": result.get("strong_updates", 0),
        "update_requests": result.get("update_requests", 0),
        "unnecessary_updates": result.get("unnecessary_updates", 0),
        "updates_during_stable_period":
            result.get("updates_during_stable_period", 0),
        "updates_during_temporary_period":
            result.get("updates_during_temporary_period", 0),
        "updates_during_large_harmless_period":
            result.get("updates_during_large_harmless_period", 0),
        "drift_detections": result.get("drift_detections", 0),
        "masked_windows": result.get("masked_windows", 0),
        "candidates_accepted": result.get("candidates_accepted", 0),
        "candidates_rejected": result.get("candidates_rejected", 0),
        "first_adaptation_delay_windows":
            result.get("first_adaptation_delay_windows"),
        "first_alarm_delay_windows": result.get("first_alarm_delay_windows"),
        "runtime_seconds": result["runtime_total"],
        "runtime_training_seconds": result.get("runtime_training", 0.0),
        "runtime_validation_seconds": result.get("runtime_validation", 0.0),
        "runtime_detection_seconds": result.get("runtime_detection", 0.0),
        "runtime_decision_seconds": result.get("runtime_decision", 0.0),
        "parameters_json": parameters_json,
        "scenario_parameters_json": json.dumps(scenario, sort_keys=True),
    }


def _save_comparison_plots(
    comparison: pd.DataFrame,
    ablation: pd.DataFrame,
    plots_dir: Path,
) -> None:
    scenarios = comparison["scenario_name"].drop_duplicates().tolist()
    x = np.arange(len(scenarios))
    width = 0.2
    plt.figure(figsize=(11, 5))
    for index, baseline in enumerate(("B1", "B2", "B3", "B4")):
        rows = comparison[comparison["baseline"] == baseline]
        scores = [
            rows.loc[rows["scenario_name"] == name, "f0.5_event"].iloc[0]
            for name in scenarios
        ]
        plt.bar(x + (index - 1.5) * width, scores, width, label=baseline)
    plt.xticks(x, scenarios, rotation=15, ha="right")
    plt.ylabel("Event-wise F0.5")
    plt.title("B1-B4 event detection across controlled scenarios")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "b1_b4_f05_comparison.png", dpi=180)
    plt.close()

    update_summary = comparison.groupby("baseline")[
        ["total_updates", "unnecessary_updates"]
    ].sum()
    x = np.arange(len(update_summary))
    width = 0.36
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width / 2, update_summary["total_updates"], width, label="All updates")
    plt.bar(
        x + width / 2,
        update_summary["unnecessary_updates"],
        width,
        label="Updates in stable/temporary/harmless scenarios",
    )
    plt.xticks(x, update_summary.index)
    plt.ylabel("Accepted model updates")
    plt.title("Continual-learning update cost")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "unnecessary_updates_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4.5))
    x = np.arange(len(ablation))
    plt.bar(x, ablation["f0.5_event"], color="#4C72B0", label="Event F0.5")
    plt.xticks(x, ablation["variant"].str.replace("_", " "), rotation=15, ha="right")
    plt.ylabel("Event-wise F0.5")
    plt.ylim(0, 1.05)
    plt.title("B4 component ablation on persistent harmful drift")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "ablation_study_f05.png", dpi=180)
    plt.close()

    masking = ablation[
        ablation["variant"].isin(("Complete_B4", "No_Anomaly_Masking"))
    ].set_index("variant")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ["Complete B4", "No anomaly masking"]
    order = ["Complete_B4", "No_Anomaly_Masking"]
    axes[0].bar(labels, [masking.loc[key, "f0.5_event"] for key in order],
                color=["#2ca02c", "#d62728"])
    axes[0].set_ylabel("Event-wise F0.5")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(
        labels,
        [masking.loc[key, "total_updates"] for key in order],
        color=["#2ca02c", "#d62728"],
    )
    axes[1].set_ylabel("Accepted model updates")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("Effect of masking injected anomaly pulses near drift")
    fig.tight_layout()
    fig.savefig(plots_dir / "masking_impact_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("B1-B4 CONTROLLED-DRIFT EVALUATION")
    print("=" * 70)
    cfg = load_config()
    device_name = cfg.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    print(f"Using device: {device}")

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
    results_dir = resolve_path(cfg["output"]["results_dir"]) / "mission1"
    plots_dir = resolve_path(cfg["output"]["plot_dir"]) / "mission1"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    parameters_json = _experiment_parameters(cfg)
    seed = int(cfg.get("seed", 42))
    comparison_rows = []
    for scenario_id in range(1, 6):
        started = time.perf_counter()
        print(f"Running scenario {scenario_id}/5...")
        result = run_benchmark_for_scenario(
            scenario_id, cfg, clean_splits, device, checkpoint
        )
        scenario_name = result["scenario"]["name"]
        for baseline, baseline_result in result["baselines"].items():
            comparison_rows.append(
                _result_row(
                    scenario_id,
                    scenario_name,
                    baseline,
                    baseline_result,
                    parameters_json,
                    result["scenario"],
                    seed,
                )
            )
        print(f"Completed {scenario_name} in {time.perf_counter() - started:.1f}s")

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = results_dir / "b1_b4_scenario_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"Saved scenario results: {comparison_path}")

    print("Running persistent-harmful-drift ablations with injected anomaly pulses...")
    base_model = load_forecaster(checkpoint, device)
    scenario_data = prepare_scenario_data(
        scenario_id=3,
        cfg=cfg,
        clean_splits=clean_splits,
        base_model=base_model,
        device=device,
        inject_anomalies=True,
    )
    train_start, train_end = map(int, clean_splits["train_bounds"])
    replay_seed = ReplayBuffer(
        capacity=int(cfg["continual"]["replay_capacity"]), seed=seed
    )
    replay_seed.populate_from_training_split(
        clean_splits["values"],
        clean_splits["exclusion_mask"],
        train_start,
        train_end,
        int(cfg["model"]["input_window"]),
        int(cfg["model"]["forecast_horizon"]),
        stride=max(1, int(cfg["model"].get("window_stride", 1)) * 5),
        max_samples=min(300, int(cfg["continual"]["replay_capacity"])),
    )
    if replay_seed.size() == 0:
        raise ValueError("No eligible training windows are available for replay.")

    ablation_rows = []
    for variant in ABLATION_VARIANTS:
        print(f"  Running {variant}...")
        result = run_ablation_simulation(
            variant,
            scenario_id=3,
            scenario_data=scenario_data,
            base_model=base_model,
            replay_seed_buffer=replay_seed,
            device=device,
            cfg=cfg,
        )
        row = _result_row(
            3,
            scenario_data["scenario"]["name"],
            "B4",
            result,
            parameters_json,
            scenario_data["scenario"],
            seed,
        )
        row["variant"] = variant
        row["injected_adaptation_anomaly_points"] = int(
            scenario_data["scenario"]["adaptation_anomaly_points"]
        )
        row["injected_evaluation_anomaly_points"] = int(
            scenario_data["scenario"]["evaluation_anomaly_points"]
        )
        ablation_rows.append(row)

    ablation = pd.DataFrame(ablation_rows)
    ablation_path = results_dir / "ablation_study_results.csv"
    ablation.to_csv(ablation_path, index=False)
    print(f"Saved ablation results: {ablation_path}")

    masking_path = results_dir / "masking_ablation_results.csv"
    ablation[
        ablation["variant"].isin(("Complete_B4", "No_Anomaly_Masking"))
    ].to_csv(masking_path, index=False)
    print(f"Saved masking comparison: {masking_path}")

    _save_comparison_plots(comparison, ablation, plots_dir)
    print(f"Saved plots under {plots_dir}")


if __name__ == "__main__":
    main()
