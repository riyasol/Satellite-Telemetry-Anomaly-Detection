"""Master experimental evaluation runner for Mission 1 Spacecraft Telemetry.

Runs:
1. B1–B4 comparative benchmark on all 5 controlled drift scenarios.
2. Ablation study isolating Severity, Persistence, Degradation, and Masking.
3. Detailed Masking vs Unmasked comparison.
4. Output generation: CSV logs, JSON summaries, and publication-ready plots.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure matplotlib uses a writable temp cache directory
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from src.utils.config import load_config, resolve_path
from src.models.anomaly_scoring import load_forecaster
from src.evaluation.benchmark import run_benchmark_for_scenario
from src.evaluation.ablation import run_ablation_simulation
from src.evaluation.benchmark import prepare_fast_windows
from src.scenarios.drift_generator import generate_scenario


def main():
    print("=" * 70)
    print("STARTING FULL EXPERIMENTAL EVALUATION ON ESA MISSION 1 TELEMETRY")
    print("=" * 70)

    cfg = load_config()
    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load Mission 1 processed splits
    work_dir = resolve_path(cfg["data"]["esa_work_dir"]) / "mission1"
    split_path = work_dir / "splits.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Run prepare_esa first.")

    print(f"Loading Mission 1 dataset from {split_path}...")
    splits_data = np.load(split_path, allow_pickle=True)
    clean_splits = {k: splits_data[k] for k in splits_data.files}

    model_dir = resolve_path(cfg["output"]["model_dir"]) / "mission1"
    base_checkpoint = model_dir / "lstm_b1.pt"
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Missing base checkpoint {base_checkpoint}")

    results_dir = resolve_path(cfg["output"]["results_dir"]) / "mission1"
    plots_dir = resolve_path(cfg["output"]["plot_dir"]) / "mission1"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Run B1–B4 across all 5 Scenarios
    # -------------------------------------------------------------
    all_scenario_results = []
    comparison_rows = []

    for sc_id in range(1, 6):
        sc_start = time.perf_counter()
        print(f"\n---> Running Benchmark on Scenario {sc_id}...")
        sc_res = run_benchmark_for_scenario(
            scenario_id=sc_id,
            cfg=cfg,
            clean_splits=clean_splits,
            device=device,
            initial_checkpoint_path=base_checkpoint,
        )
        all_scenario_results.append(sc_res)
        sc_name = sc_res["scenario"]["name"]
        print(f"     Finished Scenario {sc_id} ({sc_name}) in {time.perf_counter() - sc_start:.1f}s")

        for b_code, b_data in sc_res["baselines"].items():
            ev = b_data["event_metrics"]
            comparison_rows.append({
                "scenario_id": sc_id,
                "scenario_name": sc_name,
                "baseline": b_code,
                "baseline_name": b_data["name"],
                "f0.5_event": ev["f0.5_event"],
                "precision": ev["precision"],
                "recall": ev["recall"],
                "tp_events": ev["tp_events"],
                "fp_events": ev["fp_events"],
                "fn_events": ev["fn_events"],
                "total_updates": b_data["updates"],
                "unnecessary_updates": b_data["unnecessary_updates"],
                "candidates_accepted": b_data["candidates_accepted"],
                "candidates_rejected": b_data["candidates_rejected"],
                "runtime_seconds": b_data["runtime_total"],
            })

    df_comp = pd.DataFrame(comparison_rows)
    comp_csv_path = results_dir / "b1_b4_scenario_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\n[OK] Saved B1-B4 comparison table to {comp_csv_path}")

    # -------------------------------------------------------------
    # 2. Run Ablation Study on Scenario 3 (Persistent Harmful Drift)
    # -------------------------------------------------------------
    print("\n---> Running Ablation Studies on Scenario 3 (Persistent Harmful Drift)...")
    m_cfg = cfg["model"]
    stride = int(m_cfg.get("window_stride", 10))
    val_bounds = clean_splits["validation_bounds"].tolist()
    test_bounds = clean_splits["test_bounds"].tolist()
    test_start, test_end = test_bounds

    clean_test = clean_splits["values"][test_start:test_end].copy()
    mod_test_s3, _ = generate_scenario(clean_test, 3, seed=cfg.get("seed", 42))
    full_vals_s3 = clean_splits["values"].copy()
    full_vals_s3[test_start:test_end] = mod_test_s3

    test_starts, test_x, test_y = prepare_fast_windows(
        full_vals_s3, (test_start, test_end), m_cfg["input_window"], m_cfg["forecast_horizon"], stride=stride
    )
    val_starts, val_x, val_y = prepare_fast_windows(
        clean_splits["values"], (val_bounds[0], val_bounds[1]), m_cfg["input_window"], m_cfg["forecast_horizon"], stride=stride
    )

    from src.models.anomaly_scoring import window_has_anomaly, ema_reset_on_excluded
    test_labels = window_has_anomaly(clean_splits["labels"], test_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=False)
    test_exclusion = window_has_anomaly(clean_splits["exclusion_mask"], test_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=False)

    base_model = load_forecaster(base_checkpoint, device)
    from src.evaluation.benchmark import score_model_on_windows
    val_scores, _ = score_model_on_windows(base_model, val_x, val_y, device)
    val_bad = window_has_anomaly(clean_splits["exclusion_mask"], val_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=True)
    val_ema = ema_reset_on_excluded(val_scores, ~val_bad, cfg["anomaly"]["ema_alpha"])
    norm_val_scores = val_ema[np.isfinite(val_ema)]
    threshold = float(np.quantile(norm_val_scores, cfg["anomaly"]["threshold_quantile"]))

    from src.continual.replay_buffer import ReplayBuffer
    replay_seed = ReplayBuffer(capacity=cfg["continual"]["replay_capacity"], seed=cfg.get("seed", 42))
    replay_seed.populate_from_training_split(
        clean_splits["values"], clean_splits["exclusion_mask"],
        clean_splits["train_bounds"][0], clean_splits["train_bounds"][1],
        m_cfg["input_window"], m_cfg["forecast_horizon"], stride=50, max_samples=300
    )

    ablation_variants = [
        "Complete_B4",
        "No_Severity",
        "No_Persistence",
        "No_Degradation",
        "No_Anomaly_Masking",
    ]

    ablation_rows = []
    for var in ablation_variants:
        print(f"     Testing variant: {var}...")
        abl_res = run_ablation_simulation(
            ablation_variant=var,
            test_x=test_x, test_y=test_y,
            test_labels=test_labels, test_exclusion=test_exclusion,
            val_x=val_x, val_y=val_y,
            norm_val_scores=norm_val_scores,
            base_model=base_model,
            replay_seed_buffer=replay_seed,
            device=device,
            cfg=cfg,
            threshold=threshold,
            scenario_id=3,
        )
        ev = abl_res["event_metrics"]
        ablation_rows.append({
            "variant": var,
            "f0.5_event": ev["f0.5_event"],
            "precision": ev["precision"],
            "recall": ev["recall"],
            "tp_events": ev["tp_events"],
            "fp_events": ev["fp_events"],
            "total_updates": abl_res["updates"],
            "candidates_accepted": abl_res["candidates_accepted"],
            "candidates_rejected": abl_res["candidates_rejected"],
            "runtime_seconds": abl_res["runtime_total"],
        })

    df_abl = pd.DataFrame(ablation_rows)
    abl_csv_path = results_dir / "ablation_study_results.csv"
    df_abl.to_csv(abl_csv_path, index=False)
    print(f"[OK] Saved ablation study results to {abl_csv_path}")

    # -------------------------------------------------------------
    # 3. Generate Diagnostic Plots
    # -------------------------------------------------------------
    print("\n---> Generating Publication-Quality Figures...")

    # Plot 1: F0.5 Event-wise Comparison Across Scenarios
    plt.figure(figsize=(11, 5))
    scenarios_list = df_comp["scenario_name"].unique()
    x = np.arange(len(scenarios_list))
    width = 0.20

    for i, b in enumerate(["B1", "B2", "B3", "B4"]):
        sub = df_comp[df_comp["baseline"] == b]
        f_scores = [sub[sub["scenario_name"] == sc]["f0.5_event"].values[0] for sc in scenarios_list]
        plt.bar(x + (i - 1.5) * width, f_scores, width=width, label=b)

    plt.xticks(x, [f"S{i+1}: {s}" for i, s in enumerate(scenarios_list)], rotation=15, ha="right", fontsize=9)
    plt.ylabel("Event-wise F0.5 Score", fontsize=11)
    plt.title("Mission 1: Event-wise F0.5 Comparison Across Baselines & Drift Scenarios", fontsize=12, fontweight="bold")
    plt.legend(title="Baseline", loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1_path = plots_dir / "b1_b4_f05_comparison.png"
    plt.savefig(p1_path, dpi=200)
    plt.close()
    print(f"  Saved figure: {p1_path}")

    # Plot 2: Unnecessary Updates Comparison
    plt.figure(figsize=(9, 4.5))
    unnec_summary = df_comp.groupby("baseline")[["unnecessary_updates", "total_updates"]].sum().reset_index()
    x = np.arange(len(unnec_summary))
    w = 0.35
    plt.bar(x - w/2, unnec_summary["total_updates"], width=w, label="Total Updates Triggered", color="#4C72B0")
    plt.bar(x + w/2, unnec_summary["unnecessary_updates"], width=w, label="Unnecessary Updates (Harmless/Stable)", color="#C44E52")
    plt.xticks(x, unnec_summary["baseline"], fontsize=10)
    plt.ylabel("Cumulative Update Count", fontsize=11)
    plt.title("Cumulative Adaptation Updates & Unnecessary Updates Across All Scenarios", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2_path = plots_dir / "unnecessary_updates_comparison.png"
    plt.savefig(p2_path, dpi=200)
    plt.close()
    print(f"  Saved figure: {p2_path}")

    # Plot 3: Ablation Study Comparison
    plt.figure(figsize=(10, 4.5))
    x = np.arange(len(df_abl))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    bars = plt.bar(x, df_abl["f0.5_event"], color=colors, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(x, df_abl["variant"].str.replace("_", " "), rotation=15, ha="right", fontsize=9)
    plt.ylabel("Event-wise F0.5", fontsize=11)
    plt.ylim(0, 1.15)
    plt.title("Ablation Study on Persistent Harmful Drift (Scenario 3)", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3_path = plots_dir / "ablation_study_f05.png"
    plt.savefig(p3_path, dpi=200)
    plt.close()
    print(f"  Saved figure: {p3_path}")

    # Plot 4: Masking vs Unmasked Precision/F0.5
    plt.figure(figsize=(7, 4.5))
    mask_df = df_abl[df_abl["variant"].isin(["Complete_B4", "No_Anomaly_Masking"])]
    bars = plt.bar(["With Masking (B4)", "Without Masking"], mask_df["f0.5_event"], color=["#2ca02c", "#d62728"], width=0.45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.ylabel("Event-wise F0.5", fontsize=11)
    plt.ylim(0, 1.15)
    plt.title("Impact of Anomaly Masking on Detection Reliability", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p4_path = plots_dir / "masking_impact_comparison.png"
    plt.savefig(p4_path, dpi=200)
    plt.close()
    print(f"  Saved figure: {p4_path}")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTAL BENCHMARKS & DELIVERABLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

