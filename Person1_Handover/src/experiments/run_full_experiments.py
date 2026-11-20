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
import shutil
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
    p2_results_dir = resolve_path(cfg["output"].get("p2_results_dir", "P2_results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    p2_results_dir.mkdir(parents=True, exist_ok=True)

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
                "raw_adwin_alarms": b_data["drift_detections"],
                "confirmed_drifts": b_data.get("confirmed_drift_count", 0),
                "false_confirmations": b_data.get("false_confirmation_count", 0),
                "harmful_drift_missed": b_data.get("harmful_drift_missed", False),
                "first_raw_alarm_step": b_data.get("first_raw_alarm_step"),
                "first_confirmed_step": b_data.get("first_confirmed_step"),
                "detection_delay_observations":
                    b_data.get("detection_delay_observations"),
                "mask_suppressions": b_data.get("mask_suppression_count", 0),
                "insufficient_evidence_suppressions":
                    b_data.get("insufficient_evidence_count", 0),
                "candidates_accepted": b_data["candidates_accepted"],
                "candidates_rejected": b_data["candidates_rejected"],
                "runtime_seconds": b_data["runtime_total"],
            })

    df_comp = pd.DataFrame(comparison_rows)
    comp_csv_path = results_dir / "b1_b4_scenario_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\n[OK] Saved B1-B4 comparison table to {comp_csv_path}")
    p2_comp_csv_path = p2_results_dir / "p2_b4_comparison.csv"
    df_comp.to_csv(p2_comp_csv_path, index=False)

    p2_event_rows = []
    for sc_res in all_scenario_results:
        scenario_id = sc_res["scenario"]["scenario_id"]
        for baseline_name, baseline_data in sc_res["baselines"].items():
            for event_row in baseline_data.get("event_log", []):
                p2_event_rows.append({
                    "scenario_id": scenario_id,
                    "scenario_name": sc_res["scenario"]["name"],
                    **event_row,
                })
    p2_event_csv_path = p2_results_dir / "p2_drift_decision_log.csv"
    pd.DataFrame(p2_event_rows).to_csv(p2_event_csv_path, index=False)

    # -------------------------------------------------------------
    # 2. Run Ablation Study on Scenario 3
    #    Adaptation and final test are kept separate.
    # -------------------------------------------------------------
    print("\n---> Running Ablation Studies on Scenario 3 (Persistent Harmful Drift)...")

    m_cfg = cfg["model"]
    stride = int(m_cfg.get("window_stride", 10))

    val_start, val_end = clean_splits["validation_bounds"].tolist()
    adaptation_start, adaptation_end = clean_splits["adaptation_bounds"].tolist()
    test_start, test_end = clean_splits["test_bounds"].tolist()

    seed = cfg.get("seed", 42)

    # Generate Scenario 3 independently on adaptation and test segments.
    clean_adaptation = clean_splits["values"][adaptation_start:adaptation_end].copy()
    mod_adaptation, adaptation_scenario = generate_scenario(
        clean_adaptation, 3, seed=seed
    )

    clean_test = clean_splits["values"][test_start:test_end].copy()
    mod_test, test_scenario = generate_scenario(
        clean_test, 3, seed=seed
    )

    full_adaptation_values = clean_splits["values"].copy()
    full_adaptation_values[adaptation_start:adaptation_end] = mod_adaptation

    full_test_values = clean_splits["values"].copy()
    full_test_values[test_start:test_end] = mod_test

    # Prepare adaptation windows.
    adaptation_starts, adaptation_x, adaptation_y = prepare_fast_windows(
        full_adaptation_values,
        (adaptation_start, adaptation_end),
        m_cfg["input_window"],
        m_cfg["forecast_horizon"],
        stride=stride,
    )

    # Prepare final-test windows separately.
    test_starts, test_x, test_y = prepare_fast_windows(
        full_test_values,
        (test_start, test_end),
        m_cfg["input_window"],
        m_cfg["forecast_horizon"],
        stride=stride,
    )

    # Prepare clean validation windows.
    val_starts, val_x, val_y = prepare_fast_windows(
        clean_splits["values"],
        (val_start, val_end),
        m_cfg["input_window"],
        m_cfg["forecast_horizon"],
        stride=stride,
    )

    from src.models.anomaly_scoring import (
        window_has_anomaly,
        ema_reset_on_excluded,
    )

    # Real test labels are used only for final offline evaluation.
    test_labels = window_has_anomaly(
        clean_splits["labels"],
        test_starts,
        m_cfg["input_window"],
        m_cfg["forecast_horizon"],
        include_input=False,
    )

    # Determine the anomaly threshold using validation data only.
    base_model = load_forecaster(base_checkpoint, device)

    from src.evaluation.benchmark import score_model_on_windows

    val_scores, _ = score_model_on_windows(
        base_model, val_x, val_y, device
    )

    val_bad = window_has_anomaly(
        clean_splits["exclusion_mask"],
        val_starts,
        m_cfg["input_window"],
        m_cfg["forecast_horizon"],
        include_input=True,
    )

    val_ema = ema_reset_on_excluded(
        val_scores,
        ~val_bad,
        cfg["anomaly"]["ema_alpha"],
    )

    norm_val_scores = val_ema[np.isfinite(val_ema)]

    if len(norm_val_scores) == 0:
        raise ValueError("No valid clean validation scores for threshold calibration.")

    threshold = float(
        np.quantile(
            norm_val_scores,
            cfg["anomaly"]["threshold_quantile"],
        )
    )

    # Replay seed is created from training data only.
    from src.continual.replay_buffer import ReplayBuffer

    replay_seed = ReplayBuffer(
        capacity=cfg["continual"]["replay_capacity"],
        seed=seed,
    )

    replay_seed.populate_from_training_split(
        clean_splits["values"],
        clean_splits["exclusion_mask"],
        clean_splits["train_bounds"][0],
        clean_splits["train_bounds"][1],
        m_cfg["input_window"],
        m_cfg["forecast_horizon"],
        stride=50,
        max_samples=300,
    )

    ablation_variants = [
        "Complete_B4",
        "B4_original",
        "B4_dual",
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
            adaptation_x=adaptation_x,
            adaptation_y=adaptation_y,
            test_x=test_x,
            test_y=test_y,
            test_labels=test_labels,
            val_x=val_x,
            val_y=val_y,
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
            "adaptation_steps": abl_res["adaptation_steps"],
            "final_test_steps": abl_res["final_test_steps"],
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

    for i, b in enumerate(["B1", "B2", "B3", "B4_original", "B4_dual"]):
        sub = df_comp[df_comp["baseline"] == b]
        # Safely extract f0.5 scores; replace None with 0.0 for plotting
        f_scores = []
        for sc in scenarios_list:
            val = sub[sub["scenario_name"] == sc]["f0.5_event"].values[0]
            f_scores.append(0.0 if val is None else float(val))
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
    colors = [
        "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728",
        "#9467bd", "#8c564b", "#e377c2",
    ]
    # Safely extract f0.5 scores; use np.nan for None and label N/A in plot
    f_scores = [np.nan if v is None else float(v) for v in df_abl["f0.5_event"]]
    bars = plt.bar(x, f_scores, color=colors, width=0.55)
    for bar, val in zip(bars, f_scores):
        height = 0 if np.isnan(val) else bar.get_height()
        label = "N/A" if np.isnan(val) else f"{height:.3f}"
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.015, label,
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
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
    # Use np.nan for missing values; keep original DataFrame unchanged
    f_scores = [np.nan if v is None else float(v) for v in mask_df["f0.5_event"]]
    bars = plt.bar(["With Masking (B4)", "Without Masking"], f_scores, color=["#2ca02c", "#d62728"], width=0.45)
    for bar, val in zip(bars, f_scores):
        height = 0 if np.isnan(val) else bar.get_height()
        label = "N/A" if np.isnan(val) else f"{height:.3f}"
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, label,
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.ylabel("Event-wise F0.5", fontsize=11)
    plt.ylim(0, 1.15)
    plt.title("Impact of Anomaly Masking on Detection Reliability", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p4_path = plots_dir / "masking_impact_comparison.png"
    plt.savefig(p4_path, dpi=200)
    plt.close()
    print(f"  Saved figure: {p4_path}")

    for plot_path in (p1_path, p2_path, p3_path, p4_path):
        shutil.copy2(plot_path, p2_results_dir / plot_path.name)

    p2_summary = {
        "status": "completed",
        "scenario_count": len(all_scenario_results),
        "baselines": sorted(
            all_scenario_results[0]["baselines"].keys()
        ) if all_scenario_results else [],
        "b1_checkpoint": str(base_checkpoint),
        "b1_retrained": False,
        "comparison_csv": str(p2_comp_csv_path),
        "decision_log_csv": str(p2_event_csv_path),
        "scenario_calibration": [
            {
                "scenario_id": result["scenario"]["scenario_id"],
                "scenario_name": result["scenario"]["name"],
                **result["calibration"],
            }
            for result in all_scenario_results
        ],
        "mask_dominance_threshold":
            cfg["drift"].get("mask_dominance_threshold", 0.75),
        "threshold_note": (
            "Mask-dominance cutoff is configurable; interpret comparison results "
            "using validation/adaptation diagnostics and do not infer superiority "
            "from update counts alone."
        ),
    }
    (p2_results_dir / "p2_experiment_summary.json").write_text(
        json.dumps(p2_summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTAL BENCHMARKS & DELIVERABLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

