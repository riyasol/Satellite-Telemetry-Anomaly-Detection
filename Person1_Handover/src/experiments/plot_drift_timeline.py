"""Generate comprehensive diagnostic timeline plot for Drift, Severity, Masking, and Decisions."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".matplotlib_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from src.utils.config import load_config, resolve_path
from src.models.anomaly_scoring import load_forecaster, ema_reset_on_excluded, window_has_anomaly
from src.evaluation.benchmark import prepare_fast_windows, score_model_on_windows
from src.scenarios.drift_generator import generate_scenario
from src.drift.streaming_detectors import ADWIN
from src.drift.distribution_measures import compute_all_distribution_measures
from src.drift.severity import SeverityCalculator
from src.drift.persistence import PersistenceTracker
from src.drift.degradation import DegradationTracker
from src.drift.decision_engine import DecisionEngine, AdaptationAction
import torch


def main():
    cfg = load_config()
    device = torch.device("cpu")
    work_dir = resolve_path(cfg["data"]["esa_work_dir"]) / "mission1"
    splits_data = np.load(work_dir / "splits.npz", allow_pickle=True)
    clean_splits = {k: splits_data[k] for k in splits_data.files}
    model_dir = resolve_path(cfg["output"]["model_dir"]) / "mission1"
    plots_dir = resolve_path(cfg["output"]["plot_dir"]) / "mission1"

    model = load_forecaster(model_dir / "lstm_b1.pt", device)
    m_cfg = cfg["model"]
    test_bounds = clean_splits["test_bounds"].tolist()
    val_bounds = clean_splits["validation_bounds"].tolist()
    test_start, test_end = test_bounds

    clean_test = clean_splits["values"][test_start:test_end].copy()
    mod_test, sc_cfg = generate_scenario(clean_test, 3, seed=42)
    full_vals = clean_splits["values"].copy()
    full_vals[test_start:test_end] = mod_test

    test_starts, test_x, test_y = prepare_fast_windows(
        full_vals, (test_start, test_end), m_cfg["input_window"], m_cfg["forecast_horizon"], stride=10
    )
    val_starts, val_x, val_y = prepare_fast_windows(
        clean_splits["values"], (val_bounds[0], val_bounds[1]), m_cfg["input_window"], m_cfg["forecast_horizon"], stride=10
    )

    test_labels = window_has_anomaly(clean_splits["labels"], test_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=False)
    test_exclusion = window_has_anomaly(clean_splits["exclusion_mask"], test_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=False)

    val_scores, _ = score_model_on_windows(model, val_x, val_y, device)
    val_bad = window_has_anomaly(clean_splits["exclusion_mask"], val_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=True)
    val_ema = ema_reset_on_excluded(val_scores, ~val_bad, cfg["anomaly"]["ema_alpha"])
    norm_val = val_ema[np.isfinite(val_ema)]
    threshold = float(np.quantile(norm_val, cfg["anomaly"]["threshold_quantile"]))

    scores, _ = score_model_on_windows(model, test_x, test_y, device)

    detector = ADWIN(delta=cfg["drift"]["adwin_delta"])
    severity_calc = SeverityCalculator(weights=cfg["drift"]["severity_weights"])
    severity_calc.calibrate_from_normal_validation(norm_val)
    persistence = PersistenceTracker(window_k=cfg["drift"]["persistence_window_k"], min_detections=cfg["drift"]["persistence_min"])
    degradation = DegradationTracker(reference_f05=0.5, degradation_threshold_delta_q=0.05, reference_mse=float(np.mean(norm_val)))
    engine = DecisionEngine(theta_low=cfg["drift"]["theta_low"], theta_high=cfg["drift"]["theta_high"], persistence_min=cfg["drift"]["persistence_min"])

    steps = len(scores)
    step_indices = np.arange(steps)
    drift_alarms = []
    severities = np.zeros(steps)
    actions = []
    ref_res = list(norm_val[:200])
    recent_res = []

    for s in range(steps):
        s_val = float(scores[s])
        is_masked = bool(test_exclusion[s])
        if not is_masked:
            recent_res.append(s_val)
            if len(recent_res) > cfg["drift"]["recent_window_size"]:
                recent_res.pop(0)

        flag = detector.update(s_val) if not is_masked else False
        if flag:
            drift_alarms.append(s)

        p_rec = persistence.update(flag, step=s)
        if len(recent_res) >= 20:
            raw_m = compute_all_distribution_measures(np.array(ref_res), np.array(recent_res))
            sev_rec = severity_calc.compute_severity(raw_m, step=s)
            severities[s] = sev_rec["severity"]
        else:
            severities[s] = 0.0

        deg_rec = degradation.evaluate_residual_degradation(np.array(recent_res), step=s)
        dec = engine.decide(
            step=s, timestamp_utc=str(s), drift_detected=flag, detector_name="ADWIN",
            raw_measures={"ks": 0.0}, normalized_measures={"ks": 0.0},
            severity=severities[s], persistence_count=p_rec["persistence_count"],
            persistence_passed=p_rec["is_persistent"],
            degradation_value=deg_rec["relative_degradation"],
            degradation_passed=deg_rec["is_significant_degradation"]
        )
        actions.append(dec.action.value)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Subplot 1: Raw Telemetry Stream (Channel 41) & Anomaly Regions
    axes[0].plot(step_indices, test_x[:, 0, 0], color="#2b5c8f", lw=1.0, label="Mission 1 Telemetry (Channel 41)")
    axes[0].axvspan(sc_cfg.start_step // 10, steps, color="#ff9999", alpha=0.3, label="Injected Drift Region")
    anom_steps = np.where(test_labels)[0]
    if len(anom_steps):
        axes[0].scatter(anom_steps, test_x[anom_steps, 0, 0], color="red", s=25, zorder=5, label="True Telemetry Anomalies")
    axes[0].set_ylabel("Telemetry", fontsize=10)
    axes[0].set_title("ESA Mission 1: Multi-Layer Telemetry Monitoring, Drift Severity & Decision Timeline", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Forecaster Residual MSE & Threshold
    axes[1].plot(step_indices, scores, color="#333333", lw=1.0, label="LSTM Forecast MSE")
    axes[1].axhline(threshold, color="green", linestyle="--", lw=1.5, label=f"Anomaly Threshold ({threshold:.3f})")
    if drift_alarms:
        axes[1].scatter(drift_alarms, [scores[i] for i in drift_alarms], color="magenta", marker="^", s=60, zorder=6, label="ADWIN Drift Alarms")
    axes[1].set_ylabel("Residual MSE", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Severity Score (S_t)
    axes[2].plot(step_indices, severities, color="#d95f02", lw=1.2, label=r"Severity $S_t = 0.4 KS_{norm} + 0.3 W_{norm} + 0.3 JS_{norm}$")
    axes[2].axhline(cfg["drift"]["theta_low"], color="#e6ab02", linestyle=":", lw=1.5, label=r"$\theta_L$ (Fine-Tune)")
    axes[2].axhline(cfg["drift"]["theta_high"], color="#e7298a", linestyle=":", lw=1.5, label=r"$\theta_H$ (Strong Update)")
    axes[2].set_ylabel("Severity (0-1)", fontsize=10)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Action Sequence & Masking
    action_map = {"NO_UPDATE": 0, "FINE_TUNE": 1, "STRONG_UPDATE": 2}
    numeric_actions = [action_map[a] for a in actions]
    axes[3].step(step_indices, numeric_actions, where="mid", color="#7570b3", lw=1.2, label="Adaptation Decision")
    mask_indices = np.where(test_exclusion)[0]
    if len(mask_indices):
        axes[3].scatter(mask_indices, np.zeros_like(mask_indices), color="orange", marker="|", s=40, label="Safety Masked Intervals")
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_yticklabels(["NO_UPDATE", "FINE_TUNE", "STRONG_UPDATE"], fontsize=9)
    axes[3].set_ylabel("Decision Action", fontsize=10)
    axes[3].set_xlabel("Test Window Stride Index", fontsize=10)
    axes[3].legend(loc="upper left", fontsize=8)
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = plots_dir / "drift_severity_decision_timeline.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved comprehensive diagnostic timeline to {out_path}")


if __name__ == "__main__":
    main()

