"""Comprehensive simulation benchmark comparing B1, B2, B3, and B4 across all 5 drift scenarios on Mission 1.

Baselines:
- B1: Static LSTM baseline (no adaptation).
- B2: Kuhn Continual Learning: Replay/SER continual learning + validation gate, periodic updates.
- B3: Drift-Triggered Adaptation: Drift detection triggers update + validation gate without decision layer.
- B4: Proposed Drift-Aware System: Anomaly masking + Severity + Persistence + Degradation -> 3-way decision -> SER update + Validation Gate.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config, resolve_path
from src.models.anomaly_scoring import load_forecaster, ema_reset_on_excluded, window_has_anomaly
from src.models.lstm_forecaster import LSTMForecaster
from src.drift.streaming_detectors import ADWIN
from src.drift.distribution_measures import compute_all_distribution_measures
from src.drift.anomaly_masker import AnomalyMasker
from src.drift.severity import SeverityCalculator
from src.drift.persistence import PersistenceTracker
from src.drift.degradation import DegradationTracker
from src.drift.decision_engine import DecisionEngine, AdaptationAction
from src.continual.replay_buffer import ReplayBuffer
from src.continual.candidate_trainer import CandidateTrainer
from src.continual.validation_gate import ValidationGate
from src.scenarios.drift_generator import generate_scenario, DriftScenarioConfig
from src.evaluation.event_metrics import compute_event_metrics, compute_point_metrics


def prepare_fast_windows(
    values: np.ndarray,
    bounds: Tuple[int, int],
    input_window: int = 250,
    forecast_horizon: int = 10,
    stride: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (x, y) arrays and start indices efficiently."""
    start_bound, end_bound = bounds
    max_start = end_bound - input_window - forecast_horizon
    starts = np.arange(start_bound, max_start + 1, stride, dtype=int)
    n_windows = len(starts)
    n_channels = values.shape[1]

    x_arr = np.zeros((n_windows, input_window, n_channels), dtype=np.float32)
    y_arr = np.zeros((n_windows, forecast_horizon, n_channels), dtype=np.float32)

    for i, s in enumerate(starts):
        xe = s + input_window
        ye = xe + forecast_horizon
        x_arr[i] = values[s:xe]
        y_arr[i] = values[xe:ye]

    return starts, x_arr, y_arr


def score_model_on_windows(
    model: LSTMForecaster,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, float]:
    """Score model in fast batched mode and return MSE scores per window and inference time."""
    model.eval()
    t0 = time.perf_counter()
    n = len(x_arr)
    raw_mse = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for b in range(0, n, batch_size):
            bx = torch.from_numpy(x_arr[b : b + batch_size]).to(device)
            by = torch.from_numpy(y_arr[b : b + batch_size]).to(device)
            pred = model(bx)
            diff = pred - by
            mse = torch.mean(diff ** 2, dim=(1, 2)).cpu().numpy()
            raw_mse[b : b + batch_size] = mse

    return raw_mse, time.perf_counter() - t0


def run_benchmark_for_scenario(
    scenario_id: int,
    cfg: Dict[str, Any],
    clean_splits: Dict[str, Any],
    device: torch.device,
    initial_checkpoint_path: Path,
) -> Dict[str, Any]:
    """Run B1, B2, B3, B4 on a single drift scenario on Mission 1."""
    m_cfg = cfg["model"]
    a_cfg = cfg["anomaly"]
    d_cfg = cfg["drift"]
    c_cfg = cfg["continual"]

    values_clean = clean_splits["values"]
    labels_clean = clean_splits["labels"]
    exclusion_clean = clean_splits["exclusion_mask"]
    timestamps = clean_splits["timestamps_ns"]
    train_bounds = clean_splits["train_bounds"].tolist()
    val_bounds = clean_splits["validation_bounds"].tolist()
    test_bounds = clean_splits["test_bounds"].tolist()

    # Generate scenario values on test split
    test_start, test_end = test_bounds
    clean_test_values = values_clean[test_start:test_end].copy()
    mod_test_values, sc_config = generate_scenario(clean_test_values, scenario_id, seed=cfg.get("seed", 42))

    full_values = values_clean.copy()
    full_values[test_start:test_end] = mod_test_values

    # Pre-extract windows
    stride = int(m_cfg.get("window_stride", 10))
    test_starts, test_x, test_y = prepare_fast_windows(
        full_values, (test_start, test_end), m_cfg["input_window"], m_cfg["forecast_horizon"], stride=stride
    )
    val_starts, val_x, val_y = prepare_fast_windows(
        values_clean, (val_bounds[0], val_bounds[1]), m_cfg["input_window"], m_cfg["forecast_horizon"], stride=stride
    )

    test_labels = window_has_anomaly(labels_clean, test_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=False)
    test_exclusion = window_has_anomaly(exclusion_clean, test_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=False)

    base_model = load_forecaster(initial_checkpoint_path, device)
    val_scores, _ = score_model_on_windows(base_model, val_x, val_y, device)
    val_bad = window_has_anomaly(clean_splits["exclusion_mask"], val_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=True)
    val_ema = ema_reset_on_excluded(val_scores, ~val_bad, a_cfg["ema_alpha"])
    norm_val_scores = val_ema[np.isfinite(val_ema)]
    threshold = float(np.quantile(norm_val_scores, a_cfg["threshold_quantile"]))

    # Prepare Replay Buffer seeded from training data
    replay_seed_buffer = ReplayBuffer(capacity=c_cfg["replay_capacity"], seed=cfg.get("seed", 42))
    replay_seed_buffer.populate_from_training_split(
        values_clean, clean_splits["exclusion_mask"],
        train_bounds[0], train_bounds[1],
        m_cfg["input_window"], m_cfg["forecast_horizon"],
        stride=50, max_samples=300
    )

    results = {"scenario": sc_config.to_dict(), "baselines": {}}

    # -------------------------------------------------------------
    # B1: Static LSTM Baseline
    # -------------------------------------------------------------
    b1_start_t = time.perf_counter()
    b1_raw, b1_inf_t = score_model_on_windows(base_model, test_x, test_y, device)
    b1_ema = ema_reset_on_excluded(b1_raw, ~test_exclusion, a_cfg["ema_alpha"])
    b1_pred = (b1_ema > threshold) & np.isfinite(b1_ema)
    b1_event = compute_event_metrics(test_labels, b1_pred)
    b1_total_t = time.perf_counter() - b1_start_t

    results["baselines"]["B1"] = {
        "name": "B1 - Static LSTM",
        "event_metrics": b1_event,
        "updates": 0,
        "candidates_accepted": 0,
        "candidates_rejected": 0,
        "unnecessary_updates": 0,
        "runtime_total": b1_total_t,
        "runtime_inference": b1_inf_t,
        "runtime_training": 0.0,
        "final_model_val_loss": float(np.mean(val_scores)),
    }

    # -------------------------------------------------------------
    # Adaptive Stream Simulation Helper (B2, B3, B4)
    # -------------------------------------------------------------
    def run_adaptive_stream(baseline_name: str) -> Dict[str, Any]:
        sim_start_t = time.perf_counter()
        t_detect = 0.0
        t_decision = 0.0
        t_train = 0.0
        t_val = 0.0

        current_model = copy.deepcopy(base_model)
        val_gate = ValidationGate(val_x, val_y, device, tolerance=c_cfg["validation_tolerance"])
        replay = copy.deepcopy(replay_seed_buffer)
        trainer = CandidateTrainer(base_model, device, replay, resolve_path(cfg["output"]["model_dir"]) / "candidates")

        detector = ADWIN(delta=d_cfg["adwin_delta"])
        persistence = PersistenceTracker(window_k=d_cfg["persistence_window_k"], min_detections=d_cfg["persistence_min"])
        severity_calc = SeverityCalculator(weights=d_cfg["severity_weights"])
        severity_calc.calibrate_from_normal_validation(norm_val_scores)
        degradation_tracker = DegradationTracker(
            reference_f05=b1_event["f0.5_event"],
            degradation_threshold_delta_q=d_cfg["degradation_threshold_delta_q"],
            reference_mse=float(np.mean(norm_val_scores))
        )
        engine = DecisionEngine(
            theta_low=d_cfg["theta_low"],
            theta_high=d_cfg["theta_high"],
            persistence_min=d_cfg["persistence_min"],
            degradation_threshold=d_cfg["degradation_threshold_delta_q"]
        )

        n_steps = len(test_x)
        # Pre-score all windows with initial model for speed
        cached_scores, inf_time = score_model_on_windows(current_model, test_x, test_y, device)
        stream_pred = np.zeros(n_steps, dtype=bool)
        current_ema = 0.0
        first_step = True

        updates_applied = 0
        fine_tune_count = 0
        strong_update_count = 0
        unnecessary_updates = 0
        drift_detections_count = 0
        recent_buffer_x = []
        recent_buffer_y = []
        recent_residuals = []
        reference_residuals = list(norm_val_scores[:200])

        b2_periodic_interval = c_cfg.get("b2_periodic_interval", 1000)

        for step in range(n_steps):
            mse_w = float(cached_scores[step])

            # Proper EMA with configured ema_alpha
            alpha = float(a_cfg["ema_alpha"])
            if test_exclusion[step]:
                current_ema = np.nan
                first_step = True
            else:
                if first_step or np.isnan(current_ema):
                    current_ema = mse_w
                    first_step = False
                else:
                    current_ema = alpha * mse_w + (1.0 - alpha) * current_ema

            is_pred_anom = bool(current_ema > threshold) if np.isfinite(current_ema) else False
            stream_pred[step] = is_pred_anom

            # Candidate sample eligibility
            is_masked = bool(test_exclusion[step] or is_pred_anom)
            if not is_masked:
                recent_buffer_x.append(test_x[step])
                recent_buffer_y.append(test_y[step])
                recent_residuals.append(mse_w)
                if len(recent_buffer_x) > d_cfg["recent_window_size"]:
                    recent_buffer_x.pop(0)
                    recent_buffer_y.pop(0)
                    recent_residuals.pop(0)

            # Adaptation trigger logic
            req_action = None

            if baseline_name == "B2":
                if step > 0 and step % b2_periodic_interval == 0 and len(recent_buffer_x) >= 50:
                    req_action = "FINE_TUNE"

            elif baseline_name == "B3":
                t0_det = time.perf_counter()
                drift_flag = detector.update(mse_w)
                t_detect += time.perf_counter() - t0_det
                if drift_flag:
                    drift_detections_count += 1
                    if len(recent_buffer_x) >= 30:
                        req_action = "FINE_TUNE"

            elif baseline_name == "B4":
                t0_det = time.perf_counter()
                if not is_masked:
                    drift_flag = detector.update(mse_w)
                else:
                    drift_flag = False
                t_detect += time.perf_counter() - t0_det

                if drift_flag:
                    drift_detections_count += 1

                t0_dec = time.perf_counter()
                pers_rec = persistence.update(drift_flag, step=step)

                if len(recent_residuals) >= 20:
                    raw_measures = compute_all_distribution_measures(
                        np.array(reference_residuals), np.array(recent_residuals)
                    )
                    sev_rec = severity_calc.compute_severity(raw_measures, step=step)
                    severity_val = sev_rec["severity"]
                else:
                    raw_measures = {"ks": 0.0, "wasserstein": 0.0, "js": 0.0}
                    sev_rec = {"normalized": {"ks": 0.0, "wasserstein": 0.0, "js": 0.0}}
                    severity_val = 0.0

                deg_rec = degradation_tracker.evaluate_residual_degradation(np.array(recent_residuals), step=step)

                decision = engine.decide(
                    step=step,
                    timestamp_utc=str(timestamps[test_starts[step]]),
                    drift_detected=drift_flag,
                    detector_name="ADWIN",
                    raw_measures=raw_measures,
                    normalized_measures=sev_rec["normalized"],
                    severity=severity_val,
                    persistence_count=pers_rec["persistence_count"],
                    persistence_passed=pers_rec["is_persistent"],
                    degradation_value=deg_rec["relative_degradation"],
                    degradation_passed=deg_rec["is_significant_degradation"],
                )
                t_decision += time.perf_counter() - t0_dec

                if decision.action == AdaptationAction.STRONG_UPDATE:
                    req_action = "STRONG_UPDATE"
                elif decision.action == AdaptationAction.FINE_TUNE:
                    req_action = "FINE_TUNE"

            # Candidate Generation & Validation Gate
            if req_action is not None and len(recent_buffer_x) >= 30:
                t0_tr = time.perf_counter()
                cand_x = np.stack(recent_buffer_x, axis=0)
                cand_y = np.stack(recent_buffer_y, axis=0)
                cand_model, cand_meta, _ = trainer.create_candidate(
                    current_model, cand_x, cand_y, action_type=req_action
                )
                t_train += time.perf_counter() - t0_tr

                t0_vg = time.perf_counter()
                eval_res = val_gate.evaluate_candidate(current_model, cand_model, cand_meta)
                t_val += time.perf_counter() - t0_vg

                if eval_res.accepted:
                    current_model = cand_model
                    updates_applied += 1
                    if req_action == "STRONG_UPDATE":
                        strong_update_count += 1
                    else:
                        fine_tune_count += 1

                    for rx, ry in zip(recent_buffer_x, recent_buffer_y):
                        replay.add(rx, ry, is_anomaly_masked=False)

                    reference_residuals = list(recent_residuals)

                    # Re-score future remaining windows with accepted model
                    if step + 1 < n_steps:
                        rem_scores, _ = score_model_on_windows(
                            current_model, test_x[step + 1 :], test_y[step + 1 :], device
                        )
                        cached_scores[step + 1 :] = rem_scores

                    if scenario_id in [1, 4]:
                        unnecessary_updates += 1

        ev_metrics = compute_event_metrics(test_labels, stream_pred)
        gate_summary = val_gate.get_summary()
        tot_runtime = time.perf_counter() - sim_start_t

        return {
            "name": baseline_name,
            "event_metrics": ev_metrics,
            "updates": updates_applied,
            "fine_tune_count": fine_tune_count,
            "strong_update_count": strong_update_count,
            "drift_detections": drift_detections_count,
            "candidates_accepted": gate_summary["accepted_count"],
            "candidates_rejected": gate_summary["rejected_count"],
            "unnecessary_updates": unnecessary_updates,
            "runtime_total": tot_runtime,
            "runtime_detection": t_detect,
            "runtime_decision": t_decision,
            "runtime_training": t_train,
            "runtime_validation": t_val,
        }

    # Execute B2, B3, B4
    results["baselines"]["B2"] = run_adaptive_stream("B2")
    results["baselines"]["B3"] = run_adaptive_stream("B3")
    results["baselines"]["B4"] = run_adaptive_stream("B4")

    return results
