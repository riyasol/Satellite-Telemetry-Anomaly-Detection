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
from src.models.anomaly_scoring import load_forecaster, ema_reset_on_excluded, window_has_anomaly, exponential_moving_average
from src.models.lstm_forecaster import LSTMForecaster
from src.drift.streaming_detectors import ADWIN
from src.drift.distribution_measures import compute_all_distribution_measures
from src.drift.anomaly_masker import AnomalyMasker, confirm_adwin_alarm
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
    mask_dominance_threshold = float(d_cfg.get("mask_dominance_threshold", 0.75))
    min_clean_reference_samples = int(d_cfg.get("min_clean_reference_samples", 20))
    min_clean_recent_samples = int(d_cfg.get("min_clean_recent_samples", 20))
    if not 0.0 <= mask_dominance_threshold <= 1.0:
        raise ValueError("drift.mask_dominance_threshold must be between 0 and 1")
    if min_clean_reference_samples <= 0 or min_clean_recent_samples <= 0:
        raise ValueError("drift minimum clean sample counts must be positive")

    values_clean = clean_splits["values"]
    labels_clean = clean_splits["labels"]
    exclusion_clean = clean_splits["exclusion_mask"]
    timestamps = clean_splits["timestamps_ns"]
    train_bounds = clean_splits["train_bounds"].tolist()
    val_bounds = clean_splits["validation_bounds"].tolist()
    test_bounds = clean_splits["test_bounds"].tolist()

    # Generate scenario values on adaptation split (online monitoring)
    adapt_start, adapt_end = clean_splits["adaptation_bounds"].tolist()
    clean_adapt_values = values_clean[adapt_start:adapt_end].copy()
    mod_adapt_values, adapt_scenario_config = generate_scenario(
        clean_adapt_values, scenario_id, seed=cfg.get("seed", 42)
    )
    # Apply the synthetic drift only to the adaptation segment
    full_values = values_clean.copy()
    full_values[adapt_start:adapt_end] = mod_adapt_values

    # Pre-extract windows for online monitoring (adaptation segment)
    stride = int(m_cfg.get("window_stride", 10))
    adapt_starts, adapt_x, adapt_y = prepare_fast_windows(
        full_values, (adapt_start, adapt_end), m_cfg["input_window"], m_cfg["forecast_horizon"], stride=stride
    )
    val_starts, val_x, val_y = prepare_fast_windows(
        values_clean, (val_bounds[0], val_bounds[1]),
        m_cfg["input_window"], m_cfg["forecast_horizon"], stride=stride,
    )
    # Compute predicted anomaly flags using model residuals and EMA threshold
    base_model = load_forecaster(initial_checkpoint_path, device)
    # Score on adaptation windows
    adapt_scores, _ = score_model_on_windows(base_model, adapt_x, adapt_y, device)
    adapt_ema = exponential_moving_average(adapt_scores, a_cfg["ema_alpha"])
    # Threshold from validation split (unchanged)
    val_scores, _ = score_model_on_windows(base_model, val_x, val_y, device)
    val_bad = window_has_anomaly(clean_splits["exclusion_mask"], val_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=True)
    val_ema = ema_reset_on_excluded(val_scores, ~val_bad, a_cfg["ema_alpha"])
    norm_val_scores = val_ema[np.isfinite(val_ema)]
    threshold = float(np.quantile(norm_val_scores, a_cfg["threshold_quantile"]))
    # Predicted anomaly mask for adaptation windows
    adapt_pred_mask = (adapt_ema > threshold) & np.isfinite(adapt_ema)
    # adapt_pred_mask is used only for adaptation-stream online monitoring.

    test_start, test_end = test_bounds
    clean_test_values = values_clean[test_start:test_end].copy()
    mod_test_values, test_scenario_config = generate_scenario(
        clean_test_values, scenario_id, seed=cfg.get("seed", 42)
    )

    full_values = values_clean.copy()
    full_values[test_start:test_end] = mod_test_values

    # Pre-extract windows
    stride = int(m_cfg.get("window_stride", 10))
    test_starts, test_x, test_y = prepare_fast_windows(
        full_values, (test_start, test_end), m_cfg["input_window"], m_cfg["forecast_horizon"], stride=stride
    )
    # Ground-truth test labels are used only for frozen final-test offline evaluation.
    offline_test_labels = window_has_anomaly(labels_clean, test_starts, m_cfg["input_window"], m_cfg["forecast_horizon"], include_input=False)

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

    results = {
        "scenario": adapt_scenario_config.to_dict(),
        "test_scenario": test_scenario_config.to_dict(),
        "baselines": {},
    }

    # -------------------------------------------------------------
    # B1: Static LSTM Baseline
    # -------------------------------------------------------------
    b1_start_t = time.perf_counter()
    b1_raw, b1_inf_t = score_model_on_windows(base_model, test_x, test_y, device)
    b1_ema = exponential_moving_average(b1_raw, a_cfg["ema_alpha"])
    b1_pred = (b1_ema > threshold) & np.isfinite(b1_ema)
    b1_event = compute_event_metrics(offline_test_labels, b1_pred)
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
        detector = ADWIN(delta=d_cfg["adwin_delta"], max_window=d_cfg.get("adwin_max_window", 2000), min_window=d_cfg.get("adwin_min_window", 20))
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
        n_steps = len(adapt_x)
        cached_scores, _ = score_model_on_windows(current_model, adapt_x, adapt_y, device)
        predicted_mask_by_observation = np.zeros(len(values_clean), dtype=bool)
        for pos, start in enumerate(adapt_starts):
            left = int(start)
            right = min(
                len(values_clean),
                left + m_cfg["input_window"] + m_cfg["forecast_horizon"],
            )
            if pos < len(adapt_pred_mask):
                predicted_mask_by_observation[left:right] = bool(adapt_pred_mask[pos])
        current_ema = 0.0
        first_step = True
        updates_applied = 0
        fine_tune_count = 0
        strong_update_count = 0
        unnecessary_updates = 0
        drift_detections_count = 0
        confirmed_drift_count = 0
        mask_suppression_count = 0
        insufficient_evidence_count = 0
        event_log = []
        recent_buffer_x = []
        recent_buffer_y = []
        recent_residuals = []
        reference_residuals = list(norm_val_scores[:200])
        b2_periodic_interval = c_cfg.get("b2_periodic_interval", 1000)
        for step in range(n_steps):
            mse_w = float(cached_scores[step])
            alpha = float(a_cfg["ema_alpha"])
            if first_step or not np.isfinite(current_ema):
                current_ema = mse_w
                first_step = False
            else:
                current_ema = alpha * mse_w + (1.0 - alpha) * current_ema

            is_pred_anom = bool(current_ema > threshold) if np.isfinite(current_ema) else False
            eligible = not is_pred_anom
            if eligible:
                recent_buffer_x.append(adapt_x[step])
                recent_buffer_y.append(adapt_y[step])
                recent_residuals.append(mse_w)
                if len(recent_buffer_x) > d_cfg["recent_window_size"]:
                    recent_buffer_x.pop(0)
                    recent_buffer_y.pop(0)
                    recent_residuals.pop(0)

            req_action = None
            if baseline_name == "B2":
                if step > 0 and step % b2_periodic_interval == 0 and len(recent_buffer_x) >= 50:
                    req_action = "FINE_TUNE"
            elif baseline_name == "B3":
                t0_det = time.perf_counter()
                drift_flag = detector.update(mse_w, stream_index=int(adapt_starts[step]))
                t_detect += time.perf_counter() - t0_det
                if drift_flag:
                    drift_detections_count += 1
                    if len(recent_buffer_x) >= 30:
                        req_action = "FINE_TUNE"
            elif baseline_name in ("B4_original", "B4_dual"):
                t0_det = time.perf_counter()
                if baseline_name == "B4_original" and not eligible:
                    raw_alarm = False
                else:
                    raw_alarm = detector.update(mse_w, stream_index=int(adapt_starts[step]))
                t_detect += time.perf_counter() - t0_det
                if raw_alarm:
                    drift_detections_count += 1

                preliminary_confirmed = bool(raw_alarm)
                drift_confirmed = False
                suppression_reason = (
                    "predicted_anomaly_mask"
                    if baseline_name == "B4_original" and not eligible
                    else "none"
                )
                mask_fraction = 0.0
                clean_reference_count = 0
                clean_recent_count = 0
                clean_confirmation_measures = None

                if baseline_name == "B4_dual" and raw_alarm:
                    retained_indices = np.asarray(detector.last_cut_window_idx, dtype=np.int64)
                    confirmation = confirm_adwin_alarm(
                        detector.last_cut_window,
                        retained_indices,
                        predicted_mask_by_observation,
                        detector.last_cut_point,
                        mask_dominance_threshold,
                        min_clean_reference_samples,
                        min_clean_recent_samples,
                    )
                    drift_confirmed = bool(
                        preliminary_confirmed and confirmation["drift_confirmed"]
                    )
                    suppression_reason = confirmation["suppression_reason"]
                    mask_fraction = confirmation["anomaly_mask_fraction"]
                    clean_reference_count = confirmation["reference_clean_count"]
                    clean_recent_count = confirmation["recent_clean_count"]
                    if suppression_reason == "anomaly_mask_dominance":
                        mask_suppression_count += 1
                    elif suppression_reason.startswith("insufficient_clean"):
                        insufficient_evidence_count += 1
                    if drift_confirmed:
                        raw_measures = compute_all_distribution_measures(
                            confirmation["reference"],
                            confirmation["recent"],
                        )
                        clean_confirmation_measures = raw_measures

                if baseline_name == "B4_original":
                    drift_confirmed = preliminary_confirmed

                t0_dec = time.perf_counter()
                if clean_confirmation_measures is not None:
                    raw_measures = clean_confirmation_measures
                    sev_rec = severity_calc.compute_severity(raw_measures, step=step)
                    severity_val = sev_rec["severity"]
                elif len(recent_residuals) >= 20:
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
                pers_rec = persistence.update(drift_confirmed, step=step)

                decision = engine.decide(
                    step=step,
                    timestamp_utc=str(timestamps[int(adapt_starts[step])]),
                    drift_detected=drift_confirmed,
                    detector_name="ADWIN",
                    raw_measures=raw_measures,
                    normalized_measures=sev_rec["normalized"],
                    severity=severity_val,
                    persistence_count=pers_rec["persistence_count"],
                    persistence_passed=pers_rec["is_persistent"],
                    degradation_value=deg_rec["relative_degradation"],
                    degradation_passed=deg_rec["is_significant_degradation"],
                    raw_adwin_alarm=raw_alarm,
                    drift_confirmed=drift_confirmed,
                    suppression_reason=suppression_reason,
                    anomaly_mask_fraction=mask_fraction,
                    clean_reference_count=clean_reference_count,
                    clean_recent_count=clean_recent_count,
                    cut_point=detector.last_cut_point,
                )
                t_decision += time.perf_counter() - t0_dec

                evidence_confirmed = bool(drift_confirmed)
                final_confirmed = bool(
                    evidence_confirmed
                    and pers_rec["is_persistent"]
                    and decision.action != AdaptationAction.NO_UPDATE
                )
                drift_confirmed = final_confirmed
                decision.drift_confirmed = final_confirmed
                decision.reason = (
                    decision.reason
                    if final_confirmed
                    else "Raw ADWIN alarm not confirmed after persistence, "
                         "severity, degradation and decision criteria"
                )
                if drift_confirmed:
                    confirmed_drift_count += 1
                event_log.append({
                    "step": step,
                    "baseline": baseline_name,
                    "raw_adwin_alarm": raw_alarm,
                    "drift_confirmed": drift_confirmed,
                    "suppression_reason": suppression_reason,
                    "anomaly_mask_fraction": mask_fraction,
                    "clean_reference_count": clean_reference_count,
                    "clean_recent_count": clean_recent_count,
                    "action": decision.action.value,
                })
                if decision.action == AdaptationAction.STRONG_UPDATE:
                    req_action = "STRONG_UPDATE"
                elif decision.action == AdaptationAction.FINE_TUNE:
                    req_action = "FINE_TUNE"
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
                    if step + 1 < n_steps:
                        rem_scores, _ = score_model_on_windows(
                            current_model, adapt_x[step + 1 :], adapt_y[step + 1 :], device
                        )
                        cached_scores[step + 1 :] = rem_scores
                    if scenario_id in [1, 4]:
                        unnecessary_updates += 1
        frozen_model = copy.deepcopy(current_model)
        frozen_model.eval()
        test_scores, _ = score_model_on_windows(frozen_model, test_x, test_y, device)
        test_ema = exponential_moving_average(test_scores, a_cfg["ema_alpha"])
        stream_pred = (test_ema > threshold) & np.isfinite(test_ema)
        ev_metrics = compute_event_metrics(offline_test_labels, stream_pred)
        gate_summary = val_gate.get_summary()
        tot_runtime = time.perf_counter() - sim_start_t
        raw_alarm_steps = [
            row["step"] for row in event_log if row["raw_adwin_alarm"]
        ]
        confirmed_steps = [
            row["step"] for row in event_log if row["drift_confirmed"]
        ]
        scenario_start = int(adapt_scenario_config.start_step)
        first_confirmed_step = confirmed_steps[0] if confirmed_steps else None
        confirmed_before_shift = bool(
            scenario_start >= 0
            and first_confirmed_step is not None
            and first_confirmed_step * stride < scenario_start
        )
        detection_delay = (
            first_confirmed_step * stride - scenario_start
            if scenario_start >= 0
            and first_confirmed_step is not None
            and not confirmed_before_shift
            else None
        )
        return {
            "name": baseline_name,
            "event_metrics": ev_metrics,
            "updates": updates_applied,
            "fine_tune_count": fine_tune_count,
            "strong_update_count": strong_update_count,
            "drift_detections": drift_detections_count,
            "confirmed_drift_count": confirmed_drift_count,
            "mask_suppression_count": mask_suppression_count,
            "insufficient_evidence_count": insufficient_evidence_count,
            "candidates_accepted": gate_summary["accepted_count"],
            "candidates_rejected": gate_summary["rejected_count"],
            "unnecessary_updates": unnecessary_updates,
            "runtime_total": tot_runtime,
            "runtime_detection": t_detect,
            "runtime_decision": t_decision,
            "runtime_training": t_train,
            "runtime_validation": t_val,
            "event_log": event_log,
            "first_raw_alarm_step": raw_alarm_steps[0] if raw_alarm_steps else None,
            "first_confirmed_step": first_confirmed_step,
            "detection_delay_observations": detection_delay,
            "confirmed_before_shift": confirmed_before_shift,
            "false_confirmation_scenario": scenario_id in (1, 4),
            "false_confirmation_count": (
                len(confirmed_steps) if scenario_id in (1, 4) else 0
            ),
            "harmful_drift_missed": (
                scenario_id in (3, 5) and not any(
                    step * stride >= scenario_start for step in confirmed_steps
                )
            ),
        }

    # Execute B2, B3, B4_original, and B4_dual independently.
    results["baselines"]["B2"] = run_adaptive_stream("B2")
    results["baselines"]["B3"] = run_adaptive_stream("B3")
    results["baselines"]["B4_original"] = run_adaptive_stream("B4_original")
    results["baselines"]["B4_dual"] = run_adaptive_stream("B4_dual")

    results["calibration"] = {
        "threshold": threshold,
        "threshold_quantile": float(a_cfg["threshold_quantile"]),
        "clean_validation_windows": int(len(norm_val_scores)),
        "validation_mse_mean": float(np.mean(norm_val_scores)),
        "mask_dominance_threshold": mask_dominance_threshold,
    }
    return results
