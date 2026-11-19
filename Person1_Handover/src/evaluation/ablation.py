"""Ablation study and anomaly masking comparison runner.

Ablations:
1. Complete B4 (Proposed Method)
2. Ablation 1: No Severity (ignores magnitude / severity weighting)
3. Ablation 2: No Persistence (instant trigger on single drift signal)
4. Ablation 3: No Degradation (triggers on distribution shift alone without detector degradation check)
5. Ablation 4: No Anomaly Masking (with vs without masking: anomaly spikes enter drift detector & update buffer)
"""
from __future__ import annotations
import copy
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
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
from src.evaluation.event_metrics import compute_event_metrics
from src.evaluation.benchmark import prepare_fast_windows, score_model_on_windows


def run_ablation_simulation(
    ablation_variant: str,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
    test_exclusion: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    norm_val_scores: np.ndarray,
    base_model: LSTMForecaster,
    replay_seed_buffer: ReplayBuffer,
    device: torch.device,
    cfg: Dict[str, Any],
    threshold: float,
    scenario_id: int,
) -> Dict[str, Any]:
    """Run a specific ablation configuration."""
    m_cfg = cfg["model"]
    a_cfg = cfg["anomaly"]
    d_cfg = cfg["drift"]
    c_cfg = cfg["continual"]

    t0_all = time.perf_counter()
    current_model = copy.deepcopy(base_model)
    val_gate = ValidationGate(val_x, val_y, device, tolerance=c_cfg["validation_tolerance"])
    replay = copy.deepcopy(replay_seed_buffer)
    trainer = CandidateTrainer(base_model, device, replay, resolve_path(cfg["output"]["model_dir"]) / "candidates")

    detector = ADWIN(delta=d_cfg["adwin_delta"])
    persistence = PersistenceTracker(window_k=d_cfg["persistence_window_k"], min_detections=d_cfg["persistence_min"])
    severity_calc = SeverityCalculator(weights=d_cfg["severity_weights"])
    severity_calc.calibrate_from_normal_validation(norm_val_scores)
    degradation_tracker = DegradationTracker(
        reference_f05=0.5,
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
    cached_scores, _ = score_model_on_windows(current_model, test_x, test_y, device)
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

    use_masking = (ablation_variant != "No_Anomaly_Masking")
    use_severity = (ablation_variant != "No_Severity")
    use_persistence = (ablation_variant != "No_Persistence")
    use_degradation = (ablation_variant != "No_Degradation")

    for step in range(n_steps):
        mse_w = float(cached_scores[step])

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

        # Anomaly Masking control
        is_masked = bool((test_exclusion[step] or is_pred_anom) and use_masking)

        # Buffer accumulation
        if not is_masked:
            recent_buffer_x.append(test_x[step])
            recent_buffer_y.append(test_y[step])
            recent_residuals.append(mse_w)
            if len(recent_buffer_x) > d_cfg["recent_window_size"]:
                recent_buffer_x.pop(0)
                recent_buffer_y.pop(0)
                recent_residuals.pop(0)

        # Drift Detection
        if not is_masked:
            drift_flag = detector.update(mse_w)
        else:
            drift_flag = False

        if drift_flag:
            drift_detections_count += 1

        # Persistence control
        if use_persistence:
            pers_rec = persistence.update(drift_flag, step=step)
            pers_passed = pers_rec["is_persistent"]
            pers_count = pers_rec["persistence_count"]
        else:
            pers_passed = drift_flag
            pers_count = 1 if drift_flag else 0

        # Severity control
        if use_severity and len(recent_residuals) >= 20:
            raw_measures = compute_all_distribution_measures(
                np.array(reference_residuals), np.array(recent_residuals)
            )
            sev_rec = severity_calc.compute_severity(raw_measures, step=step)
            severity_val = sev_rec["severity"]
        else:
            raw_measures = {"ks": 0.5, "wasserstein": 0.5, "js": 0.5}
            sev_rec = {"normalized": {"ks": 0.5, "wasserstein": 0.5, "js": 0.5}}
            severity_val = 0.70 if drift_flag else 0.0

        # Degradation control
        if use_degradation:
            deg_rec = degradation_tracker.evaluate_residual_degradation(np.array(recent_residuals), step=step)
            deg_passed = deg_rec["is_significant_degradation"]
            deg_val = deg_rec["relative_degradation"]
        else:
            deg_passed = True if drift_flag else False
            deg_val = 0.10 if drift_flag else 0.0

        decision = engine.decide(
            step=step,
            timestamp_utc=str(step),
            drift_detected=drift_flag,
            detector_name="ADWIN",
            raw_measures=raw_measures,
            normalized_measures=sev_rec["normalized"],
            severity=severity_val,
            persistence_count=pers_count,
            persistence_passed=pers_passed,
            degradation_value=deg_val,
            degradation_passed=deg_passed,
        )

        req_action = None
        if decision.action == AdaptationAction.STRONG_UPDATE:
            req_action = "STRONG_UPDATE"
        elif decision.action == AdaptationAction.FINE_TUNE:
            req_action = "FINE_TUNE"

        if req_action is not None and len(recent_buffer_x) >= 30:
            cand_x = np.stack(recent_buffer_x, axis=0)
            cand_y = np.stack(recent_buffer_y, axis=0)
            cand_model, cand_meta, _ = trainer.create_candidate(
                current_model, cand_x, cand_y, action_type=req_action
            )
            eval_res = val_gate.evaluate_candidate(current_model, cand_model, cand_meta)

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
                        current_model, test_x[step + 1 :], test_y[step + 1 :], device
                    )
                    cached_scores[step + 1 :] = rem_scores

                if scenario_id in [1, 4]:
                    unnecessary_updates += 1

    ev_metrics = compute_event_metrics(test_labels, stream_pred)
    gate_summary = val_gate.get_summary()

    return {
        "variant": ablation_variant,
        "event_metrics": ev_metrics,
        "updates": updates_applied,
        "fine_tune_count": fine_tune_count,
        "strong_update_count": strong_update_count,
        "drift_detections": drift_detections_count,
        "candidates_accepted": gate_summary["accepted_count"],
        "candidates_rejected": gate_summary["rejected_count"],
        "unnecessary_updates": unnecessary_updates,
        "runtime_total": time.perf_counter() - t0_all,
    }
