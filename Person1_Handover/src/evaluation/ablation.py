
"""Ablation study with separate adaptation and final-test evaluation.

Variants:
1. Complete_B4
2. No_Severity
3. No_Persistence
4. No_Degradation
5. No_Anomaly_Masking
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Any

import numpy as np
import torch

from src.utils.config import resolve_path
from src.models.lstm_forecaster import LSTMForecaster
from src.drift.streaming_detectors import ADWIN
from src.drift.distribution_measures import compute_all_distribution_measures
from src.drift.severity import SeverityCalculator
from src.drift.persistence import PersistenceTracker
from src.drift.degradation import DegradationTracker
from src.drift.decision_engine import DecisionEngine, AdaptationAction
from src.drift.anomaly_masker import confirm_adwin_alarm
from src.continual.replay_buffer import ReplayBuffer
from src.continual.candidate_trainer import CandidateTrainer
from src.continual.validation_gate import ValidationGate
from src.evaluation.event_metrics import compute_event_metrics
from src.evaluation.benchmark import score_model_on_windows


def run_ablation_simulation(
    ablation_variant: str,
    adaptation_x: np.ndarray,
    adaptation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
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
    """Adapt on adaptation data and evaluate on untouched test data."""

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

    t0_all = time.perf_counter()

    current_model = copy.deepcopy(base_model)

    val_gate = ValidationGate(
        val_x,
        val_y,
        device,
        tolerance=c_cfg["validation_tolerance"]
    )

    replay = copy.deepcopy(replay_seed_buffer)

    trainer = CandidateTrainer(
        base_model,
        device,
        replay,
        resolve_path(cfg["output"]["model_dir"]) / "candidates"
    )

    detector = ADWIN(delta=d_cfg["adwin_delta"])

    persistence = PersistenceTracker(
        window_k=d_cfg["persistence_window_k"],
        min_detections=d_cfg["persistence_min"]
    )

    severity_calc = SeverityCalculator(
        weights=d_cfg["severity_weights"]
    )
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

    # Ablation switches
    use_masking = ablation_variant != "No_Anomaly_Masking"
    use_dual = ablation_variant in ("B4_dual", "Complete_B4")
    use_severity = ablation_variant != "No_Severity"
    use_persistence = ablation_variant != "No_Persistence"
    use_degradation = ablation_variant != "No_Degradation"

    # ---------------------------------------------------------
    # PHASE 1: ONLINE ADAPTATION
    # Only adaptation split is used in this phase.
    # ---------------------------------------------------------

    n_steps = len(adaptation_x)

    cached_scores, _ = score_model_on_windows(
        current_model,
        adaptation_x,
        adaptation_y,
        device
    )

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
    adaptation_predicted_mask = np.zeros(n_steps, dtype=bool)

    for step in range(n_steps):

        mse_w = float(cached_scores[step])

        # Causal EMA anomaly scoring.
        alpha = float(a_cfg["ema_alpha"])

        if first_step or not np.isfinite(current_ema):
            current_ema = mse_w
            first_step = False
        else:
            current_ema = (
                alpha * mse_w
                + (1.0 - alpha) * current_ema
            )

        is_pred_anom = bool(current_ema > threshold)
        adaptation_predicted_mask[step] = is_pred_anom

        # No ground-truth anomaly labels are used online.
        is_masked = bool(is_pred_anom and use_masking)

        # Collect only eligible observations.
        if not is_masked:
            recent_buffer_x.append(adaptation_x[step])
            recent_buffer_y.append(adaptation_y[step])
            recent_residuals.append(mse_w)

            max_size = d_cfg["recent_window_size"]

            if len(recent_buffer_x) > max_size:
                recent_buffer_x.pop(0)
                recent_buffer_y.pop(0)
                recent_residuals.pop(0)

        # B4_original feeds clean residuals only.
        # B4_dual feeds raw residuals and confirms alarms separately.
        if use_dual:
            raw_adwin_alarm = detector.update(
                mse_w,
                stream_index=step,
            )
        elif not is_masked:
            raw_adwin_alarm = detector.update(
                mse_w,
                stream_index=step,
            )
        else:
            raw_adwin_alarm = False

        drift_flag = bool(raw_adwin_alarm)
        confirmation_reason = (
            "predicted_anomaly_mask"
            if ablation_variant == "B4_original" and is_masked
            else "none"
        )
        mask_fraction = 0.0
        clean_reference_count = 0
        clean_recent_count = 0
        clean_confirmation_measures = None

        if use_dual and raw_adwin_alarm:
            retained_residuals = np.asarray(detector.last_cut_window, dtype=np.float64)
            retained_indices = np.asarray(detector.last_cut_window_idx, dtype=np.int64)
            confirmation = confirm_adwin_alarm(
                retained_residuals,
                retained_indices,
                adaptation_predicted_mask,
                detector.last_cut_point,
                mask_dominance_threshold,
                min_clean_reference_samples,
                min_clean_recent_samples,
            )
            drift_flag = bool(confirmation["drift_confirmed"])
            confirmation_reason = confirmation["suppression_reason"]
            mask_fraction = confirmation["anomaly_mask_fraction"]
            clean_reference_count = confirmation["reference_clean_count"]
            clean_recent_count = confirmation["recent_clean_count"]
            if drift_flag:
                clean_confirmation_measures = compute_all_distribution_measures(
                    confirmation["reference"],
                    confirmation["recent"],
                )

        if drift_flag:
            drift_detections_count += 1

        # Persistence calculation.
        if use_persistence:
            pers_rec = persistence.update(
                drift_flag,
                step=step
            )
            pers_passed = pers_rec["is_persistent"]
            pers_count = pers_rec["persistence_count"]
        else:
            pers_passed = drift_flag
            pers_count = 1 if drift_flag else 0

        # Distribution measures and severity.
        if use_dual and raw_adwin_alarm and clean_confirmation_measures is not None:
            raw_measures = clean_confirmation_measures
            sev_rec = severity_calc.compute_severity(raw_measures, step=step)
            severity_val = sev_rec["severity"]
        elif use_dual and raw_adwin_alarm:
            raw_measures = {"ks": 0.0, "wasserstein": 0.0, "js": 0.0}
            sev_rec = {"normalized": {"ks": 0.0, "wasserstein": 0.0, "js": 0.0}}
            severity_val = 0.0
        elif use_severity and len(recent_residuals) >= 20:

            raw_measures = compute_all_distribution_measures(
                np.asarray(reference_residuals),
                np.asarray(recent_residuals)
            )

            sev_rec = severity_calc.compute_severity(
                raw_measures,
                step=step
            )

            severity_val = sev_rec["severity"]

        else:
            raw_measures = {
                "ks": 0.5,
                "wasserstein": 0.5,
                "js": 0.5
            }

            sev_rec = {
                "normalized": {
                    "ks": 0.5,
                    "wasserstein": 0.5,
                    "js": 0.5
                }
            }

            severity_val = 0.70 if drift_flag else 0.0

        # Residual degradation.
        if use_degradation:

            deg_rec = degradation_tracker.evaluate_residual_degradation(
                np.asarray(recent_residuals),
                step=step
            )

            deg_passed = deg_rec["is_significant_degradation"]
            deg_val = deg_rec["relative_degradation"]

        else:
            deg_passed = bool(drift_flag)
            deg_val = 0.10 if drift_flag else 0.0

        # Decision engine.
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
            raw_adwin_alarm=bool(raw_adwin_alarm),
            drift_confirmed=bool(drift_flag),
            suppression_reason=confirmation_reason,
            anomaly_mask_fraction=mask_fraction,
            clean_reference_count=clean_reference_count,
            clean_recent_count=clean_recent_count,
            cut_point=detector.last_cut_point,
        )
        decision.drift_confirmed = bool(
            drift_flag
            and pers_passed
            and decision.action != AdaptationAction.NO_UPDATE
        )

        if decision.action == AdaptationAction.STRONG_UPDATE:
            req_action = "STRONG_UPDATE"
        elif decision.action == AdaptationAction.FINE_TUNE:
            req_action = "FINE_TUNE"
        else:
            req_action = None

        # Candidate training uses adaptation data only.
        if req_action is not None and len(recent_buffer_x) >= 30:

            cand_x = np.stack(recent_buffer_x, axis=0)
            cand_y = np.stack(recent_buffer_y, axis=0)

            cand_model, cand_meta, _ = trainer.create_candidate(
                current_model,
                cand_x,
                cand_y,
                action_type=req_action
            )

            # Validation gate decides candidate acceptance.
            eval_res = val_gate.evaluate_candidate(
                current_model,
                cand_model,
                cand_meta
            )

            if eval_res.accepted:

                current_model = cand_model
                updates_applied += 1

                if req_action == "STRONG_UPDATE":
                    strong_update_count += 1
                else:
                    fine_tune_count += 1

                for rx, ry in zip(recent_buffer_x, recent_buffer_y):
                    replay.add(
                        rx,
                        ry,
                        is_anomaly_masked=False
                    )

                reference_residuals = list(
                    recent_residuals[-200:]
                )

                # Re-score only the remaining adaptation stream.
                if step + 1 < n_steps:

                    rem_scores, _ = score_model_on_windows(
                        current_model,
                        adaptation_x[step + 1:],
                        adaptation_y[step + 1:],
                        device
                    )

                    cached_scores[step + 1:] = rem_scores

                if scenario_id in [1, 4]:
                    unnecessary_updates += 1

    # Freeze the adapted model.
    frozen_model = copy.deepcopy(current_model)
    frozen_model.eval()

    # ---------------------------------------------------------
    # PHASE 2: FINAL TEST
    # No adaptation, training, replay updates or model changes.
    # ---------------------------------------------------------

    test_scores, _ = score_model_on_windows(
        frozen_model,
        test_x,
        test_y,
        device
    )

    test_predictions = np.zeros(
        len(test_scores),
        dtype=bool
    )

    test_ema = 0.0
    test_first = True

    for i, score in enumerate(test_scores):

        score = float(score)

        if test_first or not np.isfinite(test_ema):
            test_ema = score
            test_first = False
        else:
            test_ema = (
                float(a_cfg["ema_alpha"]) * score
                + (1.0 - float(a_cfg["ema_alpha"])) * test_ema
            )

        test_predictions[i] = bool(test_ema > threshold)

    # Ground-truth test labels are used only for offline metrics.
    ev_metrics = compute_event_metrics(
        test_labels,
        test_predictions
    )

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
        "adaptation_steps": n_steps,
        "final_test_steps": len(test_x),
        "test_evaluated_after_adaptation": True,
        "runtime_total": time.perf_counter() - t0_all
    }
