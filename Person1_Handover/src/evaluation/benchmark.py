"""Reproducible B1-B4 evaluation with a separate adaptation and held-out test stream."""
from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.continual.candidate_trainer import CandidateTrainer
from src.continual.replay_buffer import ReplayBuffer
from src.continual.validation_gate import ValidationGate
from src.drift.decision_engine import AdaptationAction, DecisionEngine
from src.drift.degradation import DegradationTracker
from src.drift.distribution_measures import compute_all_distribution_measures
from src.drift.anomaly_masker import confirm_adwin_alarm
from src.drift.persistence import PersistenceTracker
from src.drift.severity import SeverityCalculator
from src.drift.streaming_detectors import ADWIN
from src.evaluation.event_metrics import compute_event_metrics
from src.models.anomaly_scoring import ema_reset_on_excluded, window_has_anomaly
from src.models.lstm_forecaster import LSTMForecaster
from src.scenarios.drift_generator import generate_scenario, inject_anomaly_pulses
from src.utils.config import resolve_path


def prepare_fast_windows(
    values: np.ndarray,
    bounds: Tuple[int, int],
    input_window: int = 250,
    forecast_horizon: int = 10,
    stride: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract input/target arrays and their input-start indices."""
    start_bound, end_bound = bounds
    if input_window <= 0 or forecast_horizon <= 0 or stride <= 0:
        raise ValueError("Window sizes and stride must be positive.")
    if start_bound < 0 or end_bound > len(values) or start_bound >= end_bound:
        raise ValueError("Invalid split bounds.")

    last_start = end_bound - input_window - forecast_horizon
    starts = np.arange(start_bound, last_start + 1, stride, dtype=np.int64)
    if not len(starts):
        raise ValueError("Split is too short for configured input and horizon.")

    channels = values.shape[1]
    x_arr = np.empty((len(starts), input_window, channels), dtype=np.float32)
    y_arr = np.empty((len(starts), forecast_horizon, channels), dtype=np.float32)
    for row, start in enumerate(starts):
        split = int(start) + input_window
        x_arr[row] = values[start:split]
        y_arr[row] = values[split:split + forecast_horizon]
    return starts, x_arr, y_arr


def score_model_on_windows(
    model: LSTMForecaster,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, float]:
    """Score forecast windows with MSE and return inference time."""
    if len(x_arr) != len(y_arr):
        raise ValueError("Input and target window counts do not match.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()
    start_time = time.perf_counter()
    scores = np.empty(len(x_arr), dtype=np.float32)
    with torch.no_grad():
        for batch_start in range(0, len(x_arr), batch_size):
            batch_end = batch_start + batch_size
            x = torch.from_numpy(x_arr[batch_start:batch_end]).to(device)
            y = torch.from_numpy(y_arr[batch_start:batch_end]).to(device)
            prediction = model(x)
            scores[batch_start:batch_end] = torch.mean(
                (prediction - y) ** 2, dim=(1, 2)
            ).cpu().numpy()
    return scores, time.perf_counter() - start_time


def _count_injected_anomaly_window_overlaps(
    scenario_data: Dict[str, Any],
    window_steps: list[int],
    input_window: int,
    forecast_horizon: int,
) -> Optional[int]:
    """Audit pulse overlap after adaptation without using labels in online decisions."""
    anomaly_points = scenario_data.get("adaptation_anomaly_points")
    starts = scenario_data.get("adaptation_starts")
    if anomaly_points is None or starts is None:
        return None
    if not window_steps:
        return 0
    selected_starts = np.asarray(starts, dtype=np.int64)[window_steps]
    contaminated = window_has_anomaly(
        np.asarray(anomaly_points, dtype=bool),
        selected_starts,
        input_window,
        forecast_horizon,
        include_input=True,
    )
    return int(np.count_nonzero(contaminated))


def _calibrate_anomaly_threshold(
    scores: np.ndarray,
    eligible: np.ndarray,
    alpha: float,
    quantile: float,
) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    if scores.ndim != 1 or eligible.ndim != 1 or len(scores) != len(eligible):
        raise ValueError("Threshold scores and eligibility mask must be aligned 1-D arrays.")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("EMA alpha must be finite and in (0, 1].")
    if not np.isfinite(quantile) or not 0.0 < float(quantile) < 1.0:
        raise ValueError("Threshold quantile must be finite and in (0, 1).")

    eligible = eligible & np.isfinite(scores)
    ema_scores = ema_reset_on_excluded(scores, eligible, float(alpha))
    clean_ema_scores = ema_scores[np.isfinite(ema_scores)]
    if not len(clean_ema_scores):
        raise ValueError("No eligible finite scores are available for threshold calibration.")
    return float(np.quantile(clean_ema_scores, float(quantile)))


def _longest_clean_interval(
    exclusion_mask: np.ndarray,
    start: int,
    end: int,
    min_length: int,
) -> Tuple[int, int]:
    clean = ~np.asarray(exclusion_mask[start:end], dtype=bool)
    padded = np.r_[False, clean, False].astype(np.int8)
    run_starts = np.flatnonzero(np.diff(padded) == 1)
    run_ends = np.flatnonzero(np.diff(padded) == -1)
    runs = [
        (start + int(left), start + int(right))
        for left, right in zip(run_starts, run_ends)
    ]
    if not runs:
        raise ValueError("No clean validation interval is available for adaptation.")
    result = max(runs, key=lambda run: run[1] - run[0])
    if result[1] - result[0] < min_length:
        raise ValueError(
            "The clean validation interval is too short for the configured "
            "adaptation stream and periodic baseline."
        )
    return result


class SustainedElevationTracker:
    """Distinguish a persistent high-residual regime from short masked episodes."""

    def __init__(self, minimum_windows: int):
        if minimum_windows <= 0:
            raise ValueError("minimum_windows must be positive.")
        self.minimum_windows = int(minimum_windows)
        self.consecutive_windows = 0

    def update(self, score: float, threshold: float) -> bool:
        if not np.isfinite(score) or not np.isfinite(threshold):
            raise ValueError("Elevation scores and threshold must be finite.")
        self.consecutive_windows = (
            self.consecutive_windows + 1 if score > threshold else 0
        )
        return self.consecutive_windows >= self.minimum_windows


def prepare_scenario_data(
    scenario_id: int,
    cfg: Dict[str, Any],
    clean_splits: Dict[str, Any],
    base_model: LSTMForecaster,
    device: torch.device,
    inject_anomalies: bool = False,
) -> Dict[str, Any]:
    """Build disjoint calibration, adaptation, adaptation-validation, and test data.

    Ground-truth annotations are used only to select clean calibration/adaptation
    windows and to compute final offline metrics. They are not passed to the
    streaming adaptation loop.
    """
    model_cfg = cfg["model"]
    anomaly_cfg = cfg["anomaly"]
    continual_cfg = cfg["continual"]
    experiment_cfg = cfg.get("experiments", {})
    values = np.asarray(clean_splits["values"], dtype=np.float32)
    labels = np.asarray(clean_splits["labels"], dtype=np.uint8)
    exclusion = np.asarray(clean_splits["exclusion_mask"], dtype=bool)
    timestamps = np.asarray(clean_splits["timestamps_ns"], dtype=np.int64)
    input_window = int(model_cfg["input_window"])
    horizon = int(model_cfg["forecast_horizon"])
    stride = int(model_cfg.get("window_stride", 1))
    batch_size = int(model_cfg["batch_size"])

    val_start, val_end = map(int, clean_splits["validation_bounds"])
    test_start, test_end = map(int, clean_splits["test_bounds"])
    adaptation_fraction = float(
        continual_cfg.get("adaptation_fraction_of_validation", 0.5)
    )
    if not 0.0 < adaptation_fraction < 1.0:
        raise ValueError("adaptation_fraction_of_validation must be in (0, 1).")
    calibration_end = val_start + int(
        (val_end - val_start) * (1.0 - adaptation_fraction)
    )
    if calibration_end <= val_start or calibration_end >= val_end:
        raise ValueError("Validation split is too short to partition.")

    gate_starts, gate_x, gate_y = prepare_fast_windows(
        values,
        (val_start, calibration_end),
        input_window,
        horizon,
        stride,
    )
    gate_excluded = window_has_anomaly(
        exclusion, gate_starts, input_window, horizon, include_input=True
    )
    gate_scores, _ = score_model_on_windows(
        base_model, gate_x, gate_y, device, batch_size
    )
    gate_ema = ema_reset_on_excluded(
        gate_scores, ~gate_excluded, float(anomaly_cfg["ema_alpha"])
    )
    normal_ema_scores = gate_ema[np.isfinite(gate_ema)]
    normal_raw_scores = gate_scores[(~gate_excluded) & np.isfinite(gate_scores)]
    valid_gate_windows = (~gate_excluded) & np.isfinite(gate_scores)
    if not len(normal_ema_scores) or not valid_gate_windows.any():
        raise ValueError("No eligible normal calibration/validation windows remain.")
    threshold = float(
        np.quantile(normal_ema_scores, float(anomaly_cfg["threshold_quantile"]))
    )
    gate_all_x = gate_x
    gate_all_y = gate_y
    gate_x = gate_x[valid_gate_windows]
    gate_y = gate_y[valid_gate_windows]

    drift_cfg = cfg["drift"]
    min_rows_for_stream = (
        input_window
        + horizon
        + max(
            int(experiment_cfg.get("minimum_adaptation_windows", 40)),
            int(continual_cfg.get("b2_periodic_interval", 1000)),
        ) * stride
    )
    candidate_validation_fraction = float(
        continual_cfg.get("candidate_validation_fraction_of_adaptation", 0.25)
    )
    if not 0.0 < candidate_validation_fraction < 1.0:
        raise ValueError(
            "candidate_validation_fraction_of_adaptation must be in (0, 1)."
        )
    minimum_candidate_validation_windows = int(
        experiment_cfg.get("minimum_candidate_validation_windows", 20)
    )
    if minimum_candidate_validation_windows <= 0:
        raise ValueError("minimum_candidate_validation_windows must be positive.")
    min_rows_for_candidate_validation = (
        input_window
        + horizon
        + minimum_candidate_validation_windows * stride
    )
    min_rows_for_stream = max(
        math.ceil(min_rows_for_stream / (1.0 - candidate_validation_fraction)),
        math.ceil(
            min_rows_for_candidate_validation / candidate_validation_fraction
        ),
    )
    adaptation_bounds = _longest_clean_interval(
        exclusion, calibration_end, val_end, min_rows_for_stream
    )
    adaptation_validation_start = adaptation_bounds[0] + int(
        (adaptation_bounds[1] - adaptation_bounds[0])
        * (1.0 - candidate_validation_fraction)
    )
    adaptation_stream_bounds = (
        adaptation_bounds[0],
        adaptation_validation_start,
    )
    adaptation_validation_bounds = (
        adaptation_validation_start,
        adaptation_bounds[1],
    )
    adaptation_clean = values[
        adaptation_stream_bounds[0]:adaptation_stream_bounds[1]
    ].copy()
    adaptation_validation_clean = values[
        adaptation_validation_bounds[0]:adaptation_validation_bounds[1]
    ].copy()
    scenario_parameters = experiment_cfg.get("scenarios", {}).get(
        str(scenario_id), {}
    )
    adaptation_values, scenario = generate_scenario(
        adaptation_clean,
        scenario_id,
        seed=int(cfg.get("seed", 42)),
        scenario_parameters=scenario_parameters,
    )
    adaptation_validation_values, _ = generate_scenario(
        adaptation_validation_clean,
        scenario_id,
        seed=int(cfg.get("seed", 42)) + 1,
        scenario_parameters=scenario_parameters,
    )
    adaptation_anomaly_points = np.zeros(len(adaptation_values), dtype=np.uint8)
    if inject_anomalies and scenario.start_step >= 0:
        pulse_cfg = experiment_cfg.get("masking_ablation", {})
        adaptation_values, adaptation_anomaly_points = inject_anomaly_pulses(
            adaptation_values,
            drift_start_step=scenario.start_step,
            stride=stride,
            channels=pulse_cfg.get("anomaly_channels"),
            magnitude=float(pulse_cfg.get("anomaly_magnitude", 8.0)),
            pulse_length=int(pulse_cfg.get("anomaly_pulse_length", 3)),
            offsets_in_strides=pulse_cfg.get("anomaly_offsets_in_strides"),
        )
    adaptation_starts, adaptation_x, adaptation_y = prepare_fast_windows(
        adaptation_values,
        (0, len(adaptation_values)),
        input_window,
        horizon,
        stride,
    )
    adaptation_validation_starts, adaptation_validation_x, adaptation_validation_y = (
        prepare_fast_windows(
            adaptation_validation_values,
            (0, len(adaptation_validation_values)),
            input_window,
            horizon,
            stride,
        )
    )

    test_clean = values[test_start:test_end].copy()
    evaluation_segment, _ = generate_scenario(
        test_clean,
        scenario_id,
        seed=int(cfg.get("seed", 42)),
        scenario_parameters=scenario_parameters,
    )
    evaluation_anomaly_points = np.zeros(len(evaluation_segment), dtype=np.uint8)
    if inject_anomalies and scenario.start_step >= 0:
        pulse_cfg = experiment_cfg.get("masking_ablation", {})
        evaluation_segment, evaluation_anomaly_points = inject_anomaly_pulses(
            evaluation_segment,
            drift_start_step=scenario.start_step,
            stride=stride,
            channels=pulse_cfg.get("anomaly_channels"),
            magnitude=float(pulse_cfg.get("anomaly_magnitude", 8.0)),
            pulse_length=int(pulse_cfg.get("anomaly_pulse_length", 3)),
            offsets_in_strides=pulse_cfg.get("anomaly_offsets_in_strides"),
        )
    evaluation_values = values.copy()
    evaluation_values[test_start:test_end] = evaluation_segment
    evaluation_starts, evaluation_x, evaluation_y = prepare_fast_windows(
        evaluation_values,
        (test_start, test_end),
        input_window,
        horizon,
        stride,
    )
    evaluation_labels = labels.copy()
    evaluation_labels[test_start:test_end] = np.maximum(
        evaluation_labels[test_start:test_end], evaluation_anomaly_points
    )
    evaluation_true = window_has_anomaly(
        evaluation_labels,
        evaluation_starts,
        input_window,
        horizon,
        include_input=False,
    )
    ignore_mask = np.asarray(
        clean_splits.get("evaluation_ignore_mask", np.zeros(len(values))),
        dtype=bool,
    )
    evaluation_valid = ~window_has_anomaly(
        ignore_mask,
        evaluation_starts,
        input_window,
        horizon,
        include_input=False,
    )

    gaps_seconds = (
        np.diff(timestamps[adaptation_stream_bounds[0]:adaptation_stream_bounds[1]])
        / 1e9
    )
    nominal_gap = float(np.median(gaps_seconds)) if len(gaps_seconds) else 30.0
    window_seconds = max(nominal_gap * stride, 1.0)
    pre_margin_windows = int(
        math.ceil(float(drift_cfg.get("pre_anomaly_margin_seconds", 0)) / window_seconds)
    )
    post_margin_windows = int(
        math.ceil(float(drift_cfg.get("post_anomaly_margin_seconds", 0)) / window_seconds)
    )
    drift_start_window = (
        int(np.searchsorted(adaptation_starts, scenario.start_step, side="left"))
        if scenario.start_step >= 0
        else None
    )

    scenario_dict = scenario.to_dict()
    scenario_dict.update({
        "adaptation_bounds": [
            int(adaptation_stream_bounds[0]),
            int(adaptation_stream_bounds[1]),
        ],
        "adaptation_validation_bounds": [
            int(adaptation_validation_bounds[0]),
            int(adaptation_validation_bounds[1]),
        ],
        "calibration_bounds": [int(val_start), int(calibration_end)],
        "evaluation_bounds": [int(test_start), int(test_end)],
        "adaptation_drift_start_window": drift_start_window,
        "adaptation_anomaly_points": int(adaptation_anomaly_points.sum()),
        "evaluation_anomaly_points": int(evaluation_anomaly_points.sum()),
        "scenario_parameters": scenario_parameters,
        "masking_ablation_parameters": (
            experiment_cfg.get("masking_ablation", {}) if inject_anomalies else {}
        ),
    })
    return {
        "scenario": scenario_dict,
        "scenario_config": scenario,
        "threshold": threshold,
        "normal_validation_ema_scores": normal_ema_scores,
        "normal_validation_raw_scores": normal_raw_scores,
        "gate_all_x": gate_all_x,
        "gate_all_y": gate_all_y,
        "gate_valid_mask": valid_gate_windows,
        "gate_x": gate_x,
        "gate_y": gate_y,
        "adaptation_validation_x": adaptation_validation_x,
        "adaptation_validation_y": adaptation_validation_y,
        "adaptation_validation_starts": adaptation_validation_starts,
        "adaptation_starts": adaptation_starts,
        "adaptation_x": adaptation_x,
        "adaptation_y": adaptation_y,
        "adaptation_drift_start_window": drift_start_window,
        "evaluation_x": evaluation_x,
        "evaluation_y": evaluation_y,
        "evaluation_starts": evaluation_starts,
        "evaluation_true": evaluation_true,
        "evaluation_valid": evaluation_valid,
        "pre_margin_windows": pre_margin_windows,
        "post_margin_windows": post_margin_windows,
        "adaptation_anomaly_points": adaptation_anomaly_points,
        "evaluation_anomaly_points": evaluation_anomaly_points,
    }


def _evaluate_frozen_model(
    model: LSTMForecaster,
    scenario_data: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, Any], np.ndarray, float]:
    scores, inference_seconds = score_model_on_windows(
        model,
        scenario_data["evaluation_x"],
        scenario_data["evaluation_y"],
        device,
        int(cfg["model"]["batch_size"]),
    )
    ema = ema_reset_on_excluded(
        scores,
        np.ones(len(scores), dtype=bool),
        float(cfg["anomaly"]["ema_alpha"]),
    )
    evaluation_threshold = (
        float(scenario_data["threshold"]) if threshold is None else float(threshold)
    )
    predictions = (ema > evaluation_threshold) & np.isfinite(ema)
    metrics = compute_event_metrics(
        scenario_data["evaluation_true"],
        predictions,
        tolerance_steps=int(cfg.get("evaluation", {}).get("event_tolerance_steps", 2)),
        valid_mask=scenario_data["evaluation_valid"],
    )
    return metrics, predictions, inference_seconds


def run_adaptive_simulation(
    baseline_name: str,
    scenario_id: int,
    scenario_data: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    base_model: LSTMForecaster,
    replay_seed_buffer: ReplayBuffer,
    ablation_variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Adapt on the validation stream, then evaluate once on the frozen test stream."""
    simulation_started = time.perf_counter()
    if baseline_name not in {"B2", "B3", "B4"}:
        raise ValueError(f"Unsupported adaptive baseline: {baseline_name}")
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_cfg = cfg["model"]
    anomaly_cfg = cfg["anomaly"]
    drift_cfg = cfg["drift"]
    continual_cfg = cfg["continual"]
    experiment_cfg = cfg.get("experiments", {})
    minimum_candidate_windows = int(
        experiment_cfg.get("minimum_candidate_windows", 30)
    )
    periodic_minimum_windows = int(
        experiment_cfg.get("periodic_minimum_candidate_windows", 50)
    )
    minimum_distribution_samples = int(
        experiment_cfg.get("minimum_distribution_samples", 20)
    )
    current_model = copy.deepcopy(base_model).to(device)
    current_threshold = float(scenario_data["threshold"])
    validation_gate = ValidationGate(
        scenario_data["gate_x"],
        scenario_data["gate_y"],
        device,
        tolerance=float(continual_cfg["validation_tolerance"]),
        batch_size=int(model_cfg["batch_size"]),
        adaptation_val_x=scenario_data["adaptation_validation_x"],
        adaptation_val_y=scenario_data["adaptation_validation_y"],
    )
    replay = copy.deepcopy(replay_seed_buffer)
    namespace = (
        f"mission1_s{scenario_id}_{baseline_name.lower()}"
        if ablation_variant is None
        else f"mission1_s{scenario_id}_{baseline_name.lower()}_{ablation_variant.lower()}"
    )
    trainer = CandidateTrainer(
        current_model,
        device,
        replay,
        resolve_path(cfg["output"]["model_dir"]) / "candidates",
        candidate_namespace=namespace,
    )

    detector = ADWIN(delta=float(drift_cfg["adwin_delta"]))
    persistence = PersistenceTracker(
        window_k=int(drift_cfg["persistence_window_k"]),
        min_detections=int(drift_cfg["persistence_min"]),
    )
    severity_calculator = SeverityCalculator(
        weights=drift_cfg["severity_weights"]
    )
    severity_calculator.calibrate_from_normal_validation(
        scenario_data["normal_validation_raw_scores"]
    )
    reference_residuals = list(
        scenario_data["normal_validation_raw_scores"][
            :int(drift_cfg["reference_window_size"])
        ]
    )
    if not reference_residuals:
        raise ValueError("No normal validation residuals available for drift reference.")
    degradation = DegradationTracker(
        degradation_threshold_delta_q=float(
            drift_cfg["degradation_threshold_delta_q"]
        ),
        reference_mse=max(
            1e-8,
            float(np.mean(np.square(scenario_data["normal_validation_raw_scores"]))),
        ),
    )
    engine = DecisionEngine(
        theta_low=float(drift_cfg["theta_low"]),
        theta_high=float(drift_cfg["theta_high"]),
        persistence_min=int(drift_cfg["persistence_min"]),
        degradation_threshold=float(drift_cfg["degradation_threshold_delta_q"]),
    )

    use_masking = ablation_variant != "No_Anomaly_Masking"
    use_severity = ablation_variant != "No_Severity"
    use_persistence = ablation_variant != "No_Persistence"
    use_degradation = ablation_variant != "No_Degradation"
    mask_dominance_threshold = float(
        drift_cfg.get("mask_dominance_threshold", 0.75)
    )
    min_clean_reference_samples = int(
        drift_cfg.get("min_clean_reference_samples", 20)
    )
    min_clean_recent_samples = int(
        drift_cfg.get("min_clean_recent_samples", 20)
    )
    if not 0.0 <= mask_dominance_threshold <= 1.0:
        raise ValueError("drift.mask_dominance_threshold must be between 0 and 1.")
    if min_clean_reference_samples <= 0 or min_clean_recent_samples <= 0:
        raise ValueError("Minimum clean ADWIN sample counts must be positive.")
    adaptation_x = scenario_data["adaptation_x"]
    adaptation_y = scenario_data["adaptation_y"]
    cached_scores, adaptation_inference_seconds = score_model_on_windows(
        current_model,
        adaptation_x,
        adaptation_y,
        device,
        int(model_cfg["batch_size"]),
    )

    recent_entries = []
    recent_residuals = []
    ema_state = None
    adaptation_ema_state = None
    elevation_tracker = SustainedElevationTracker(
        int(drift_cfg.get("persistent_elevation_min_windows", 100))
    )
    blocked_until = -1
    update_counts = {"FINE_TUNE": 0, "STRONG_UPDATE": 0}
    request_count = 0
    candidate_training_window_steps = []
    accepted_replay_window_steps = []
    drift_alarm_steps = []
    confirmed_drift_steps = []
    mask_suppression_count = 0
    insufficient_evidence_count = 0
    masked_window_indices = set()
    predicted_mask_by_stream_index = np.zeros(len(cached_scores), dtype=bool)
    first_decision_step = None
    first_alarm_after_drift = None
    training_seconds = 0.0
    validation_seconds = 0.0
    detection_seconds = 0.0
    decision_seconds = 0.0
    last_candidate_attempt_step = None
    candidate_cooldown_windows = max(
        1, int(drift_cfg["recent_window_size"])
    )
    trace = {
        "raw_mse": cached_scores.copy(),
        "ema_score": np.full(len(cached_scores), np.nan, dtype=np.float32),
        "threshold": np.full(
            len(cached_scores), current_threshold, dtype=np.float32
        ),
        "predicted_anomaly": np.zeros(len(cached_scores), dtype=np.uint8),
        "sustained_elevation": np.zeros(len(cached_scores), dtype=np.uint8),
        "masked": np.zeros(len(cached_scores), dtype=np.uint8),
        "raw_adwin_alarm": np.zeros(len(cached_scores), dtype=np.uint8),
        "drift_alarm": np.zeros(len(cached_scores), dtype=np.uint8),
        "adwin_cut_stream_index": np.full(
            len(cached_scores), -1, dtype=np.int64
        ),
        "mask_dominance_fraction": np.zeros(len(cached_scores), dtype=np.float32),
        "suppression_reason": np.full(len(cached_scores), "none", dtype=object),
        "severity": np.zeros(len(cached_scores), dtype=np.float32),
        "ks": np.zeros(len(cached_scores), dtype=np.float32),
        "wasserstein": np.zeros(len(cached_scores), dtype=np.float32),
        "js": np.zeros(len(cached_scores), dtype=np.float32),
        "persistence_count": np.zeros(len(cached_scores), dtype=np.int32),
        "persistence_passed": np.zeros(len(cached_scores), dtype=np.uint8),
        "relative_degradation": np.zeros(len(cached_scores), dtype=np.float32),
        "degradation_passed": np.zeros(len(cached_scores), dtype=np.uint8),
        "action": np.full(len(cached_scores), "NO_UPDATE", dtype=object),
        "accepted_update": np.zeros(len(cached_scores), dtype=np.uint8),
    }
    drift_start_window = scenario_data["adaptation_drift_start_window"]

    if ablation_variant == "No_Anomaly_Masking":
        pre_margin = post_margin = 0
    else:
        pre_margin = int(scenario_data["pre_margin_windows"])
        post_margin = int(scenario_data["post_margin_windows"])

    def update_b4_detector(step: int, score: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        nonlocal detection_seconds
        nonlocal mask_suppression_count, insufficient_evidence_count

        started = time.perf_counter()
        raw_alarm = detector.update(score, stream_index=step)
        detection_seconds += time.perf_counter() - started
        if not raw_alarm:
            return False, None

        drift_alarm_steps.append(step)
        trace["raw_adwin_alarm"][step] = 1
        confirmation = confirm_adwin_alarm(
            detector.last_cut_window,
            np.asarray(detector.last_cut_window_idx, dtype=np.int64),
            predicted_mask_by_stream_index,
            detector.last_cut_point,
            mask_dominance_threshold,
            min_clean_reference_samples,
            min_clean_recent_samples,
        )
        trace["mask_dominance_fraction"][step] = float(
            confirmation["anomaly_mask_fraction"]
        )
        trace["suppression_reason"][step] = confirmation["suppression_reason"]
        cut_point = int(confirmation["cut_point"])
        if 0 <= cut_point < len(detector.last_cut_window_idx):
            trace["adwin_cut_stream_index"][step] = int(
                detector.last_cut_window_idx[cut_point]
            )
        if confirmation["drift_confirmed"]:
            confirmed_drift_steps.append(step)
        elif confirmation["suppression_reason"] == "anomaly_mask_dominance":
            mask_suppression_count += 1
        elif confirmation["suppression_reason"].startswith("insufficient_clean"):
            insufficient_evidence_count += 1
        return bool(confirmation["drift_confirmed"]), confirmation

    for step in range(len(adaptation_x)):
        score = float(cached_scores[step])
        alpha = float(anomaly_cfg["ema_alpha"])
        adaptation_ema_state = (
            score
            if adaptation_ema_state is None
            else alpha * score + (1.0 - alpha) * adaptation_ema_state
        )
        trace["threshold"][step] = current_threshold
        sustained_elevation = elevation_tracker.update(
            adaptation_ema_state, current_threshold
        )
        trace["sustained_elevation"][step] = int(sustained_elevation)

        if use_masking and step <= blocked_until and not sustained_elevation:
            ema_state = None
            trace["masked"][step] = 1
            predicted_mask_by_stream_index[step] = True
            masked_window_indices.add(step)
            continue

        if sustained_elevation:
            predicted_mask_by_stream_index[step] = False
            trace["masked"][step] = 0
            masked_window_indices.discard(step)

        ema_state = score if ema_state is None else alpha * score + (1.0 - alpha) * ema_state
        trace["ema_score"][step] = ema_state
        is_predicted_anomaly = bool(ema_state > current_threshold)
        trace["predicted_anomaly"][step] = int(is_predicted_anomaly)

        if use_masking and is_predicted_anomaly and not sustained_elevation:
            pre_start = max(0, step - pre_margin)
            recent_entries = [
                entry for entry in recent_entries if entry["step"] < pre_start
            ]
            recent_residuals = [entry["score"] for entry in recent_entries]
            mask_end = min(len(cached_scores), step + post_margin + 1)
            predicted_mask_by_stream_index[pre_start:mask_end] = True
            trace["masked"][pre_start:mask_end] = 1
            masked_window_indices.update(range(pre_start, mask_end))
            if baseline_name != "B4":
                detector.reset()
            persistence.reset()
            blocked_until = step + post_margin
            ema_state = None
            continue

        recent_entries.append({
            "step": step,
            "x": adaptation_x[step],
            "y": adaptation_y[step],
            "score": score,
        })
        max_recent = int(drift_cfg["recent_window_size"])
        if len(recent_entries) > max_recent:
            recent_entries.pop(0)
        recent_residuals = [entry["score"] for entry in recent_entries]

        request_action = None
        drift_flag = False
        if baseline_name == "B2":
            interval = int(continual_cfg["b2_periodic_interval"])
            if (
                step > 0
                and step % interval == 0
                and len(recent_entries) >= periodic_minimum_windows
            ):
                request_action = "FINE_TUNE"
        elif baseline_name == "B4":
            drift_flag, b4_confirmation = update_b4_detector(step, score)
            if drift_flag:
                trace["drift_alarm"][step] = 1
                if (
                    drift_start_window is not None
                    and step >= drift_start_window
                    and first_alarm_after_drift is None
                ):
                    first_alarm_after_drift = step
        else:
            started = time.perf_counter()
            drift_flag = detector.update(score, stream_index=step)
            detection_seconds += time.perf_counter() - started
            if drift_flag:
                drift_alarm_steps.append(step)
                confirmed_drift_steps.append(step)
                trace["raw_adwin_alarm"][step] = 1
                trace["drift_alarm"][step] = 1
                if (
                    drift_start_window is not None
                    and step >= drift_start_window
                    and first_alarm_after_drift is None
                ):
                    first_alarm_after_drift = step

        if baseline_name == "B3":
            if drift_flag and len(recent_entries) >= minimum_candidate_windows:
                request_action = "FINE_TUNE"
        elif baseline_name == "B4":
            started = time.perf_counter()
            decision_drift_evidence = bool(drift_flag or sustained_elevation)
            persistence_record = persistence.update(
                decision_drift_evidence, step=step
            )
            distribution_reference = reference_residuals
            distribution_recent = recent_residuals
            if drift_flag and b4_confirmation is not None:
                distribution_reference = b4_confirmation["reference"]
                distribution_recent = b4_confirmation["recent"]
            if len(distribution_recent) >= minimum_distribution_samples:
                measures = compute_all_distribution_measures(
                    np.asarray(distribution_reference),
                    np.asarray(distribution_recent),
                )
                severity_record = severity_calculator.compute_severity(
                    measures, step=step
                )
                severity = float(severity_record["severity"])
                normalized = severity_record["normalized"]
            else:
                measures = {"ks": 0.0, "wasserstein": 0.0, "js": 0.0}
                normalized = {"ks": 0.0, "wasserstein": 0.0, "js": 0.0}
                severity = 0.0

            trace["ks"][step] = float(measures["ks"])
            trace["wasserstein"][step] = float(measures["wasserstein"])
            trace["js"][step] = float(measures["js"])
            if not use_severity:
                severity = 1.0 if decision_drift_evidence else 0.0
                normalized = {"ks": severity, "wasserstein": severity, "js": severity}
            trace["severity"][step] = severity

            degradation_record = degradation.evaluate_residual_degradation(
                np.asarray(recent_residuals), step=step
            )
            trace["relative_degradation"][step] = float(
                degradation_record["relative_degradation"]
            )
            trace["degradation_passed"][step] = int(
                degradation_record["is_significant_degradation"]
            )
            trace["persistence_count"][step] = int(
                persistence_record["persistence_count"]
            )
            persistence_passed = (
                persistence_record["is_persistent"]
                if use_persistence
                else decision_drift_evidence
            )
            trace["persistence_passed"][step] = int(persistence_passed)
            degradation_passed = (
                degradation_record["is_significant_degradation"]
                if use_degradation
                else decision_drift_evidence
            )
            decision = engine.decide(
                step=step,
                timestamp_utc=str(step),
                drift_detected=decision_drift_evidence,
                detector_name=(
                    "ADWIN+SUSTAINED_RESIDUAL"
                    if sustained_elevation else "ADWIN"
                ),
                raw_measures=measures,
                normalized_measures=normalized,
                severity=severity,
                persistence_count=(
                    persistence_record["persistence_count"]
                    if use_persistence else int(bool(drift_flag))
                ),
                persistence_passed=persistence_passed,
                degradation_value=(
                    degradation_record["relative_degradation"]
                    if use_degradation
                    else float(drift_cfg["degradation_threshold_delta_q"]) + 1.0
                    if drift_flag else 0.0
                ),
                degradation_passed=degradation_passed,
                raw_adwin_alarm=bool(trace["raw_adwin_alarm"][step]),
                drift_confirmed=decision_drift_evidence,
                suppression_reason=str(trace["suppression_reason"][step]),
                anomaly_mask_fraction=float(
                    trace["mask_dominance_fraction"][step]
                ),
                clean_reference_count=(
                    int(b4_confirmation.get("reference_clean_count", 0))
                    if b4_confirmation is not None else 0
                ),
                clean_recent_count=(
                    int(b4_confirmation.get("recent_clean_count", 0))
                    if b4_confirmation is not None else 0
                ),
                cut_point=(
                    int(b4_confirmation["cut_point"])
                    if b4_confirmation is not None else 0
                ),
            )
            decision_seconds += time.perf_counter() - started
            if decision.action == AdaptationAction.STRONG_UPDATE:
                request_action = "STRONG_UPDATE"
            elif decision.action == AdaptationAction.FINE_TUNE:
                request_action = "FINE_TUNE"

        if request_action is None or len(recent_entries) < minimum_candidate_windows:
            continue
        if (
            baseline_name == "B4"
            and last_candidate_attempt_step is not None
            and step - last_candidate_attempt_step < candidate_cooldown_windows
        ):
            continue
        if baseline_name == "B4":
            last_candidate_attempt_step = step
        request_count += 1
        trace["action"][step] = request_action
        if (
            drift_start_window is not None
            and step >= drift_start_window
            and first_decision_step is None
        ):
            first_decision_step = step

        candidate_x = np.stack([entry["x"] for entry in recent_entries])
        candidate_y = np.stack([entry["y"] for entry in recent_entries])
        candidate_training_window_steps.extend(
            int(entry["step"]) for entry in recent_entries
        )
        is_strong = request_action == "STRONG_UPDATE"
        training_started = time.perf_counter()
        candidate_model, candidate_metadata, _ = trainer.create_candidate(
            current_model,
            candidate_x,
            candidate_y,
            action_type=request_action,
            epochs=int(
                continual_cfg["strong_epochs"]
                if is_strong else continual_cfg["finetune_epochs"]
            ),
            lr=float(
                continual_cfg["strong_lr"]
                if is_strong else continual_cfg["finetune_lr"]
            ),
            replay_ratio=float(
                continual_cfg.get("strong_replay_ratio", continual_cfg["replay_ratio"])
                if is_strong else continual_cfg["replay_ratio"]
            ),
            batch_size=int(model_cfg["batch_size"]),
            window_stride=int(model_cfg.get("window_stride", 1)),
        )
        training_seconds += time.perf_counter() - training_started

        validation_started = time.perf_counter()
        result = validation_gate.evaluate_candidate(
            current_model, candidate_model, candidate_metadata
        )
        validation_seconds += time.perf_counter() - validation_started
        if not result.accepted:
            continue

        current_model = candidate_model
        threshold_scores, inference_seconds = score_model_on_windows(
            current_model,
            scenario_data["gate_all_x"],
            scenario_data["gate_all_y"],
            device,
            int(model_cfg["batch_size"]),
        )
        current_threshold = _calibrate_anomaly_threshold(
            threshold_scores,
            scenario_data["gate_valid_mask"],
            float(anomaly_cfg["ema_alpha"]),
            float(anomaly_cfg["threshold_quantile"]),
        )
        adaptation_inference_seconds += inference_seconds
        update_counts[request_action] += 1
        trace["accepted_update"][step] = 1
        for entry in recent_entries:
            accepted_replay_window_steps.append(int(entry["step"]))
            replay.add(entry["x"], entry["y"], is_anomaly_masked=False)
        reference_residuals = list(recent_residuals)
        if step + 1 < len(adaptation_x):
            remaining_scores, inference_seconds = score_model_on_windows(
                current_model,
                adaptation_x[step + 1:],
                adaptation_y[step + 1:],
                device,
                int(model_cfg["batch_size"]),
            )
            cached_scores[step + 1:] = remaining_scores
            trace["raw_mse"][step + 1:] = remaining_scores
            adaptation_inference_seconds += inference_seconds

    evaluation_started = time.perf_counter()
    event_metrics, test_predictions, evaluation_inference_seconds = _evaluate_frozen_model(
        current_model, scenario_data, cfg, device, threshold=current_threshold
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    gate_summary = validation_gate.get_summary()
    total_updates = sum(update_counts.values())
    candidate_training_pulse_overlaps = _count_injected_anomaly_window_overlaps(
        scenario_data,
        candidate_training_window_steps,
        int(model_cfg["input_window"]),
        int(model_cfg["forecast_horizon"]),
    )
    accepted_replay_pulse_overlaps = _count_injected_anomaly_window_overlaps(
        scenario_data,
        accepted_replay_window_steps,
        int(model_cfg["input_window"]),
        int(model_cfg["forecast_horizon"]),
    )
    harmless_scenario = scenario_id in {1, 2, 4}
    delay_windows = (
        first_decision_step - drift_start_window
        if first_decision_step is not None and drift_start_window is not None
        else None
    )

    return {
        "name": baseline_name,
        "ablation_variant": ablation_variant,
        "event_metrics": event_metrics,
        "updates": total_updates,
        "fine_tune_updates": update_counts["FINE_TUNE"],
        "strong_updates": update_counts["STRONG_UPDATE"],
        "update_requests": request_count,
        "unnecessary_updates": total_updates if harmless_scenario else 0,
        "updates_during_stable_period": total_updates if scenario_id == 1 else 0,
        "updates_during_temporary_period": total_updates if scenario_id == 2 else 0,
        "updates_during_large_harmless_period": total_updates if scenario_id == 4 else 0,
        "candidates_accepted": int(gate_summary["accepted_count"]),
        "candidates_rejected": int(gate_summary["rejected_count"]),
        "candidate_training_window_exposures": len(
            candidate_training_window_steps
        ),
        "candidate_training_pulse_contaminated_exposures":
            candidate_training_pulse_overlaps,
        "accepted_replay_sample_additions": len(accepted_replay_window_steps),
        "accepted_replay_pulse_contaminated_additions":
            accepted_replay_pulse_overlaps,
        "candidate_retention_rejections": int(
            gate_summary["rejection_counts"]["retention"]
        ),
        "candidate_adaptation_validation_rejections": int(
            gate_summary["rejection_counts"]["adaptation_validation"]
        ),
        "candidate_other_rejections": int(
            gate_summary["rejection_counts"]["other"]
        ),
        "drift_detections": len(drift_alarm_steps),
        "confirmed_drift_detections": len(confirmed_drift_steps),
        "mask_suppressed_detections": mask_suppression_count,
        "insufficient_evidence_suppressions": insufficient_evidence_count,
        "masked_windows": len(masked_window_indices),
        "sustained_elevation_windows": int(
            trace["sustained_elevation"].sum()
        ),
        "first_adaptation_delay_windows": delay_windows,
        "first_alarm_delay_windows": (
            first_alarm_after_drift - drift_start_window
            if first_alarm_after_drift is not None and drift_start_window is not None
            else None
        ),
        "runtime_total": time.perf_counter() - simulation_started,
        "runtime_adaptation_inference": adaptation_inference_seconds,
        "runtime_evaluation_inference": evaluation_inference_seconds,
        "runtime_detection": detection_seconds,
        "runtime_decision": decision_seconds,
        "runtime_training": training_seconds,
        "runtime_validation": validation_seconds,
        "runtime_final_evaluation": evaluation_seconds,
        "test_predictions": test_predictions,
        "trace": trace,
    }


def run_benchmark_for_scenario(
    scenario_id: int,
    cfg: Dict[str, Any],
    clean_splits: Dict[str, Any],
    device: torch.device,
    initial_checkpoint_path: Path,
) -> Dict[str, Any]:
    """Compare B1-B4 on the same scenario with no test-time model updates."""
    from src.models.anomaly_scoring import load_forecaster

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base_model = load_forecaster(initial_checkpoint_path, device)
    scenario_data = prepare_scenario_data(
        scenario_id, cfg, clean_splits, base_model, device
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

    results: Dict[str, Any] = {
        "scenario": scenario_data["scenario"],
        "baselines": {},
    }
    b1_started = time.perf_counter()
    b1_model = copy.deepcopy(base_model)
    b1_metrics, b1_predictions, b1_inference = _evaluate_frozen_model(
        b1_model, scenario_data, cfg, device
    )
    results["baselines"]["B1"] = {
        "name": "B1 - Static LSTM",
        "event_metrics": b1_metrics,
        "updates": 0,
        "fine_tune_updates": 0,
        "strong_updates": 0,
        "update_requests": 0,
        "unnecessary_updates": 0,
        "updates_during_stable_period": 0,
        "updates_during_temporary_period": 0,
        "updates_during_large_harmless_period": 0,
        "candidates_accepted": 0,
        "candidates_rejected": 0,
        "runtime_total": time.perf_counter() - b1_started,
        "runtime_evaluation_inference": b1_inference,
        "test_predictions": b1_predictions,
    }

    for baseline_name in ("B2", "B3", "B4"):
        results["baselines"][baseline_name] = run_adaptive_simulation(
            baseline_name,
            scenario_id,
            scenario_data,
            cfg,
            device,
            base_model,
            replay_seed,
        )
    return results
