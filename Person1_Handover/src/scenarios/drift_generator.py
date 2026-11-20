"""Controlled drift generation on Mission 1 telemetry.

Implements the five scenarios defined in the project plan:
1. Stable telemetry (clean, no injected drift).
2. Temporary drift (transient offset that returns to normal).
3. Persistent harmful drift (persistent shift degrading detection quality).
4. Large but harmless drift (large statistical shift without detector degradation).
5. Moderate harmful drift (moderate persistent shift that noticeably reduces F0.5).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


@dataclass
class DriftScenarioConfig:
    scenario_id: int
    name: str
    description: str
    transformation_type: str
    start_step: int
    end_step: int
    affected_channels: List[int]
    magnitude: float
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_scenario(
    base_values: np.ndarray,
    scenario_id: int,
    seed: int = 42,
    scenario_parameters: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, DriftScenarioConfig]:
    """Generate modified telemetry stream for the specified scenario on Mission 1 data."""
    rng = np.random.default_rng(seed)
    n_rows, n_channels = base_values.shape
    modified = base_values.copy()
    parameters = scenario_parameters or {}

    # Place drift events well within the test stream (e.g. starting around step 1500 to 4500)
    start_cap = int(parameters.get("start_step_cap", 1500))
    start_denominator = int(parameters.get("start_fraction_denominator", 4))
    if start_cap < 0 or start_denominator <= 0:
        raise ValueError("Scenario start parameters must be non-negative and valid.")
    mid_start = min(start_cap, n_rows // start_denominator)
    configured_channels = parameters.get("affected_channels", [])
    if any(int(channel) < 0 or int(channel) >= n_channels for channel in configured_channels):
        raise ValueError("Scenario affected_channels must reference available features.")

    if scenario_id == 1:
        cfg = DriftScenarioConfig(
            scenario_id=1,
            name="Stable Telemetry",
            description="Nominal clean operating telemetry with no injected drift.",
            transformation_type="none",
            start_step=-1,
            end_step=-1,
            affected_channels=[],
            magnitude=0.0,
            seed=seed,
        )

    elif scenario_id == 2:
        # Temporary drift: transient shift for 600 steps then returns to baseline
        duration = int(parameters.get("duration_steps", 600))
        affected_channels = list(parameters.get("affected_channels", [0, 1]))
        magnitude = float(parameters.get("magnitude", 1.5))
        cfg = DriftScenarioConfig(
            scenario_id=2,
            name="Temporary Drift",
            description="Short-lived transient sensor bias pulse that returns to nominal.",
            transformation_type="transient_bias_pulse",
            start_step=mid_start,
            end_step=min(mid_start + duration, n_rows),
            affected_channels=affected_channels,
            magnitude=magnitude,
            seed=seed,
        )
        for ch in cfg.affected_channels:
            modified[cfg.start_step : cfg.end_step, ch] += cfg.magnitude

    elif scenario_id == 3:
        # Persistent harmful drift: permanent step and variance shift across primary target channels
        affected_channels = list(parameters.get("affected_channels", [0, 1, 2]))
        magnitude = float(parameters.get("magnitude", 2.2))
        noise_std = float(parameters.get("noise_std", 0.5))
        cfg = DriftScenarioConfig(
            scenario_id=3,
            name="Persistent Harmful Drift",
            description="Persistent step and variance shift that severely degrades prediction accuracy.",
            transformation_type="persistent_step_variance",
            start_step=mid_start,
            end_step=n_rows,
            affected_channels=affected_channels,
            magnitude=magnitude,
            seed=seed,
        )
        for ch in cfg.affected_channels:
            modified[cfg.start_step :, ch] += cfg.magnitude
            noise = rng.normal(0.0, noise_std, size=n_rows - cfg.start_step)
            modified[cfg.start_step :, ch] += noise

    elif scenario_id == 4:
        # Large but harmless drift: large shift on secondary/independent channel that doesn't harm forecast accuracy
        # (e.g. channel 5 with zero cross-correlation impact on anomaly scoring)
        affected_channels = list(parameters.get("affected_channels", [5]))
        magnitude = float(parameters.get("magnitude", 3.0))
        scale = float(parameters.get("scale", 1.8))
        cfg = DriftScenarioConfig(
            scenario_id=4,
            name="Large Harmless Drift",
            description="Large statistical distribution shift that does not cause detector degradation.",
            transformation_type="harmless_auxiliary_scale_shift",
            start_step=mid_start,
            end_step=n_rows,
            affected_channels=affected_channels,
            magnitude=magnitude,
            seed=seed,
        )
        for ch in cfg.affected_channels:
            modified[cfg.start_step :, ch] = modified[cfg.start_step :, ch] * scale + cfg.magnitude

    elif scenario_id == 5:
        # Moderate harmful drift: moderate persistent slope/bias on correlated channels that induces false alarms
        affected_channels = list(parameters.get("affected_channels", [1, 3]))
        magnitude = float(parameters.get("magnitude", 1.2))
        ramp_steps = int(parameters.get("ramp_steps", 500))
        cfg = DriftScenarioConfig(
            scenario_id=5,
            name="Moderate Harmful Drift",
            description="Moderate persistent shift that noticeably degrades detection quality and F0.5.",
            transformation_type="moderate_persistent_ramp_offset",
            start_step=mid_start,
            end_step=n_rows,
            affected_channels=affected_channels,
            magnitude=magnitude,
            seed=seed,
        )
        ramp_len = min(ramp_steps, n_rows - cfg.start_step)
        for ch in cfg.affected_channels:
            modified[cfg.start_step : cfg.start_step + ramp_len, ch] += np.linspace(0.0, cfg.magnitude, ramp_len)
            modified[cfg.start_step + ramp_len :, ch] += cfg.magnitude

    else:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")

    return modified, cfg


def get_all_scenarios(base_values: np.ndarray, seed: int = 42) -> List[Tuple[np.ndarray, DriftScenarioConfig]]:
    """Return all 5 scenarios."""
    return [generate_scenario(base_values, sc_id, seed=seed) for sc_id in range(1, 6)]


def inject_anomaly_pulses(
    base_values: np.ndarray,
    drift_start_step: int,
    stride: int = 10,
    channels: Optional[List[int]] = None,
    magnitude: float = 8.0,
    pulse_length: int = 3,
    offsets_in_strides: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject reproducible short anomaly pulses near a drift and return point labels."""
    values = np.asarray(base_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("base_values must be a non-empty 2-D telemetry array.")
    if stride <= 0 or pulse_length <= 0:
        raise ValueError("stride and pulse_length must be positive.")
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("magnitude must be finite and positive.")

    affected = list(channels if channels is not None else [0, 1])
    if not affected or any(channel < 0 or channel >= values.shape[1] for channel in affected):
        raise ValueError("Anomaly channels must be valid telemetry feature indices.")

    modified = values.copy()
    labels = np.zeros(len(modified), dtype=np.uint8)
    offsets = offsets_in_strides if offsets_in_strides is not None else [-2, 2, 6]
    for offset in offsets:
        start = min(
            max(0, int(drift_start_step) + int(offset) * stride),
            len(modified),
        )
        end = min(len(modified), start + pulse_length)
        if end <= start:
            continue
        modified[start:end, affected] += float(magnitude)
        labels[start:end] = 1

    return modified, labels

