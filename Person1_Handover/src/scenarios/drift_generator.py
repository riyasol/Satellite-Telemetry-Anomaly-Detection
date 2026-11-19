"""Controlled drift generation on Mission 1 telemetry.

Implements the five scenarios defined in the project plan:
1. Stable telemetry (clean, no injected drift).
2. Temporary drift (transient offset that returns to normal).
3. Persistent harmful drift (persistent shift degrading detection quality).
4. Large but harmless drift (large statistical shift without detector degradation).
5. Moderate harmful drift (moderate persistent shift that noticeably reduces F0.5).
"""
from __future__ import annotations
import copy
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
) -> Tuple[np.ndarray, DriftScenarioConfig]:
    """Generate modified telemetry stream for the specified scenario on Mission 1 data."""
    rng = np.random.default_rng(seed)
    n_rows, n_channels = base_values.shape
    modified = base_values.copy()

    # Place drift events well within the test stream (e.g. starting around step 1500 to 4500)
    mid_start = min(1500, n_rows // 4)
    mid_end = min(mid_start + 800, n_rows // 2)

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
        cfg = DriftScenarioConfig(
            scenario_id=2,
            name="Temporary Drift",
            description="Short-lived transient sensor bias pulse that returns to nominal.",
            transformation_type="transient_bias_pulse",
            start_step=mid_start,
            end_step=mid_start + 600,
            affected_channels=[0, 1],
            magnitude=1.5,
            seed=seed,
        )
        for ch in cfg.affected_channels:
            modified[cfg.start_step : cfg.end_step, ch] += cfg.magnitude

    elif scenario_id == 3:
        # Persistent harmful drift: permanent step and variance shift across primary target channels
        cfg = DriftScenarioConfig(
            scenario_id=3,
            name="Persistent Harmful Drift",
            description="Persistent step and variance shift that severely degrades prediction accuracy.",
            transformation_type="persistent_step_variance",
            start_step=mid_start,
            end_step=n_rows,
            affected_channels=[0, 1, 2],
            magnitude=2.2,
            seed=seed,
        )
        for ch in cfg.affected_channels:
            modified[cfg.start_step :, ch] += cfg.magnitude
            noise = rng.normal(0.0, 0.5, size=n_rows - cfg.start_step)
            modified[cfg.start_step :, ch] += noise

    elif scenario_id == 4:
        # Large but harmless drift: large shift on secondary/independent channel that doesn't harm forecast accuracy
        # (e.g. channel 5 with zero cross-correlation impact on anomaly scoring)
        cfg = DriftScenarioConfig(
            scenario_id=4,
            name="Large Harmless Drift",
            description="Large statistical distribution shift that does not cause detector degradation.",
            transformation_type="harmless_auxiliary_scale_shift",
            start_step=mid_start,
            end_step=n_rows,
            affected_channels=[5],
            magnitude=3.0,
            seed=seed,
        )
        modified[cfg.start_step :, 5] = modified[cfg.start_step :, 5] * 1.8 + cfg.magnitude

    elif scenario_id == 5:
        # Moderate harmful drift: moderate persistent slope/bias on correlated channels that induces false alarms
        cfg = DriftScenarioConfig(
            scenario_id=5,
            name="Moderate Harmful Drift",
            description="Moderate persistent shift that noticeably degrades detection quality and F0.5.",
            transformation_type="moderate_persistent_ramp_offset",
            start_step=mid_start,
            end_step=n_rows,
            affected_channels=[1, 3],
            magnitude=1.2,
            seed=seed,
        )
        ramp_len = min(500, n_rows - cfg.start_step)
        for ch in cfg.affected_channels:
            modified[cfg.start_step : cfg.start_step + ramp_len, ch] += np.linspace(0.0, cfg.magnitude, ramp_len)
            modified[cfg.start_step + ramp_len :, ch] += cfg.magnitude

    else:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")

    return modified, cfg


def get_all_scenarios(base_values: np.ndarray, seed: int = 42) -> List[Tuple[np.ndarray, DriftScenarioConfig]]:
    """Return all 5 scenarios."""
    return [generate_scenario(base_values, sc_id, seed=seed) for sc_id in range(1, 6)]

