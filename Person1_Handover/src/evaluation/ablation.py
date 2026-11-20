"""Ablation study runner using the same P3 adaptation/evaluation protocol as B4."""
from __future__ import annotations

from typing import Any, Dict

import torch

from src.continual.replay_buffer import ReplayBuffer
from src.evaluation.benchmark import run_adaptive_simulation
from src.models.lstm_forecaster import LSTMForecaster


ABLATION_VARIANTS = (
    "Complete_B4",
    "No_Severity",
    "No_Persistence",
    "No_Degradation",
    "No_Anomaly_Masking",
)


def run_ablation_simulation(
    ablation_variant: str,
    scenario_id: int,
    scenario_data: Dict[str, Any],
    base_model: LSTMForecaster,
    replay_seed_buffer: ReplayBuffer,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run B4 with one decision or masking component removed."""
    if ablation_variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {ablation_variant}")
    return run_adaptive_simulation(
        baseline_name="B4",
        scenario_id=scenario_id,
        scenario_data=scenario_data,
        cfg=cfg,
        device=device,
        base_model=base_model,
        replay_seed_buffer=replay_seed_buffer,
        ablation_variant=(
            None if ablation_variant == "Complete_B4" else ablation_variant
        ),
    )
