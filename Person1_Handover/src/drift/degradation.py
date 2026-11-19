"""Detector degradation tracking module.

Measures whether the anomaly detector's operational performance has degraded
compared to the reference baseline:
    Delta_F0.5 = F0.5_reference - F0.5_current
or telemetry residual prediction degradation:
    Delta_MSE = MSE_current - MSE_reference
A positive value indicates performance deterioration.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np


class DegradationTracker:
    """Tracks performance or forecast error degradation relative to reference baseline."""

    def __init__(
        self,
        reference_f05: float = 0.929,
        degradation_threshold_delta_q: float = 0.05,
        reference_mse: float = 0.20,
    ):
        self.reference_f05 = float(reference_f05)
        self.degradation_threshold_delta_q = float(degradation_threshold_delta_q)
        self.reference_mse = float(reference_mse)
        self.history: List[Dict[str, Any]] = []

    def evaluate_f05_degradation(
        self, current_f05: float, step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute Delta F0.5 = F0.5_ref - F0.5_curr. Positive means degraded."""
        delta_f05 = self.reference_f05 - float(current_f05)
        is_significant = delta_f05 >= self.degradation_threshold_delta_q

        record = {
            "step": step,
            "type": "F0.5",
            "reference_f05": self.reference_f05,
            "current_f05": float(current_f05),
            "delta_f05": float(delta_f05),
            "threshold_delta_q": self.degradation_threshold_delta_q,
            "is_significant_degradation": bool(is_significant),
        }
        self.history.append(record)
        return record

    def evaluate_residual_degradation(
        self, current_residuals: np.ndarray, step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute residual MSE increase over reference normal baseline as proxy."""
        cur_clean = np.asarray(current_residuals).flatten()
        cur_clean = cur_clean[np.isfinite(cur_clean)]
        if len(cur_clean) == 0:
            cur_mse = self.reference_mse
        else:
            cur_mse = float(np.mean(cur_clean ** 2))

        # Normalized increase in error
        rel_degradation = (cur_mse - self.reference_mse) / max(1e-4, self.reference_mse)
        is_significant = rel_degradation >= self.degradation_threshold_delta_q

        record = {
            "step": step,
            "type": "Residual_MSE",
            "reference_mse": self.reference_mse,
            "current_mse": cur_mse,
            "relative_degradation": float(rel_degradation),
            "threshold_delta_q": self.degradation_threshold_delta_q,
            "is_significant_degradation": bool(is_significant),
        }
        self.history.append(record)
        return record

