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
        reference_f05: Optional[float] = None,
        degradation_threshold_delta_q: float = 0.05,
        reference_mse: float = 0.20,
    ) -> None:
        """Create a DegradationTracker.

        Parameters
        ----------
        reference_f05: Optional[float]
            Baseline event‑wise F0.5 from the initial B1 model. If ``None`` the
            tracker will *not* evaluate F0.5‑based degradation and will rely only
            on residual‑MSE degradation.
        degradation_threshold_delta_q: float
            Relative degradation threshold used for both F0.5 and residual MSE.
        reference_mse: float
            Reference MSE from clean validation data (used for residual degradation).
        """
        self.reference_f05 = float(reference_f05) if reference_f05 is not None else None
        self.degradation_threshold_delta_q = float(degradation_threshold_delta_q)
        self.reference_mse = float(reference_mse)
        self.history: List[Dict[str, Any]] = []

    def evaluate_f05_degradation(
        self, current_f05: float, step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute Delta F0.5 = F0.5_ref - F0.5_curr. Positive means degraded.

        If ``self.reference_f05`` is ``None`` (no baseline F0.5 is available) the
        method returns a record indicating that F0.5‑based degradation is not
        applicable. The caller should then rely on the residual‑MSE degradation
        metric.
        """
        if self.reference_f05 is None:
            # No reference F0.5 – skip this metric.
            record = {
                "step": step,
                "type": "F0.5",
                "reference_f05": None,
                "current_f05": float(current_f05),
                "delta_f05": None,
                "threshold_delta_q": self.degradation_threshold_delta_q,
                "is_significant_degradation": False,
            }
            self.history.append(record)
            return record
        # Normal case with a valid reference.
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

