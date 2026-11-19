"""Severity calculation and normalization module.

Combines KS, Wasserstein, and JS divergence into a unified severity score:
    S_t = w_ks * KS_norm + w_w * W_norm + w_js * JS_norm
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


class SeverityCalculator:
    """Normalizes distribution measures and computes weighted drift severity."""

    def __init__(
        self,
        weights: List[float] = None,
        ks_range: tuple = (0.0, 0.40),
        wasserstein_range: tuple = (0.0, 0.60),
        js_range: tuple = (0.0, 0.35),
    ):
        if weights is None:
            self.weights = [0.4, 0.3, 0.3]
        else:
            w = np.array(weights, dtype=np.float64)
            self.weights = (w / w.sum()).tolist()

        self.ks_range = ks_range
        self.wasserstein_range = wasserstein_range
        self.js_range = js_range
        self.history_logs: List[Dict[str, Any]] = []

    def calibrate_from_normal_validation(
        self, normal_val_residuals: np.ndarray, window_size: int = 100, num_trials: int = 40
    ):
        """Empirically calibrate normalization upper bounds using normal validation variation.

        Samples pairs of normal validation sub-windows to determine background variation scale.
        """
        from src.drift.distribution_measures import (
            compute_ks_statistic, compute_wasserstein_distance, compute_js_divergence
        )
        n = len(normal_val_residuals)
        if n < window_size * 2:
            return

        ks_vals, w_vals, js_vals = [], [], []
        rng = np.random.default_rng(42)

        for _ in range(num_trials):
            idx1 = rng.integers(0, n - window_size)
            idx2 = rng.integers(0, n - window_size)
            w1 = normal_val_residuals[idx1 : idx1 + window_size]
            w2 = normal_val_residuals[idx2 : idx2 + window_size]

            ks_vals.append(compute_ks_statistic(w1, w2))
            w_vals.append(compute_wasserstein_distance(w1, w2))
            js_vals.append(compute_js_divergence(w1, w2))

        # 99th percentile of background variation as baseline scale
        ks_max = max(0.20, float(np.percentile(ks_vals, 99)) * 2.5)
        w_max = max(0.25, float(np.percentile(w_vals, 99)) * 2.5)
        js_max = max(0.15, float(np.percentile(js_vals, 99)) * 2.5)

        self.ks_range = (0.0, ks_max)
        self.wasserstein_range = (0.0, w_max)
        self.js_range = (0.0, js_max)

    def normalize_value(self, val: float, val_range: tuple) -> float:
        vmin, vmax = val_range
        if vmax <= vmin:
            return 0.0
        norm = (val - vmin) / (vmax - vmin)
        return float(np.clip(norm, 0.0, 1.0))

    def compute_severity(
        self, raw_measures: Dict[str, float], step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute normalized measures and weighted severity score."""
        ks_raw = float(raw_measures.get("ks", 0.0))
        w_raw = float(raw_measures.get("wasserstein", 0.0))
        js_raw = float(raw_measures.get("js", 0.0))

        ks_norm = self.normalize_value(ks_raw, self.ks_range)
        w_norm = self.normalize_value(w_raw, self.wasserstein_range)
        js_norm = self.normalize_value(js_raw, self.js_range)

        severity = (
            self.weights[0] * ks_norm +
            self.weights[1] * w_norm +
            self.weights[2] * js_norm
        )
        severity = float(np.clip(severity, 0.0, 1.0))

        record = {
            "step": step,
            "raw": {"ks": ks_raw, "wasserstein": w_raw, "js": js_raw},
            "normalized": {"ks": ks_norm, "wasserstein": w_norm, "js": js_norm},
            "weights": self.weights,
            "ranges": {
                "ks": self.ks_range,
                "wasserstein": self.wasserstein_range,
                "js": self.js_range,
            },
            "severity": severity,
        }
        self.history_logs.append(record)
        return record

