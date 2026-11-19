"""Streaming drift detectors: ADWIN and Page-Hinkley.

Both detectors process a real-valued observation stream (such as the forecasting
error residual stream) sequentially in chronological order.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Optional, Tuple, Dict, Any


class ADWIN:
    """Adaptive Windowing (ADWIN) drift detector.

    ADWIN maintains a variable-length window W of recent values. When two
    sub-windows exhibit a statistically significant difference in their means
    exceeding the Hoeffding bound, drift is signaled and older observations
    are pruned.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 2000, min_window: int = 20):
        self.delta = float(delta)
        self.max_window = int(max_window)
        self.min_window = int(min_window)
        self.window: deque[float] = deque()
        self.total_sum = 0.0
        self.total_count = 0
        self.drift_detected = False
        self.last_cut_point = 0
        self.history_detected = []

    def reset(self):
        self.window.clear()
        self.total_sum = 0.0
        self.total_count = 0
        self.drift_detected = False
        self.last_cut_point = 0

    def update(self, val: float) -> bool:
        """Add a new observation and check for drift. Returns True if drift detected."""
        val = float(val)
        if not math.isfinite(val):
            return False

        self.window.append(val)
        self.total_sum += val
        self.total_count += 1
        self.drift_detected = False

        if len(self.window) > self.max_window:
            removed = self.window.popleft()
            self.total_sum -= removed

        n = len(self.window)
        if n < self.min_window * 2:
            return False

        # Evaluate potential sub-window splits
        # To keep it efficient, check split points at step sizes
        step = max(5, n // 30)
        prefix_sum = 0.0
        elements = list(self.window)

        for i in range(step, n - step, step):
            if i < self.min_window or (n - i) < self.min_window:
                continue

            prefix_sum = sum(elements[:i])
            n0 = i
            n1 = n - i
            mu0 = prefix_sum / n0
            mu1 = (self.total_sum - prefix_sum) / n1

            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            # Hoeffding bound with Bonferroni correction for multiple split checks
            epsilon_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * n / self.delta))

            if abs(mu0 - mu1) > epsilon_cut:
                self.drift_detected = True
                self.last_cut_point = i
                # Prune older sub-window
                for _ in range(i):
                    val_rem = self.window.popleft()
                    self.total_sum -= val_rem
                break

        return self.drift_detected

    def get_state(self) -> Dict[str, Any]:
        n = len(self.window)
        mean_val = (self.total_sum / n) if n > 0 else 0.0
        return {
            "name": "ADWIN",
            "window_size": n,
            "current_mean": mean_val,
            "drift_detected": self.drift_detected,
            "delta": self.delta
        }


class PageHinkley:
    """Page-Hinkley test for cumulative change detection.

    Tracks cumulative deviations from a running mean. A drift is flagged
    when the difference between cumulative sum m_t and minimum observed
    cumulative sum M_t exceeds threshold lambda.
    """

    def __init__(self, delta: float = 0.05, threshold: float = 25.0, alpha: float = 0.01):
        self.delta = float(delta)         # Allowed drift / tolerance magnitude
        self.threshold = float(threshold) # Detection threshold (lambda)
        self.alpha = float(alpha)         # Forgetting factor / EMA weight
        self.mean = 0.0
        self.count = 0
        self.m_sum = 0.0
        self.m_min = 0.0
        self.ph_statistic = 0.0
        self.drift_detected = False

    def reset(self):
        self.mean = 0.0
        self.count = 0
        self.m_sum = 0.0
        self.m_min = 0.0
        self.ph_statistic = 0.0
        self.drift_detected = False

    def update(self, val: float) -> bool:
        """Add observation and check for drift. Returns True if drift detected."""
        val = float(val)
        if not math.isfinite(val):
            return False

        self.count += 1
        # Update running mean
        self.mean += (val - self.mean) / self.count

        # Cumulative difference
        self.m_sum += (val - self.mean - self.delta)
        if self.m_sum < self.m_min:
            self.m_min = self.m_sum

        self.ph_statistic = self.m_sum - self.m_min

        if self.ph_statistic > self.threshold:
            self.drift_detected = True
            # Reset after alarm to allow monitoring subsequent stream changes
            self.m_sum = 0.0
            self.m_min = 0.0
        else:
            self.drift_detected = False

        return self.drift_detected

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": "Page-Hinkley",
            "count": self.count,
            "running_mean": self.mean,
            "ph_statistic": self.ph_statistic,
            "threshold": self.threshold,
            "drift_detected": self.drift_detected
        }

