"""Persistence tracking module.

Evaluates whether detected drift persists across multiple consecutive monitoring checks
rather than being an isolated transient fluctuation.
"""
from __future__ import annotations
from collections import deque
from typing import List, Dict, Any, Optional


class PersistenceTracker:
    """Tracks drift detections over a rolling window of k checks."""

    def __init__(self, window_k: int = 5, min_detections: int = 3):
        self.window_k = int(window_k)
        self.min_detections = int(min_detections)
        self.history: deque[int] = deque(maxlen=self.window_k)
        self.full_timeline: List[Dict[str, Any]] = []

    def reset(self):
        self.history.clear()
        self.full_timeline.clear()

    def update(self, drift_flag: bool, step: Optional[int] = None) -> Dict[str, Any]:
        """Record detection status and check if persistence threshold is satisfied."""
        flag_int = 1 if bool(drift_flag) else 0
        self.history.append(flag_int)

        current_count = sum(self.history)
        is_persistent = (current_count >= self.min_detections) and (len(self.history) >= self.min_detections)

        record = {
            "step": step,
            "drift_flag": bool(drift_flag),
            "recent_detections": list(self.history),
            "persistence_count": current_count,
            "window_k": self.window_k,
            "min_detections": self.min_detections,
            "is_persistent": is_persistent,
        }
        self.full_timeline.append(record)
        return record

    def get_state(self) -> Dict[str, Any]:
        count = sum(self.history)
        return {
            "window_size": len(self.history),
            "persistence_count": count,
            "is_persistent": count >= self.min_detections,
        }

