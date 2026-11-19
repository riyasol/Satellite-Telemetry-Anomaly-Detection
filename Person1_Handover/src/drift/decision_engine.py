"""Three-way adaptation decision engine.

Synthesizes drift detection, severity, persistence, and detector degradation
into one of three actions:
    1. NO_UPDATE: Keep current model unchanged
    2. FINE_TUNE: Request moderate continual learning fine-tuning
    3. STRONG_UPDATE: Request strong retraining with replay buffer
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, List
import json


class AdaptationAction(str, Enum):
    NO_UPDATE = "NO_UPDATE"
    FINE_TUNE = "FINE_TUNE"
    STRONG_UPDATE = "STRONG_UPDATE"


@dataclass
class DecisionObject:
    step: int
    timestamp_utc: str
    drift_detected: bool
    detector_name: str
    raw_measures: Dict[str, float]
    normalized_measures: Dict[str, float]
    severity: float
    persistence_count: int
    persistence_passed: bool
    degradation_value: float
    degradation_passed: bool
    action: AdaptationAction
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


class DecisionEngine:
    """Evaluates multi-criteria conditions to make safe, drift-aware adaptation decisions."""

    def __init__(
        self,
        theta_low: float = 0.35,
        theta_high: float = 0.65,
        persistence_min: int = 3,
        degradation_threshold: float = 0.05,
    ):
        self.theta_low = float(theta_low)
        self.theta_high = float(theta_high)
        self.persistence_min = int(persistence_min)
        self.degradation_threshold = float(degradation_threshold)
        self.decisions_log: List[DecisionObject] = []

    def decide(
        self,
        step: int,
        timestamp_utc: str,
        drift_detected: bool,
        detector_name: str,
        raw_measures: Dict[str, float],
        normalized_measures: Dict[str, float],
        severity: float,
        persistence_count: int,
        persistence_passed: bool,
        degradation_value: float,
        degradation_passed: bool,
    ) -> DecisionObject:
        """Apply three-way decision logic."""
        if not persistence_passed or severity < self.theta_low:
            action = AdaptationAction.NO_UPDATE
            if not persistence_passed:
                reason = (
                    f"Drift not sufficiently persistent (count={persistence_count} < min={self.persistence_min})"
                )
            else:
                reason = f"Severity ({severity:.3f}) below lower threshold theta_L ({self.theta_low:.3f})"

        elif severity >= self.theta_high and degradation_passed:
            action = AdaptationAction.STRONG_UPDATE
            reason = (
                f"Persistent high-severity drift (S={severity:.3f} >= theta_H={self.theta_high:.3f}) "
                f"with significant degradation ({degradation_value:.3f} >= {self.degradation_threshold:.3f})"
            )

        elif severity >= self.theta_low:
            action = AdaptationAction.FINE_TUNE
            reason = (
                f"Persistent moderate drift (S={severity:.3f} >= theta_L={self.theta_low:.3f}) "
                f"with controlled/limited degradation"
            )
        else:
            action = AdaptationAction.NO_UPDATE
            reason = "Conditions for update not satisfied"

        decision = DecisionObject(
            step=step,
            timestamp_utc=str(timestamp_utc),
            drift_detected=bool(drift_detected),
            detector_name=str(detector_name),
            raw_measures=raw_measures,
            normalized_measures=normalized_measures,
            severity=float(severity),
            persistence_count=int(persistence_count),
            persistence_passed=bool(persistence_passed),
            degradation_value=float(degradation_value),
            degradation_passed=bool(degradation_passed),
            action=action,
            reason=reason,
        )
        self.decisions_log.append(decision)
        return decision

    def get_summary(self) -> Dict[str, int]:
        counts = {a.value: 0 for a in AdaptationAction}
        for d in self.decisions_log:
            counts[d.action.value] += 1
        return counts

