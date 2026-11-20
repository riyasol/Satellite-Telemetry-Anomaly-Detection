"""Validation gate for continual learning.

Candidates must retain performance on clean telemetry and improve on a separate
held-out adaptation segment before promotion.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_forecaster import LSTMForecaster


@dataclass
class CandidateEvaluationResult:
    candidate_id: str
    parent_version: str
    accepted: bool
    current_val_loss: float
    candidate_val_loss: float
    loss_improvement: float
    tolerance: float
    evaluation_time_seconds: float
    rejection_reason: Optional[str]
    metadata: Dict[str, Any]
    current_adaptation_val_loss: Optional[float] = None
    candidate_adaptation_val_loss: Optional[float] = None
    adaptation_loss_improvement: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ValidationGate:
    """Rigorous gate controlling candidate deployment."""

    def __init__(
        self,
        val_x: np.ndarray,
        val_y: np.ndarray,
        device: torch.device,
        tolerance: float = 0.02, # Candidate loss must not exceed current loss * (1 + tolerance)
        batch_size: int = 128,
        adaptation_val_x: Optional[np.ndarray] = None,
        adaptation_val_y: Optional[np.ndarray] = None,
    ):
        val_x = np.asarray(val_x, dtype=np.float32)
        val_y = np.asarray(val_y, dtype=np.float32)
        if val_x.ndim != 3 or val_y.ndim != 3:
            raise ValueError("Validation inputs and targets must be 3-D window arrays.")
        if len(val_x) == 0 or len(val_x) != len(val_y):
            raise ValueError("Validation inputs and targets must have equal, non-zero lengths.")
        if val_x.shape[2] != val_y.shape[2]:
            raise ValueError("Validation inputs and targets must have the same feature count.")
        if not np.isfinite(val_x).all() or not np.isfinite(val_y).all():
            raise ValueError("Validation data must contain only finite values.")
        if not np.isfinite(tolerance) or float(tolerance) < 0:
            raise ValueError("tolerance must be finite and non-negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if (adaptation_val_x is None) != (adaptation_val_y is None):
            raise ValueError(
                "Adaptation validation inputs and targets must be provided together."
            )

        self.val_x = val_x
        self.val_y = val_y
        self.device = device
        self.tolerance = float(tolerance)
        self.batch_size = int(batch_size)

        ds = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())
        self.val_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        self.adaptation_val_loader = None
        if adaptation_val_x is not None and adaptation_val_y is not None:
            adaptation_val_x = np.asarray(adaptation_val_x, dtype=np.float32)
            adaptation_val_y = np.asarray(adaptation_val_y, dtype=np.float32)
            if adaptation_val_x.ndim != 3 or adaptation_val_y.ndim != 3:
                raise ValueError(
                    "Adaptation validation inputs and targets must be 3-D window arrays."
                )
            if (
                len(adaptation_val_x) == 0
                or len(adaptation_val_x) != len(adaptation_val_y)
            ):
                raise ValueError(
                    "Adaptation validation inputs and targets must have equal, "
                    "non-zero lengths."
                )
            if (
                adaptation_val_x.shape[2] != val_x.shape[2]
                or adaptation_val_y.shape[2] != val_y.shape[2]
                or adaptation_val_x.shape[1] != val_x.shape[1]
                or adaptation_val_y.shape[1] != val_y.shape[1]
            ):
                raise ValueError(
                    "Adaptation and retention validation window shapes must match."
                )
            if (
                not np.isfinite(adaptation_val_x).all()
                or not np.isfinite(adaptation_val_y).all()
            ):
                raise ValueError("Adaptation validation data must contain only finite values.")
            adaptation_ds = TensorDataset(
                torch.from_numpy(adaptation_val_x).float(),
                torch.from_numpy(adaptation_val_y).float(),
            )
            self.adaptation_val_loader = DataLoader(
                adaptation_ds, batch_size=self.batch_size, shuffle=False
            )
        self.criterion = nn.MSELoss()

        self.evaluation_history: List[CandidateEvaluationResult] = []
        self.accepted_promotions_count = 0
        self.rejected_count = 0

    def evaluate_model_loss(
        self, model: LSTMForecaster, adaptation: bool = False
    ) -> float:
        """Compute average validation MSE loss."""
        loader = (
            self.adaptation_val_loader if adaptation else self.val_loader
        )
        if loader is None:
            raise ValueError("Adaptation validation data was not configured.")
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                pred = model(bx)
                loss = self.criterion(pred, by)
                bs = bx.size(0)
                total_loss += loss.item() * bs
                total_samples += bs
        return float(total_loss / total_samples) if total_samples > 0 else float("inf")

    def evaluate_candidate(
        self,
        current_model: LSTMForecaster,
        candidate_model: LSTMForecaster,
        candidate_metadata: Dict[str, Any],
        parent_version: str = "accepted_model",
    ) -> CandidateEvaluationResult:
        """Evaluate candidate vs current model and decide whether to accept or reject."""
        start_time = time.perf_counter()

        current_loss = self.evaluate_model_loss(current_model)
        candidate_loss = self.evaluate_model_loss(candidate_model)
        current_adaptation_loss = None
        candidate_adaptation_loss = None
        adaptation_loss_improvement = None
        if self.adaptation_val_loader is not None:
            current_adaptation_loss = self.evaluate_model_loss(
                current_model, adaptation=True
            )
            candidate_adaptation_loss = self.evaluate_model_loss(
                candidate_model, adaptation=True
            )
            adaptation_loss_improvement = (
                current_adaptation_loss - candidate_adaptation_loss
            )

        loss_improvement = current_loss - candidate_loss
        max_allowable_loss = current_loss * (1.0 + self.tolerance)

        if not np.isfinite(current_loss):
            raise RuntimeError("Current model produced a non-finite validation loss.")
        if (
            current_adaptation_loss is not None
            and not np.isfinite(current_adaptation_loss)
        ):
            raise RuntimeError(
                "Current model produced a non-finite adaptation validation loss."
            )

        retention_passed = (
            np.isfinite(candidate_loss) and candidate_loss <= max_allowable_loss
        )
        adaptation_passed = (
            candidate_adaptation_loss is None
            or (
                np.isfinite(candidate_adaptation_loss)
                and candidate_adaptation_loss < current_adaptation_loss
            )
        )

        if retention_passed and adaptation_passed:
            accepted = True
            rejection_reason = None
            self.accepted_promotions_count += 1
        elif not retention_passed:
            accepted = False
            rejection_reason = (
                f"Candidate retention validation MSE ({candidate_loss:.6f}) exceeded "
                f"allowable limit ({max_allowable_loss:.6f}, baseline={current_loss:.6f})"
            )
            self.rejected_count += 1
        else:
            accepted = False
            rejection_reason = (
                "Candidate did not improve held-out adaptation validation MSE "
                f"({candidate_adaptation_loss:.6f} vs. {current_adaptation_loss:.6f})."
            )
            self.rejected_count += 1

        eval_time = time.perf_counter() - start_time

        result = CandidateEvaluationResult(
            candidate_id=candidate_metadata.get("candidate_id", "unknown"),
            parent_version=parent_version,
            accepted=accepted,
            current_val_loss=current_loss,
            candidate_val_loss=candidate_loss,
            loss_improvement=loss_improvement,
            tolerance=self.tolerance,
            evaluation_time_seconds=eval_time,
            rejection_reason=rejection_reason,
            metadata=candidate_metadata,
            current_adaptation_val_loss=current_adaptation_loss,
            candidate_adaptation_val_loss=candidate_adaptation_loss,
            adaptation_loss_improvement=adaptation_loss_improvement,
        )
        self.evaluation_history.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        rejection_counts = {
            "retention": 0,
            "adaptation_validation": 0,
            "other": 0,
        }
        for result in self.evaluation_history:
            reason = result.rejection_reason
            if reason is None:
                continue
            if reason.startswith("Candidate retention validation MSE"):
                rejection_counts["retention"] += 1
            elif reason.startswith("Candidate did not improve held-out adaptation"):
                rejection_counts["adaptation_validation"] += 1
            else:
                rejection_counts["other"] += 1
        return {
            "total_candidates_evaluated": len(self.evaluation_history),
            "accepted_count": self.accepted_promotions_count,
            "rejected_count": self.rejected_count,
            "rejection_counts": rejection_counts,
            "acceptance_rate": (
                self.accepted_promotions_count / len(self.evaluation_history)
                if self.evaluation_history else 0.0
            ),
        }

