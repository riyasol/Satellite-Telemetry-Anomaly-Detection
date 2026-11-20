"""Candidate model generator and continual learner.

Creates separate candidate checkpoints without overwriting the current accepted model.
Implements Strong Experience Replay (SER) by mixing recent eligible normal samples
with historical replay samples.
"""
from __future__ import annotations
import copy
import re
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.lstm_forecaster import LSTMForecaster
from src.continual.replay_buffer import ReplayBuffer


class CandidateTrainer:
    """Trains candidate models for Fine-Tune or Strong-Update requests."""

    def __init__(
        self,
        base_model: LSTMForecaster,
        device: torch.device,
        replay_buffer: ReplayBuffer,
        checkpoints_dir: Path,
        candidate_namespace: str = "m1",
    ):
        self.device = device
        self.replay_buffer = replay_buffer
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_namespace = re.sub(
            r"[^A-Za-z0-9_-]+", "_", candidate_namespace
        ).strip("_-") or "candidate"
        self.candidate_count = sum(
            1
            for path in self.checkpoints_dir.glob(
                f"{self.candidate_namespace}_*_*.pt"
            )
        )

    def create_candidate(
        self,
        current_model: LSTMForecaster,
        recent_x: np.ndarray,
        recent_y: np.ndarray,
        action_type: str = "FINE_TUNE",
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        replay_ratio: float = 0.3,
        batch_size: int = 64,
    ) -> Tuple[LSTMForecaster, Dict[str, Any], Path]:
        """Train a candidate model using recent eligible telemetry + SER replay buffer."""
        action_type = action_type.upper()
        if action_type not in {"FINE_TUNE", "STRONG_UPDATE"}:
            raise ValueError(f"Unsupported candidate action: {action_type}")
        if epochs is not None and int(epochs) <= 0:
            raise ValueError("epochs must be positive.")
        if lr is not None and (not np.isfinite(lr) or float(lr) <= 0):
            raise ValueError("lr must be finite and positive.")
        if not np.isfinite(replay_ratio) or not 0.0 <= float(replay_ratio) < 1.0:
            raise ValueError("replay_ratio must be in [0, 1).")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        recent_x = np.asarray(recent_x, dtype=np.float32)
        recent_y = np.asarray(recent_y, dtype=np.float32)
        if recent_x.ndim != 3 or recent_y.ndim != 3:
            raise ValueError("Candidate inputs and targets must be 3-D window arrays.")
        if len(recent_x) == 0 or len(recent_x) != len(recent_y):
            raise ValueError("Candidate inputs and targets must have equal, non-zero lengths.")
        if recent_x.shape[2] != current_model.input_size:
            raise ValueError("Candidate feature count does not match the current model.")
        if recent_y.shape[1:] != (current_model.horizon, current_model.input_size):
            raise ValueError("Candidate target shape does not match the current model.")
        if not np.isfinite(recent_x).all() or not np.isfinite(recent_y).all():
            raise ValueError("Candidate training data must contain only finite values.")

        self.candidate_count += 1
        candidate_id = (
            f"{self.candidate_namespace}_{self.candidate_count:03d}_"
            f"{action_type.lower()}"
        )
        start_time = time.perf_counter()

        # Clone current model weights as the starting point
        candidate_model = copy.deepcopy(current_model).to(self.device)
        candidate_model.train()

        # Update hyperparameters based on update strength
        if action_type == "STRONG_UPDATE":
            num_epochs = int(epochs) if epochs is not None else 6
            learning_rate = float(lr) if lr is not None else 0.001
            eff_replay_ratio = max(0.4, replay_ratio)
        else:
            num_epochs = int(epochs) if epochs is not None else 3
            learning_rate = float(lr) if lr is not None else 0.0003
            eff_replay_ratio = replay_ratio

        recent_t_x = torch.from_numpy(recent_x)
        recent_t_y = torch.from_numpy(recent_y)
        n_recent = len(recent_t_x)

        # Mix with replay buffer (SER)
        n_replay = int(n_recent * (eff_replay_ratio / (1.0 - eff_replay_ratio))) if eff_replay_ratio < 1.0 else n_recent
        n_replay = min(n_replay, self.replay_buffer.size())

        if n_replay > 0:
            replay_t_x, replay_t_y = self.replay_buffer.sample(n_replay)
            train_x = torch.cat([recent_t_x, replay_t_x], dim=0)
            train_y = torch.cat([recent_t_y, replay_t_y], dim=0)
        else:
            train_x = recent_t_x
            train_y = recent_t_y

        dataset = TensorDataset(train_x, train_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(candidate_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        epoch_losses = []
        for ep in range(num_epochs):
            total_loss = 0.0
            count = 0
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                optimizer.zero_grad()
                pred = candidate_model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

                bs = bx.size(0)
                total_loss += loss.item() * bs
                count += bs
            epoch_losses.append(total_loss / count if count else 0.0)

        elapsed = time.perf_counter() - start_time
        save_path = self.checkpoints_dir / f"{candidate_id}.pt"

        metadata = {
            "candidate_id": candidate_id,
            "action_type": action_type,
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "recent_samples_count": int(n_recent),
            "replay_samples_count": int(n_replay),
            "replay_ratio_actual": float(n_replay / (n_recent + n_replay)) if (n_recent + n_replay) else 0.0,
            "training_time_seconds": float(elapsed),
            "final_train_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "epoch_losses": epoch_losses,
        }

        # Save candidate checkpoint separately
        torch.save({
            "candidate_id": candidate_id,
            "model_state_dict": candidate_model.state_dict(),
            "metadata": metadata,
        }, save_path)

        return candidate_model, metadata, save_path

