"""Replay buffer for continual learning and Strong Experience Replay (SER).

Stores eligible normal historical windows and newly accepted normal operational windows.
Never stores anomaly-masked or contaminated windows.
"""
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch


class ReplayBuffer:
    """Stores and samples historical normal input-forecast pairs for experience replay."""

    def __init__(self, capacity: int = 1000, seed: int = 42):
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        # In-memory storage for (x, y) tensors
        self.buffer_x: List[np.ndarray] = []
        self.buffer_y: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.total_added = 0

    def size(self) -> int:
        return len(self.buffer_x)

    def add(
        self,
        x: np.ndarray,
        y: np.ndarray,
        is_anomaly_masked: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add an eligible sample to the replay buffer.

        Safety constraint: if is_anomaly_masked is True, the sample is strictly rejected.
        """
        if is_anomaly_masked:
            # Replay buffer must never contain anomaly contaminated samples
            return False

        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if len(self.buffer_x) < self.capacity:
            self.buffer_x.append(x_arr)
            self.buffer_y.append(y_arr)
            self.metadata.append(meta or {})
        else:
            # Reservoir replacement policy
            replace_idx = self.rng.randint(0, self.capacity - 1)
            self.buffer_x[replace_idx] = x_arr
            self.buffer_y[replace_idx] = y_arr
            self.metadata[replace_idx] = meta or {}

        self.total_added += 1
        return True

    def populate_from_training_split(
        self,
        values: np.ndarray,
        exclusion_mask: np.ndarray,
        train_start: int,
        train_end: int,
        input_window: int = 250,
        forecast_horizon: int = 10,
        stride: int = 50,
        max_samples: int = 500,
    ):
        """Seed the replay buffer with initial normal training windows."""
        for start_idx in range(train_start, train_end - input_window - forecast_horizon, stride):
            if self.size() >= max_samples:
                break
            x_end = start_idx + input_window
            y_end = x_end + forecast_horizon

            # Check if any point in x or y intersects the exclusion mask
            if np.any(exclusion_mask[start_idx:y_end]):
                continue

            x = values[start_idx:x_end]
            y = values[x_end:y_end]
            self.add(x, y, is_anomaly_masked=False, meta={"source": "initial_train", "start_idx": start_idx})

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of historical windows uniformly at random."""
        cur_size = len(self.buffer_x)
        if cur_size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        sample_size = min(batch_size, cur_size)
        indices = self.rng.sample(range(cur_size), sample_size)

        batch_x = np.stack([self.buffer_x[i] for i in indices], axis=0)
        batch_y = np.stack([self.buffer_y[i] for i in indices], axis=0)

        return torch.from_numpy(batch_x), torch.from_numpy(batch_y)

    def save(self, file_path: Union[str, Path]):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            buffer_x=np.array(self.buffer_x, dtype=np.float32),
            buffer_y=np.array(self.buffer_y, dtype=np.float32),
            total_added=np.array(self.total_added),
        )

    def load(self, file_path: Union[str, Path]):
        path = Path(file_path)
        data = np.load(path)
        self.buffer_x = list(data["buffer_x"])
        self.buffer_y = list(data["buffer_y"])
        self.total_added = int(data["total_added"])

