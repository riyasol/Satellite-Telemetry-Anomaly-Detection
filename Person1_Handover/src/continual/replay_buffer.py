"""Replay buffer for continual learning and Strong Experience Replay (SER).

Stores eligible normal historical windows and newly accepted normal operational windows.
Never stores anomaly-masked or contaminated windows.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import torch


class ReplayBuffer:
    """Stores and samples historical normal input-forecast pairs for experience replay."""

    def __init__(self, capacity: int = 1000, seed: int = 42):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")
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
        if x_arr.ndim != 2 or y_arr.ndim != 2:
            raise ValueError("Replay samples must be 2-D input and target windows.")
        if x_arr.shape[1] != y_arr.shape[1]:
            raise ValueError("Replay input and target windows must have the same feature count.")
        if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
            raise ValueError("Replay samples must contain only finite values.")

        if len(self.buffer_x) < self.capacity:
            self.buffer_x.append(x_arr)
            self.buffer_y.append(y_arr)
            self.metadata.append(meta or {})
        else:
            # Reservoir sampling gives every observed sample an equal chance to remain.
            replace_idx = self.rng.randrange(self.total_added + 1)
            if replace_idx < self.capacity:
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
        if stride <= 0 or max_samples < 0:
            raise ValueError("stride must be positive and max_samples non-negative.")
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("values must be a two-dimensional telemetry array.")
        if input_window <= 0 or forecast_horizon <= 0:
            raise ValueError("input_window and forecast_horizon must be positive.")
        if not 0 <= train_start < train_end <= len(values):
            raise ValueError("Invalid training split bounds.")
        exclusion = np.asarray(exclusion_mask, dtype=bool)
        if exclusion.ndim != 1 or len(exclusion) != len(values):
            raise ValueError("exclusion_mask must be one-dimensional and match values.")
        added = 0
        last_start = train_end - input_window - forecast_horizon
        for start_idx in range(train_start, last_start + 1, stride):
            if added >= max_samples:
                break
            x_end = start_idx + input_window
            y_end = x_end + forecast_horizon

            # Eligibility covers the complete input-plus-target span.
            if np.any(exclusion[start_idx:y_end]):
                continue

            x = values[start_idx:x_end]
            y = values[x_end:y_end]
            self.add(
                x,
                y,
                is_anomaly_masked=False,
                meta={"source": "initial_train", "start_idx": start_idx},
            )
            added += 1

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of historical windows uniformly at random."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
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
            metadata_json=np.array(json.dumps(self.metadata)),
            capacity=np.array(self.capacity),
            seed=np.array(self.seed),
        )

    def load(self, file_path: Union[str, Path]):
        path = Path(file_path)
        data = np.load(path, allow_pickle=False)
        self.buffer_x = list(data["buffer_x"])
        self.buffer_y = list(data["buffer_y"])
        self.total_added = int(data["total_added"])
        if "metadata_json" in data.files:
            self.metadata = json.loads(str(data["metadata_json"]))
        else:
            self.metadata = [{} for _ in self.buffer_x]
        if "capacity" in data.files:
            self.capacity = int(data["capacity"])
        if "seed" in data.files:
            self.seed = int(data["seed"])
            self.rng = random.Random(self.seed)

