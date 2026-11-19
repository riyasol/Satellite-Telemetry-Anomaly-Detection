import numpy as np
import torch
from torch.utils.data import Dataset

class ForecastWindowDataset(Dataset):
    """Windows remain entirely inside [start, end); indices denote input start."""
    def __init__(self, values, start, end, input_window=250, horizon=10,
                 excluded_labels=None, exclude_anomaly_windows=False, stride=1):
        self.values = values
        self.start, self.end = int(start), int(end)
        self.input_window, self.horizon = int(input_window), int(horizon)
        self.stride = max(1, int(stride))
        last_exclusive = self.end - self.input_window - self.horizon + 1
        if last_exclusive <= self.start:
            raise ValueError("Split too short for input_window + forecast_horizon.")
        starts = np.arange(self.start, last_exclusive, self.stride, dtype=np.int64)
        if exclude_anomaly_windows and excluded_labels is not None:
            labels = np.asarray(excluded_labels, dtype=np.int8)
            # Prefix sums avoid a Python loop over hundreds of thousands of windows.
            bad = (labels == 1).astype(np.int64)
            prefix = np.concatenate(([0], np.cumsum(bad)))
            ends = starts + self.input_window + self.horizon
            counts = prefix[ends] - prefix[starts]
            starts = starts[counts == 0]
        if not len(starts):
            raise ValueError("No usable windows remain after filtering.")
        self.valid_starts = starts

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        i = int(self.valid_starts[int(idx)])
        x = self.values[i:i+self.input_window]
        y = self.values[i+self.input_window:i+self.input_window+self.horizon]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.float32)), i

