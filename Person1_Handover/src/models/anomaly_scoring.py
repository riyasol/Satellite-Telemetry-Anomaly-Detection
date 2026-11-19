import numpy as np
import torch
from torch.utils.data import DataLoader
from src.models.lstm_forecaster import LSTMForecaster
from src.models.window_dataset import ForecastWindowDataset

def exponential_moving_average(scores, alpha=0.2):
    scores = np.asarray(scores, dtype=float)
    if not len(scores): return scores
    out = np.empty_like(scores)
    out[0] = scores[0]
    for i in range(1, len(scores)):
        out[i] = alpha * scores[i] + (1-alpha) * out[i-1]
    return out

def ema_reset_on_excluded(scores, eligible, alpha=0.2):
    """EMA per contiguous eligible run; excluded windows reset its state."""
    scores = np.asarray(scores, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    out = np.full(len(scores), np.nan, dtype=float)
    state = None
    for i, score in enumerate(scores):
        if not eligible[i]:
            state = None
            continue
        state = score if state is None else alpha * score + (1-alpha) * state
        out[i] = state
    return out

def window_has_anomaly(labels, starts, input_window, horizon, include_input=True):
    labels = np.asarray(labels, dtype=np.int8)
    starts = np.asarray(starts, dtype=np.int64)
    left = starts if include_input else starts + input_window
    right = starts + input_window + horizon
    bad = (labels == 1).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(bad)))
    return (prefix[right] - prefix[left]) > 0

def load_forecaster(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    c = checkpoint["config"]
    model = LSTMForecaster(checkpoint["input_size"], c["hidden_size"], c["num_layers"],
                           c["dropout"], c["forecast_horizon"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

@torch.no_grad()
def score_windows(model, values, start, end, input_window, horizon,
                  batch_size=256, device="cpu", stride=1):
    ds = ForecastWindowDataset(values, start, end, input_window, horizon, stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    indices, scores, forecasts, targets = [], [], [], []
    for x, y, idx in loader:
        pred = model(x.to(device)).cpu().numpy()
        target = y.numpy()
        err = np.mean((target-pred)**2, axis=(1,2))
        indices.extend(idx.numpy().tolist())
        scores.extend(err.tolist())
        forecasts.append(pred); targets.append(target)
    return (np.asarray(indices), np.asarray(scores),
            np.concatenate(forecasts), np.concatenate(targets))
