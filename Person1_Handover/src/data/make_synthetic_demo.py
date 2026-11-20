import numpy as np
import pandas as pd
from src.utils.config import resolve_path

def main():
    rng = np.random.default_rng(42)
    n = 5000
    t = np.arange(n)
    data = {}
    for j in range(6):
        data[f"channel_{j+1}"] = (
            np.sin(t / (70 + 7*j)) + 0.15*np.sin(t / (9 + j)) +
            rng.normal(0, 0.08, n)
        )
    labels = np.zeros(n, dtype=int)
    labels[3100:3140] = 1
    for j in range(6):
        data[f"channel_{j+1}"][3100:3140] += 2.5
    df = pd.DataFrame(data)
    df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=n, freq="min"))
    df["anomaly"] = labels
    path = resolve_path("data/raw/synthetic_demo.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Synthetic software-test data written to {path}")
    print("For this file only, set channel_columns to channel_1 ... channel_6 and channel_indices: [].")

if __name__ == "__main__":
    main()
