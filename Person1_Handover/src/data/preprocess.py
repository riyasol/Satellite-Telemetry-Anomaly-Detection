import json
import numpy as np
import pandas as pd
from src.utils.config import load_config, resolve_path, ensure_dirs

def select_channels(df, dcfg):
    names = dcfg.get("channel_columns") or []
    if names:
        missing = [c for c in names if c not in df.columns]
        if missing:
            raise ValueError(f"Configured channel columns not found: {missing}")
        return names
    indices = dcfg.get("channel_indices") or []
    if not indices:
        raise ValueError("Set channel_columns or channel_indices in config.")
    base = int(dcfg.get("channel_index_base", 1))
    positions = [int(i) - base for i in indices]
    if min(positions) < 0 or max(positions) >= len(df.columns):
        raise ValueError(f"Channel positions {indices} are outside the {len(df.columns)} CSV columns.")
    chosen = [df.columns[i] for i in positions]
    print("WARNING: selected by CSV column position. Confirm these are the intended telemetry channels:")
    print(chosen)
    return chosen

def main():
    cfg = load_config()
    ensure_dirs(cfg)
    dcfg = cfg["data"]
    path = resolve_path(dcfg["csv_path"])
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}. Edit config/config.yaml.")
    df = pd.read_csv(path)

    time_col = dcfg.get("timestamp_column")
    label_col = dcfg.get("label_column")
    for col in (time_col, label_col):
        if col and col not in df.columns:
            raise ValueError(f"Configured column '{col}' is not in CSV.")

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
        if df[time_col].isna().any():
            raise ValueError("Some timestamps could not be parsed. Resolve before continuing.")

    channels = select_channels(df, dcfg)
    telemetry = df[channels].apply(pd.to_numeric, errors="coerce")
    labels = None
    if label_col:
        raw_labels = df[label_col]
        anomaly_values = [str(x).strip().lower() for x in dcfg.get("anomaly_values", [])]
        labels = raw_labels.astype(str).str.strip().str.lower().isin(anomaly_values).astype(np.int8).to_numpy()

    telemetry = telemetry.replace([np.inf, -np.inf], np.nan)
    telemetry = telemetry.ffill()
    valid = telemetry.notna().all(axis=1).to_numpy()
    if not valid.all():
        if dcfg.get("drop_rows_still_missing", True):
            telemetry = telemetry.loc[valid].reset_index(drop=True)
            if labels is not None:
                labels = labels[valid]
            if time_col:
                df = df.loc[valid].reset_index(drop=True)
        else:
            raise ValueError("Missing telemetry remains after forward fill. Resolve it or enable row dropping.")

    values = telemetry.to_numpy(dtype=np.float32)
    n = len(values)
    if n < cfg["model"]["input_window"] + cfg["model"]["forecast_horizon"] + 10:
        raise ValueError(f"Not enough rows ({n}) for configured input/horizon windows.")
    train_f = float(dcfg["train_fraction"])
    val_f = float(dcfg["validation_fraction"])
    test_f = float(dcfg["test_fraction"])
    if not np.isclose(train_f + val_f + test_f, 1.0):
        raise ValueError("Train/validation/test fractions must sum to 1.")
    train_end = int(n * train_f)
    val_end = train_end + int(n * val_f)
    row_indices = {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, n]}

    # Fit standardization on normal training rows only.
    train_values = values[:train_end]
    if labels is not None:
        normal_train = train_values[labels[:train_end] == 0]
        if len(normal_train) == 0:
            raise ValueError("No normal observations in training split.")
        fit_values = normal_train
    else:
        fit_values = train_values
        print("WARNING: no anomaly label column configured; scaler is fitted on all training rows.")
    mean = fit_values.mean(axis=0)
    std = fit_values.std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = ((values - mean) / std).astype(np.float32)

    outdir = resolve_path(cfg["output"]["processed_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    prepared = pd.DataFrame(normalized, columns=channels)
    if time_col:
        prepared.insert(0, time_col, df[time_col].astype(str).to_numpy())
    if labels is not None:
        prepared["label"] = labels
    prepared.to_csv(outdir / "telemetry_prepared.csv", index=False)

    np.savez_compressed(
        outdir / "splits.npz",
        values=normalized,
        labels=labels if labels is not None else np.full(n, -1, dtype=np.int8),
        has_labels=np.array(labels is not None),
        train_bounds=np.array(row_indices["train"]),
        validation_bounds=np.array(row_indices["validation"]),
        test_bounds=np.array(row_indices["test"]),
        channel_names=np.array(channels),
    )
    metadata = {
        "source_file": str(path), "rows_after_cleaning": n, "channels": channels,
        "timestamp_column": time_col, "label_column": label_col,
        "has_labels": labels is not None, "train_bounds": row_indices["train"],
        "validation_bounds": row_indices["validation"], "test_bounds": row_indices["test"],
        "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
        "anomaly_count": int(labels.sum()) if labels is not None else None,
        "note": "Chronological split; scaler fitted on normal training rows only."
    }
    (outdir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {n} rows and {len(channels)} channels.")
    print(f"Splits: train={train_end}, validation={val_end-train_end}, test={n-val_end}")
    print(f"Saved outputs under {outdir}")

if __name__ == "__main__":
    main()
