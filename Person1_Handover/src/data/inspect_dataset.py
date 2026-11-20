import pandas as pd
from src.utils.config import load_config, resolve_path

def main():
    cfg = load_config()
    path = resolve_path(cfg["data"]["csv_path"])
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}\nEdit data.csv_path in config/config.yaml.")
    # Read a sample first; this avoids loading a very large telemetry file just to inspect schema.
    sample = pd.read_csv(path, nrows=1000)
    print(f"File: {path}")
    print(f"Sample shape (up to 1000 rows): {sample.shape}")
    print("\nColumns:")
    for i, col in enumerate(sample.columns, start=1):
        print(f"{i:4d}. {col} | dtype={sample[col].dtype} | sample={sample[col].iloc[0]!r}")
    print("\nMissing values in sample:")
    print(sample.isna().sum().to_string())
    out = resolve_path("results/dataset_inspection.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"File: {path}\nSample shape: {sample.shape}\n\nColumns:\n" +
        "\n".join(f"{i}. {c} ({sample[c].dtype})" for i, c in enumerate(sample.columns, 1)) +
        "\n\nMissing values:\n" + sample.isna().sum().to_string(),
        encoding="utf-8"
    )
    print(f"\nInspection summary saved to {out}")

if __name__ == "__main__":
    main()
