"""Verify exact timestamp identity across selected channel archives."""
from pathlib import Path
import zipfile
from io import BytesIO
import numpy as np
import pandas as pd
from src.utils.config import load_config, resolve_path
from src.data.esa_loader import mission_channels, load_channel_zip


def main():
    cfg = load_config()
    root = resolve_path(cfg["data"]["esa_raw_root"])
    for mission in cfg["data"]["missions"]:
        folder = root / f"ESA-{mission}"
        channels = [f"channel_{i}" for i in mission_channels(mission)]
        reference = None
        print(f"\n{mission}")
        for ch in channels:
            frame = load_channel_zip(folder / "channels" / f"{ch}.zip", ch)
            idx = frame.index
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            ns = idx.asi8
            if reference is None:
                reference = ns.copy()
                status = "reference"
            else:
                status = "EXACT MATCH" if np.array_equal(reference, ns) else "MISMATCH"
            print(f"{ch}: {status}; rows={len(ns):,}")
            if status == "MISMATCH":
                raise RuntimeError(f"{mission}: timestamp mismatch at {ch}; stop before preprocessing.")
            del frame, ns
        print("All selected timestamps match exactly.")
if __name__ == "__main__":
    main()
