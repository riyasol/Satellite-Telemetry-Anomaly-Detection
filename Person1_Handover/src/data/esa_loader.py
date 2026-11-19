"""Memory-conscious ESA-ADB Mission 1 and Mission 2 channel loader."""
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


def load_channel_zip(zip_path: Path, expected_name: str) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"{zip_path}: expected one data member, found {members}")
        payload = archive.read(members[0])

    # ESA archives are trusted project inputs. Pickle loading can execute code,
    # so only load data from an authorized dataset source.
    obj = pd.read_pickle(BytesIO(payload))
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{zip_path}: expected DataFrame, got {type(obj)}")
    if obj.shape[1] != 1:
        raise ValueError(f"{zip_path}: expected one column, got {obj.shape}")
    if obj.columns[0] != expected_name:
        raise ValueError(f"{zip_path}: expected column {expected_name}, got {obj.columns[0]}")
    return obj


def mission_channels(mission: str) -> list[str]:
    if mission == "Mission1":
        return [f"channel_{i}" for i in range(41, 47)]
    if mission == "Mission2":
        return [f"channel_{i}" for i in range(18, 29)]
    raise ValueError(f"Unsupported mission: {mission}")


def load_mission_matrix(raw_root: Path, mission: str, output_dir: Path) -> dict:
    """Write aligned source channels to float32 and int64 UTC memmaps."""
    folder = raw_root / f"ESA-{mission}"
    channels = mission_channels(mission)

    metadata_path = folder / "channels.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    meta = pd.read_csv(metadata_path).set_index("Channel")
    missing_metadata = [channel for channel in channels if channel not in meta.index]
    if missing_metadata:
        raise ValueError(f"{mission}: missing channel metadata {missing_metadata}")

    targets = meta.loc[channels, "Target"].astype(str).str.upper()
    if not targets.eq("YES").all():
        raise ValueError(f"{mission}: selected channels not all Target=YES: {targets.to_dict()}")

    first = load_channel_zip(folder / "channels" / f"{channels[0]}.zip", channels[0])
    n_rows = len(first)
    if not first.index.is_monotonic_increasing or first.index.has_duplicates:
        raise ValueError(f"{mission}/{channels[0]} timestamps must be sorted and unique")

    canonical_index = first.index
    canonical_index = (
        canonical_index.tz_localize("UTC")
        if canonical_index.tz is None
        else canonical_index.tz_convert("UTC")
    )
    timestamps = canonical_index.asi8.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"{mission.lower()}_values.npy"
    timestamp_path = output_dir / f"{mission.lower()}_timestamps.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(n_rows, len(channels))
    )
    timestamp_map = np.lib.format.open_memmap(
        timestamp_path, mode="w+", dtype=np.int64, shape=(n_rows,)
    )
    timestamp_map[:] = timestamps

    first_values = pd.to_numeric(first.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(first_values).all():
        raise ValueError(f"{mission}/{channels[0]} contains NaN or infinite telemetry values")
    matrix[:, 0] = first_values
    del first, first_values

    for column, channel in enumerate(channels[1:], start=1):
        frame = load_channel_zip(folder / "channels" / f"{channel}.zip", channel)
        if len(frame) != n_rows:
            raise ValueError(f"{mission}/{channel}: row count {len(frame)} != {n_rows}")

        channel_index = frame.index
        channel_index = (
            channel_index.tz_localize("UTC")
            if channel_index.tz is None
            else channel_index.tz_convert("UTC")
        )
        if not np.array_equal(channel_index.asi8, timestamps):
            raise ValueError(
                f"{mission}/{channel}: timestamps differ from {channels[0]}; "
                "do not align channels by row position"
            )

        channel_values = pd.to_numeric(
            frame.iloc[:, 0], errors="coerce"
        ).to_numpy(dtype=np.float32)
        if not np.isfinite(channel_values).all():
            raise ValueError(f"{mission}/{channel} contains NaN or infinite telemetry values")
        matrix[:, column] = channel_values
        del frame

    matrix.flush()
    timestamp_map.flush()
    del matrix, timestamp_map

    return {
        "mission": mission,
        "channels": channels,
        "rows": int(n_rows),
        "matrix_path": str(matrix_path),
        "timestamps_path": str(timestamp_path),
        "start_utc": pd.Timestamp(timestamps[0], tz="UTC").isoformat(),
        "end_utc": pd.Timestamp(timestamps[-1], tz="UTC").isoformat(),
        "median_gap_seconds": float(np.median(np.diff(timestamps) / 1e9)),
        "target_metadata": targets.to_dict(),
    }
