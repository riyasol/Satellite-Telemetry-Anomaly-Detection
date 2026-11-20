import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, resolve_path
from src.models.lstm_forecaster import LSTMForecaster
from src.models.window_dataset import ForecastWindowDataset


def create_run_directory(model_root, mission, run_id):
    run_dir = model_root / mission.lower() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_checkpoint_safely(checkpoint, destination, overwrite=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        created = False
        try:
            with destination.open("xb") as output_file:
                created = True
                torch.save(checkpoint, output_file)
        except Exception:
            if created and destination.exists():
                destination.unlink()
            raise
        return

    temporary_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("xb") as output_file:
            torch.save(checkpoint, output_file)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_epoch(model, loader, criterion, device, optimizer=None,
              max_batches=None, phase="Training"):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    start_time = time.perf_counter()
    total_batches = len(loader)

    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    for batch_number, (x, y, _) in enumerate(loader, start=1):
        if max_batches is not None and batch_number > max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(x)
            loss = criterion(predictions, y)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_number % 10 == 0 or batch_number == total_batches:
            elapsed = time.perf_counter() - start_time
            print(
                f"  {phase}: batch {batch_number}/{total_batches} "
                f"({100 * batch_number / total_batches:.1f}%) "
                f"loss={total_loss / total_samples:.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True
            )

    return total_loss / total_samples, time.perf_counter() - start_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="Explicitly replace the legacy lstm_b1.pt checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_config()
    seed = int(cfg["seed"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cpu":
        thread_count = min(4, os.cpu_count() or 1)
        torch.set_num_threads(thread_count)
        print(f"CPU threads: {thread_count}", flush=True)

    print(f"Device: {device}", flush=True)

    m = cfg["model"]
    missions = cfg["data"].get("missions", ["Mission1"])

    epochs = 1 if args.smoke_test else int(m["epochs"])
    max_batches = 10 if args.smoke_test else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    if args.smoke_test:
        print("\nSMOKE TEST: maximum 10 train and validation batches per mission.")
        print("No model checkpoints will be overwritten.\n", flush=True)
    elif args.overwrite_checkpoint:
        print(
            "\nWARNING: --overwrite-checkpoint explicitly replaces the legacy "
            "checkpoint for each configured mission.\n",
            flush=True,
        )

    for mission in missions:
        print(f"\n{'=' * 55}\nStarting {mission}\n{'=' * 55}", flush=True)

        mission_start = time.perf_counter()
        pdir = resolve_path(cfg["data"]["esa_work_dir"]) / mission.lower()

        data = np.load(pdir / "splits.npz", allow_pickle=True)

        values = data["values"]
        labels = data["exclusion_mask"]
        channel_names = data["channel_names"].tolist()

        train_start, train_end = data["train_bounds"].tolist()
        val_start, val_end = data["validation_bounds"].tolist()

        stride = int(m.get("window_stride", 10))

        print("Preparing training windows...", flush=True)
        train_ds = ForecastWindowDataset(
            values, train_start, train_end,
            m["input_window"], m["forecast_horizon"],
            excluded_labels=labels,
            exclude_anomaly_windows=True,
            stride=stride
        )

        print("Preparing validation windows...", flush=True)
        val_ds = ForecastWindowDataset(
            values, val_start, val_end,
            m["input_window"], m["forecast_horizon"],
            excluded_labels=labels,
            exclude_anomaly_windows=True,
            stride=stride
        )

        print(
            f"Windows ready: train={len(train_ds):,}, "
            f"validation={len(val_ds):,}",
            flush=True
        )

        train_loader = DataLoader(
            train_ds, batch_size=m["batch_size"],
            shuffle=True, num_workers=0, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=m["batch_size"],
            shuffle=False, num_workers=0, drop_last=False
        )

        model = LSTMForecaster(
            values.shape[1],
            m["hidden_size"],
            m["num_layers"],
            m["dropout"],
            m["forecast_horizon"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m["learning_rate"],
            weight_decay=m.get("weight_decay", 0.0)
        )

        criterion = torch.nn.MSELoss()
        history = []
        best_val = float("inf")
        best_epoch = 0
        stale = 0
        best_checkpoint = None
        patience = int(m.get("early_stopping_patience", 5))

        if args.smoke_test:
            run_dir = None
        else:
            model_root = resolve_path(cfg["output"]["model_dir"])
            mission_model_dir = model_root / mission.lower()
            run_dir = create_run_directory(
                model_root,
                mission,
                run_id,
            )

        for epoch in range(1, epochs + 1):
            print(f"\n{mission} | Epoch {epoch}/{epochs}", flush=True)

            train_loss, train_seconds = run_epoch(
                model, train_loader, criterion, device,
                optimizer=optimizer,
                max_batches=max_batches,
                phase="Training"
            )

            val_loss, val_seconds = run_epoch(
                model, val_loader, criterion, device,
                optimizer=None,
                max_batches=max_batches,
                phase="Validation"
            )

            print(
                f"{mission} epoch {epoch}: "
                f"train_mse={train_loss:.6f}, "
                f"validation_mse={val_loss:.6f}, "
                f"train_time={train_seconds:.1f}s, "
                f"validation_time={val_seconds:.1f}s",
                flush=True
            )

            history.append({
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": val_loss,
                "train_seconds": train_seconds,
                "validation_seconds": val_seconds
            })

            if args.smoke_test:
                continue

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                stale = 0

                if not args.smoke_test:
                    best_checkpoint = {
                        "model_state_dict": {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in model.state_dict().items()
                        },
                        "input_size": values.shape[1],
                        "config": m,
                        "best_validation_mse": best_val,
                        "best_epoch": best_epoch,
                        "channel_names": channel_names,
                        "mission": mission,
                        "seed": seed
                    }
            else:
                stale += 1
                if stale >= patience:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best epoch was {best_epoch}.",
                        flush=True
                    )
                    break

        if not args.smoke_test:
            if best_checkpoint is None:
                raise RuntimeError(
                    f"{mission}: training completed without a best checkpoint."
                )

            checkpoint_path = (
                mission_model_dir / "lstm_b1.pt"
                if args.overwrite_checkpoint
                else run_dir / "lstm_b1.pt"
            )
            save_checkpoint_safely(
                best_checkpoint,
                checkpoint_path,
                overwrite=args.overwrite_checkpoint,
            )
            pd.DataFrame(history).to_csv(
                run_dir / "training_history.csv", index=False
            )

            metadata = json.loads(
                (pdir / "preprocessing_metadata.json").read_text(
                    encoding="utf-8"
                )
            )

            (run_dir / "scaler.json").write_text(
                json.dumps({
                    "mean": metadata["scaler_mean"],
                    "std": metadata["scaler_std"],
                    "channels": metadata["channels"]
                }, indent=2),
                encoding="utf-8"
            )

            print(
                f"\n{mission}: best validation MSE={best_val:.6f}, "
                f"best epoch={best_epoch}; checkpoint={checkpoint_path}",
                flush=True
            )

        print(
            f"{mission} completed in "
            f"{(time.perf_counter() - mission_start) / 60:.1f} minutes.",
            flush=True
        )

        data.close()

    if args.smoke_test:
        print("\nSMOKE TEST COMPLETED. No final checkpoints were changed.")
    else:
        print("\nALL MISSION TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
