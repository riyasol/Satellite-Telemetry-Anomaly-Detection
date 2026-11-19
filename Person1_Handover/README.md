# Spacecraft Telemetry — Person 1 Implementation

This project prepares the selected ESA-ADB channels and implements Person 1's
static LSTM forecasting baseline (B1), residual scoring, EMA, thresholding,
and evaluation.

## Selected channels

- Mission 1: `channel_41` through `channel_46` (all reported `Target=YES`).

Sampling interval is 30 s nominal (observation index based).

## Before running

1. Keep the official extracted folders under `data/raw/ESA-Mission1`.
2. Review `config/config.yaml`.
3. Mission 1 uses 112 days (16-week block starting 2011-01-15T00:00:00Z) with 322,560 samples.
5. **Confirm the professor-approved pre/post anomaly masking margins.** The
   supplied project notes say a surrounding margin is required but do not state
   its duration. Current config defaults to zero seconds as a placeholder.
   Do not present this as final until confirmed.

## Run

```powershell
python -m src.data.prepare_esa
python -m src.models.train
python -m src.models.evaluate_b1
```

Or run `.\run_person1_pipeline.ps1`.

The ESA loader reads the single Pickle member from each official channel ZIP
directly; it does not extract duplicate channel files. It verifies that all
selected channels within a mission have exactly matching timestamps and stops
if they do not. Original timestamps are retained; no resampling is performed.
Thus each LSTM step is an observation index, not a guaranteed fixed duration.

## Outputs

For each mission, under `data/processed/esa_adb/<mission-lowercase>/`:
- `splits.npz`
- `preprocessing_metadata.json`
- `mission*_values.npy` and `mission*_timestamps.npy` (memmap source arrays)

For each mission, under `results/models/<mission-lowercase>/`:
- `lstm_b1.pt`
- `scaler.json`
- `training_history.csv`

For each mission, under `results/<mission-lowercase>/`:
- `b1_metrics.json`
- `b1_scores.csv`

Plots are under `results/plots/<mission-lowercase>/`.

## Important methodological notes

- The scaler is fitted only on training rows outside the expanded exclusion mask.
- Exact ground-truth labels and margin-expanded exclusion masks are stored separately.
- Training and validation windows overlapping the exclusion mask are excluded.
- Test scoring excludes target windows intersecting annotations that cross a split boundary.
- A test window is labelled anomalous when any point in its forecast target
  horizon is annotated anomalous.
- Threshold is the configured quantile of EMA scores from contiguous normal
  validation windows. EMA state resets over excluded anomalous validation
  windows to avoid contamination across masked intervals.
- The reported F0.5 is point-window F0.5, not event-wise F0.5. Event matching
  rules must be agreed with the group before reporting event-level results.
- `window_stride: 10` is a compute-saving initial setting. Report it and keep it
  consistent across experiments. Set to 1 only if compute permits and the team
  wants dense overlapping windows.
- Mission 1 `id_145` crosses the validation/test boundary in Block 33; it is
  retained in exact labels and metadata but excluded from validation/test metrics.
- Unlabelled timestamps are treated as normal under the current interval-label
  interpretation. Confirm that this matches the dataset annotation semantics.

## Synthetic demo

The original synthetic CSV scripts remain for smoke testing only. They are not
used by the ESA pipeline.
