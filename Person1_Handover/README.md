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
3. The supplied processed blocks contain 338,689 Mission 1 observations and
   645,191 Mission 2 observations. These counts come from the packaged NPZ
   files; raw ESA source verification is still pending. A requested 112-day
   span does not imply a fixed row count because the source timestamps are
   irregular.
4. **Confirm the guide/team-approved pre/post anomaly masking margins.**
   Preprocessing currently uses zero seconds as a provisional placeholder;
   drift masking separately uses 60 seconds. Neither value is approved.
   Do not regenerate or reinterpret existing splits until the decision is
   recorded.

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

Existing checkpoints at this location are retained for evaluation. New
training runs write to `results/models/<mission-lowercase>/runs/<UTC-run-id>/`.
Replacing the legacy checkpoint requires the explicit
`--overwrite-checkpoint` option.

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
- The existing note that Mission 1 `id_145` crosses the validation/test
  boundary in Block 33 conflicts with the supplied NPZ metadata, which lists no
  crossing IDs and has an all-zero `evaluation_ignore_mask`. Raw labels are
  unavailable, so verify this discrepancy from the authoritative source before
  claiming either result.
- Unlabelled timestamps are treated as normal under the current interval-label
  interpretation. Confirm that this matches the dataset annotation semantics.

## Synthetic demo

The original synthetic CSV scripts remain for smoke testing only. They are not
used by the ESA pipeline.

## P1 ESA data contract

P1 owns loading, preprocessing, label separation, split construction, window
eligibility and data-integrity validation. P2 owns the final streaming
`DecisionPacket`. P3 owns replay and candidate training.

### Required local ESA assets

Place trusted ESA-ADB archives locally under:

```text
data/raw/ESA-Mission1/
data/raw/ESA-Mission2/
```

Do not commit `data/`, raw channel ZIPs, `*.npz`, `*.npy`, virtual
environments or `*.pt` checkpoints.

### P1 modules

- `src/data/esa_loader.py`: exact timestamp-aligned channel loading.
- `src/data/esa_labels.py`: exact ground-truth annotations and expanded masks.
- `src/data/prepare_esa.py`: block selection, normalization and chronological splits.
- `src/data/validate_processed.py`: structural and integrity validation.
- `src/data/generate_manifest.py`: data/window manifest generation.
- `src/drift/window_eligibility.py`: complete input-plus-horizon eligibility.
- `src/models/evaluate_b1.py`: validation threshold and held-out B1 output.

### Execution

From `Person1_Handover`:

```powershell
python -m src.data.prepare_esa
python -m src.data.validate_processed
python -m src.models.train
python -m src.models.evaluate_b1
python -m src.data.generate_manifest
python -m unittest discover -s tests -v
```

The P1 commands do not validate or reproduce the B1 checkpoint unless the
local processed data and checkpoint are present. To reproduce the baseline,
first verify the raw-source/processed-data manifest and checkpoint hashes,
then run only `python -m src.models.evaluate_b1`; training is not required.

### Label and test protocol

The `labels` array contains exact ESA interval annotations. The
`exclusion_mask` is the expanded training/evaluation exclusion mask.
`evaluation_ignore_mask` identifies annotations crossing train/validation/test
boundaries and is used only for offline window exclusion.

Ground-truth labels must never be passed into operational masking, drift
detection or candidate-update decisions. The final test split is for offline
evaluation only.

Point/window F0.5 and event-wise F0.5 are separate metrics. The dedicated
`python -m src.evaluation.esa_event_evaluation` pipeline writes one JSON report
per mission. Its current two-window tolerance and one-to-one matching rule are
provisional; final event metrics remain pending raw-source verification and
guide/team approval.

### Complete-window eligibility

A candidate or replay window starting at observation `i` spans:

```text
[i, i + input_window + forecast_horizon)
```

The candidate is eligible only if every observation in that complete span is
outside the P1 observation exclusion mask.

P1 provides P2/P3 with:

- start index
- input start/end
- target start/end
- timestamp
- exact/final anomaly state as available
- validity
- eligibility
- mask reason
