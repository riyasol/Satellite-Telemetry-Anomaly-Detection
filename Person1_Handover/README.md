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
local processed data and checkpoint are present.

### Label and test protocol

The `labels` array contains exact ESA interval annotations. The
`exclusion_mask` is the expanded training/evaluation exclusion mask.
`evaluation_ignore_mask` identifies annotations crossing train/validation/test
boundaries and is used only for offline window exclusion.

Ground-truth labels must never be passed into operational masking, drift
detection or candidate-update decisions. The final test split is for offline
evaluation only.

Point/window F0.5 and event-wise F0.5 are separate metrics. A two-window event
tolerance currently appears in the existing implementation and is configured as
provisional in `config/config.yaml`. It must not be reported as approved until
the team confirms it.

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

## P2 Kaggle experiment handover

Supply the existing Mission 1 processed split and B1 checkpoint as local Kaggle
inputs. Do not upload private ESA credentials or commit the data/checkpoint:

```text
/kaggle/input/esa-mission1/esa_adb/mission1/splits.npz
/kaggle/input/esa-mission1/models/mission1/lstm_b1.pt
```

The experiment expects repository-relative configuration paths. In Kaggle, copy
the input assets into the repository paths or set equivalent configuration
paths before execution:

```bash
git clone https://github.com/riyasol/Satellite-Telemetry-Anomaly-Detection.git
cd Satellite-Telemetry-Anomaly-Detection
git checkout feature/p2-drift-streaming
cd Person1_Handover
mkdir -p data/processed/esa_adb/mission1 results/models/mission1
cp /kaggle/input/esa-mission1/esa_adb/mission1/splits.npz data/processed/esa_adb/mission1/splits.npz
cp /kaggle/input/esa-mission1/models/mission1/lstm_b1.pt results/models/mission1/lstm_b1.pt
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test*.py"
python -m src.experiments.run_full_experiments
```

P2 writes the comparison CSV, detailed decision log, summary JSON, and
comparison plots under `P2_results/`.
The full benchmark does not retrain B1. B1 must be supplied; candidate model
training may use CPU and is controlled by P3 interfaces.
The default anomaly-mask dominance threshold is `0.75`, with at least 20 clean
reference and recent residuals required. Treat these as provisional: inspect the
validation/adaptation diagnostics and configure them explicitly before making
claims about detector superiority.
