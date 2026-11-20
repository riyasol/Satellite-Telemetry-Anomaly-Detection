# Satellite Telemetry Anomaly Detection — Integrated Mission 1 Pipeline

The integrated P1/P2/P3 project prepares ESA-ADB Mission 1 data, trains or
loads the static LSTM baseline (B1), detects streaming drift, and evaluates
safe continual-learning candidates against B1-B4 on controlled scenarios.

## Selected channels

- Mission 1: `channel_41` through `channel_46` (all reported `Target=YES`).

Sampling interval is 30 s nominal (observation index based).

## Before running

1. Keep the official extracted folders under `data/raw/ESA-Mission1` if
   preparing data or training B1 from scratch.
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
- the existing `lstm_b1.pt` remains available to `evaluate_b1`

New training runs write `lstm_b1.pt`, `scaler.json`, and
`training_history.csv` under
`results/models/<mission-lowercase>/runs/<UTC-run-id>/`. Existing checkpoints
are not replaced by default. Replacing the legacy checkpoint requires the
explicit `python -m src.models.train --overwrite-checkpoint` option.

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
eligibility and data-integrity validation. P2 owns streaming drift detection
and update decisions. P3 owns replay, candidate training, validation gating,
and the held-out benchmark protocol.

### Required local ESA assets

Place the trusted ESA-ADB archive locally under:

```text
data/raw/ESA-Mission1/
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

## Integrated P1/P2/P3 experiment run

From `Person1_Handover`, install dependencies and provide the Mission 1
processed split and B1 checkpoint. Do not commit the data or checkpoint:

```text
data/processed/esa_adb/mission1/splits.npz
results/models/mission1/lstm_b1.pt
```

Then run:

```powershell
python -m pip install -r requirements.txt
python run_project.py
```

The runner executes the combined P1/P2/P3 tests, B1 evaluation, B1-B4
comparisons over five controlled scenarios, component ablations, and the
diagnostic timeline. Main deliverables are:

- `results/mission1/b1_metrics.json` and `b1_scores.csv`
- `results/mission1/b1_b4_scenario_comparison.csv`
- `results/mission1/ablation_study_results.csv` and `masking_ablation_results.csv`
- `results/plots/mission1/` comparison, ablation, masking, baseline, and timeline plots

For Kaggle, copy the supplied `splits.npz` and `lstm_b1.pt` into the paths above
before running. The benchmark does not retrain B1; it trains only candidates
that pass the P3 validation gate. Adaptation uses validation data only, and the
test split is reserved for final evaluation.

The event-tolerance setting and anomaly-mask thresholds are provisional. Treat
the generated metrics as measured results, not proof of superiority over a
paper: make that claim only after the scenarios demonstrate valid, reproducible
improvement against the cited paper's protocol and baselines.
