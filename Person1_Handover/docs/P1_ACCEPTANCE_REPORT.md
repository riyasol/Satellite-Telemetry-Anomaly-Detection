# P1 Technical Acceptance Report

**Status: NOT ACCEPTED / NOT HANDOFF READY.** The active checkout has no raw
ESA source files, so source-to-processed integrity and baseline reproduction
cannot be certified. Event metrics and margin choices also await guide/team
approval. No raw or processed data was changed, and no model was trained.

Audit date: 2026-11-20  
Active repository: `feature/p1-data-contract` at
`f8732e3ce3b6d2f87a8c1dcb4e234836fd2af3ac`

## 1. Issues resolved locally

- `src/models/train.py` now writes new runs under unique
  `results/models/<mission>/runs/<UTC-run-id>/` directories. The existing
  `lstm_b1.pt` is retained for evaluation; replacing it requires the explicit
  `--overwrite-checkpoint` option. A run writes its best checkpoint once, and
  explicit replacement uses a temporary file followed by atomic replacement.
- Added an independent ESA B1 event-evaluation pipeline. Its outputs are
  approval- and source-verification-gated; it does not read controlled or
  synthetic scenario result directories.
- Corrected the P1 README's Mission 1 count and clarified the Mission 1
  observation-count discrepancy in the repository README. The `id_145`
  boundary statement is now flagged as conflicting with the supplied NPZ
  metadata rather than presented as verified.
- Added a processed-artifact audit, a raw-source integrity report, and
  per-mission pending event-evaluation JSON files.
- Added tests for checkpoint non-overwrite behavior, explicit replacement,
  run-directory uniqueness, event matching, tolerance boundaries, and
  approval gating.

## 2. Issues still pending

- Raw ESA Mission 1 and Mission 2 archives are absent from all requested
  locations. Raw channel order, row counts, timestamp alignment, duplicate,
  missing, NaN, Inf, invalid, and source sampling-gap counts are therefore
  **unverified**.
- The processed arrays are in an unversioned external handover bundle. Its
  source fingerprint says that its generation code does not match the active
  branch. Copy equality is not raw-source provenance.
- Mission 1 B1 score/threshold files conflict across the active repository and
  packaged copies. Mission 2 B1 score/metric files are absent from the active
  repository.
- Event-wise metrics are pending raw-source verification and approval of the
  event protocol.
- Preprocessing and streaming-mask margins remain provisional and inconsistent.
- Mission 2 channels 25-28 remain at near-zero scale in the supplied processed
  arrays under the current `std < 1e-8 -> 1.0` rule. No correction was made.
- No baseline evaluation was run and no model was retrained. Two exploratory
  plots were generated from the unverified processed NPZ and labeled as such;
  no existing plot was overwritten.

## 3. Raw-source verification

The following raw roots were checked and were absent:

- `C:\IT\RM\Satellite-Telemetry-Anomaly-Detection\Person1_Handover\data\raw`
- `C:\IT\RM\Person2_Drift_Decision_Completion_Pack_v3\Person1_Handover\data\raw`
- `D:\IT\RM\spacecraft_telemetry\data\raw`

For **each** absent root, the following expected files are missing:

- Mission 1 metadata: `ESA-Mission1\channels.csv` and `labels.csv`.
- Mission 1 channel ZIPs: `ESA-Mission1\channels\channel_41.zip` through
  `channel_46.zip`.
- Mission 2 metadata: `ESA-Mission2\channels.csv` and `labels.csv`.
- Mission 2 channel ZIPs: `ESA-Mission2\channels\channel_18.zip` through
  `channel_28.zip`.

The top-level ZIPs in `C:\IT\RM` were inspected without extraction; no expected
raw-channel or `ESA-Mission1/ESA-Mission2` source members were found. Their
archive SHA-256 values and the exact missing file paths are recorded in
[`source_integrity_report.json`](../results/data_integration/source_integrity_report.json).
Consequently, no raw row counts, raw statistics, raw timestamp alignment,
source-to-processed comparison, or raw training-only scaler recomputation is
claimed.

## 4. Checkpoint preservation

Before changing training code, byte-for-byte backups were created under
`C:\IT\RM\P1_Kaggle_Handoff\checkpoint_backups\before_checkpoint_safety_change`.
Source and backup SHA-256 hashes were checked again after the code change:

| Mission | Original SHA-256 | Backup SHA-256 | Result |
|---|---|---|---|
| Mission 1 | `19a7d4c4e40a2f5b8251d71d2496730c8fd331aa13e5dd33b3dfdb4532be09af` | same | MATCH |
| Mission 2 | `a73de4507d85f5ca323c182d93f1b87d9e8e619cdbfb6d32a4b184f1f567439c` | same | MATCH |

Both hashes also match the corresponding checkpoint members in
`Person1_Handover.zip` and `Person2_Drift_Decision_Completion_Pack_v3.zip`.
This verifies byte identity among those local copies; it does not prove which
raw observations or code revision produced the weights. The active repository
does not contain either mission checkpoint. No checkpoint was overwritten or
retrained during this audit.

## 5. Event-wise evaluation

The evaluator is
[`esa_event_evaluation.py`](../src/evaluation/esa_event_evaluation.py). The
proposed one-to-one maximum-cardinality matching and current two-scored-window
tolerance are explicitly provisional. The reports contain `metrics: null`:

- [`mission1_b1_event_metrics.json`](../results/event_evaluation/mission1_b1_event_metrics.json):
  pending guide/team approval and raw-source verification.
- [`mission2_b1_event_metrics.json`](../results/event_evaluation/mission2_b1_event_metrics.json):
  same blockers, plus missing active-repository B1 scores and metrics sidecar.

No synthetic event result was used as an ESA baseline result. The existing
general `event_metrics.py` helper is not used by this pipeline because it can
credit one predicted event to multiple true events.

## 6. Margin configuration and comparison

Current configuration has **0/0 seconds** in `data` preprocessing and
**60/60 seconds** in `drift`. `prepare_esa.py` passes the data values to
`esa_labels.py`; the drift values are not wired to that preprocessing path.
`AnomalyMasker` separately defaults to 60/60 seconds and converts seconds to
observation steps using its nominal-step argument. The P2 packaged config also
uses `pre_windows: 3` and `post_windows: 3`, a different unit. These settings
were not unified because approval is missing and P2/P3 implementation was not
changed.

The following counts are measured only on the supplied processed NPZ files.
Each candidate expands exact labels by equal pre/post margins using stored
timestamps. Candidates were **not** applied to saved data. Eligible windows
are listed as train / validation / test, with input 250, horizon 10, stride 10.

| Mission | Margin each side | Masked observations | Eligible windows Train / Val / Test |
|---|---:|---:|---:|
| M1 | 0 s (stored) | 924 | 23,547 / 5,023 / 4,929 |
| M1 | 30 s | 945 | 23,547 / 5,023 / 4,929 |
| M1 | 60 s | 966 | 23,546 / 5,023 / 4,927 |
| M1 | 120 s | 1,010 | 23,546 / 5,023 / 4,926 |
| M2 | 0 s (stored) | 9,567 | 44,166 / 9,190 / 8,989 |
| M2 | 30 s | 9,670 | 44,161 / 9,187 / 8,983 |
| M2 | 60 s | 9,878 | 44,153 / 9,185 / 8,980 |
| M2 | 120 s | 10,198 | 44,138 / 9,175 / 8,974 |

The exact per-split mask counts and the 0/30/60/120-second comparison are in
[`processed_artifact_audit.json`](../results/data_integration/processed_artifact_audit.json).
These are not raw-source counts and do not imply approval of any candidate.

## 7. Splits, leakage, and processed-array statistics

Independent structural checks on the supplied processed arrays found
contiguous, non-overlapping 70/15/15 bounds:

- Mission 1: 338,689 rows; train `[0, 237082)`, validation `[237082, 287885)`,
  test `[287885, 338689)`.
- Mission 2: 645,191 rows; train `[0, 451633)`, validation `[451633, 548411)`,
  test `[548411, 645191)`.

Both arrays have the expected channel order, zero non-finite values, strictly
increasing timestamps, zero exact duplicate timestamps, and zero
`evaluation_ignore_mask` observations. Stored labels equal the stored
exclusion mask because the packaged metadata says the margins were zero.
These are checks of the packaged artifacts, not confirmation against raw ESA
labels. The README's `id_145` crossing note conflicts with the packaged
Mission 1 metadata's empty crossing-ID list; the discrepancy remains
unresolved without raw labels.

In the supplied processed arrays, the unmasked training means/stds for
Mission 1 and Mission 2 channels 18-24 are approximately 0/1. Mission 2
channels 25-28 have metadata scaler std `1.0`, but their measured unmasked
normalized-array stds are approximately `2.02e-12`, `4.79e-13`, `1.20e-12`,
and `6.51e-13`. Raw normalization statistics cannot be independently
recomputed. The split artifact contains only train, validation, and test; no
adaptation split was invented. The current protocol does not require another
split, but the guide/team must confirm whether it is required for the agreed
methodology.

The timestamp gap report is likewise processed-artifact-only: M1 median/min/max
gaps are 30.0 / 6.627 / 30.003 seconds; M2 are 18.0 / 0.018 / 2268.003
seconds. The reported missing-slot estimates (0 and 125) are explicitly
heuristics based on median cadence, not verified raw missing-record counts.

Two exploratory plots were generated from the supplied processed arrays and
timestamps. Both are labeled in-image as unverified artifacts:

- [`processed_split_channel_statistics.png`](../results/data_integration/plots/processed_split_channel_statistics.png)
- [`processed_timestamp_gap_distributions.png`](../results/data_integration/plots/processed_timestamp_gap_distributions.png)

They are not raw-source plots or baseline evaluation plots.

## 8. Artifact consistency, baseline reproduction, and environment

The active repository branch is authoritative for **current code**, but no
single Mission 1 threshold/score artifact is authoritative:

- The active repository's Mission 1 `b1_scores.csv` is 512,877 bytes with SHA
  `50fb4739f7b68e343004b385bf82c39f23798a34c163bc628fd2c9eea7a1b37a` and
  stores threshold `0.32750271869164854`.
- The extracted Person2 copy's 522,944-byte Mission 1 score file has SHA
  `027ffcba558b2ec668e8e0ba0dd5f948bf13ecd3fc985d0b49e1d826a4a93671` and
  threshold `0.3275027203202959`. It matches the score file in
  `Person1_Handover.zip`. The Person2 packaged ZIP contains another 522,938-byte
  score file with SHA
  `55943a1e7c3689dac514b25dca93f8bc05a1fce1df38d833d1cd567ab93ebb33`.
- Mission 1 point-window F0.5 fields happen to match across these copies, but
  that does not establish common provenance or reproduction. The threshold
  values and score file hashes conflict; do not promote one copy's result into
  the primary repository.
- Active `results/models/scaler.json` contains generic `channel_1` through
  `channel_6` values and is not the Mission 1 ESA scaler. The separate Mission
  1 scaler hash is
  `4c4534934496a388b109c66a8e91c3c545578bbf149545cb9ad8102830370e73` and
  matches the packaged P1 copy, but its source-data provenance remains
  unverified. The active repository has no Mission 2 scaler or checkpoint.
- Mission 1 and Mission 2 checkpoints in the external P1/P2 copies are byte
  identical to the original handover archive members (Section 4), but that
  byte identity does not make the accompanying score files or preprocessing
  authoritative. The P2 packaged README still gives nominal cadence estimates
  of 322,560 / 537,600 rows; its NPZ artifacts contain 338,689 / 645,191 rows.

The supplied P1 Kaggle manifest's file hashes and listed files were checked;
they match that bundle. Its `source_fingerprints.json` explicitly says the
active repository code does not match the data-generation code. This is
structural/package consistency only.

Baseline reproduction is **not verified**. The active checkout lacks
`data/processed/esa_adb` and both mission checkpoints. Its existing Mission 1
score and metrics files conflict with the packaged Mission 1 files, while
Mission 2 score/metrics files are absent. Existing values must not be reported
as reproduced ESA metrics until their generating code/data revision is
identified and validated.

Once raw-source provenance and hashes are established, the intended bounded
reproduction is:

```powershell
python -m src.data.validate_processed
python -m src.models.evaluate_b1
```

Training is not required for reproduction. It was not run here. The observed
local environment was Python 3.12.5, NumPy 1.26.4, pandas 2.2.3, PyYAML 6.0.2,
PyTorch 2.5.1+cu118, Matplotlib 3.9.2, SciPy 1.13.1; CUDA was unavailable.
`requirements.txt` uses minimum versions rather than a fully pinned lockfile.
The bounded command `python -m unittest discover -s tests -v` passed all 17
tests, the YAML config parsed, and `git diff --check` passed. No training or
full-data B1 evaluation was run.

## 9. Artifact manifest

[`artifact_manifest.json`](../results/data_integration/artifact_manifest.json)
enumerates the in-scope primary repository, processed handoff bundle,
checkpoint backups, packaged P2 handover files, and top-level source ZIP
archives, with file sizes and SHA-256 hashes. Raw ESA channel files are listed
as missing, not assigned fabricated hashes.

## 10. Files changed or generated

Modified:

- Repository `README.md` and `Person1_Handover/README.md`.
- `Person1_Handover/DATA_EVENT_PROTOCOL.md`.
- `Person1_Handover/config/config.yaml`.
- `Person1_Handover/src/models/train.py`.

Added:

- `Person1_Handover/src/evaluation/esa_event_evaluation.py`.
- `Person1_Handover/tests/test_checkpoint_safety.py`.
- `Person1_Handover/tests/test_esa_event_evaluation.py`.
- `Person1_Handover/docs/P1_ACCEPTANCE_REPORT.md`.
- `Person1_Handover/results/data_integration/processed_artifact_audit.json`.
- `Person1_Handover/results/data_integration/source_integrity_report.json`.
- `Person1_Handover/results/data_integration/artifact_manifest.json`.
- `Person1_Handover/results/data_integration/plots/processed_split_channel_statistics.png`.
- `Person1_Handover/results/data_integration/plots/processed_timestamp_gap_distributions.png`.
- `Person1_Handover/results/event_evaluation/mission1_b1_event_metrics.json`.
- `Person1_Handover/results/event_evaluation/mission2_b1_event_metrics.json`.

Outside the repository, two byte-verified checkpoint backups were created in
`C:\IT\RM\P1_Kaggle_Handoff\checkpoint_backups\before_checkpoint_safety_change`.
No raw data, processed array, scaler, checkpoint source, or P2/P3 implementation
was modified.

## 11. Remaining manual approvals and inputs

1. Supply the authoritative Mission 1 and Mission 2 raw source folders listed
   in Section 3, plus provenance/licensing information.
2. Confirm preprocessing margins and streaming-mask margins, including units
   and mission-specific cadence treatment.
3. Approve the event tolerance, one-to-one versus alternative matching rule,
   event adjacency convention, and undefined-metric policy.
4. Confirm the `id_145` split-boundary discrepancy and whether unlabelled rows
   may be treated as normal.
5. Confirm whether the method requires a separate adaptation split.
6. Identify the exact generation code/config revision for the supplied
   processed arrays and Mission 1 score files before selecting a canonical
   threshold/results artifact.

## 12. P2/P3 handoff readiness

**Not ready for final P2/P3 data handoff.** The code contract and safe training
workflow are documented, and processed-artifact structure is inspectable, but
critical raw-source integrity, score provenance, Mission 2 active result
availability, and event protocol approval remain unresolved. Do not call P1
complete or treat the provided processed bundle as authoritative until those
items are closed.
