# P1 Data and Evaluation Protocol

## Dataset scope

P1 supports ESA-ADB Mission 1 and Mission 2 from the official channel and
annotation layout.

The active repository does not contain raw ESA archives or processed
`splits.npz` data. No large dataset or checkpoint should be copied into Git.

## Channel lists

- Mission 1: `channel_41` through `channel_46`.
- Mission 2: `channel_18` through `channel_28`.

Channels are aligned only by exact UTC timestamp identity. P1 must not resample
channels or align by row position. P1 retains original timestamps and represents
LSTM steps as observation indices.

## Mission 2 low-scale finding

A low-scale finding for Mission 2 was documented in the reference handover:
channels 25–28 have source values near `1e-11` to `7e-11`, and the existing
preprocessing reference replaces standard deviations below `1e-8` with `1.0`.
That finding is evidence in the reference archive, not a result reproduced from
the active repository because active Mission 2 raw data and processed data are
not present.

P1 must verify this behavior from the authoritative local Mission 2 data before
changing preprocessing. The `std < 1e-8 -> 1.0` rule must remain unchanged until
the original data owner approves a correction, because changing it could
invalidate supplied B1 checkpoints.

## Annotation arrays

P1 maintains four distinct concepts:

| Array | Meaning | Permitted use |
|---|---|---|
| `labels` | Exact interval ground truth | Offline evaluation only |
| `exclusion_mask` | Exact annotations plus configured margins | Training and offline safety filtering |
| `evaluation_ignore_mask` | Annotation intervals crossing split boundaries | Offline window exclusion |
| `predicted_anomaly` | Model or threshold output | Operational masking |

P1 must never put `labels` or `evaluation_ignore_mask` into P2/P3 online
training decisions. P1 may provide exact test labels to P3 only after a
prediction has been generated and only for offline metric computation.

## Splits

The default chronology is:

```text
train: 70%
validation: 15%
test: 15%
```

The split boundaries are observation indices. P1 must validate that:

- `train_end + validation_length + test_length == row_count`
- train precedes validation precedes test.
- timestamps are strictly increasing.
- channel order is stable.
- normalization uses only training observations outside the expanded
  exclusion mask.
- exact annotations crossing split boundaries remain in metadata and exact
  labels but are excluded from window-level boundary metrics.

## Window convention

For an input window `I`, forecast horizon `H` and start `i`:

```text
input  = values[i:i+I]
target = values[i+I:i+I+H]
span   = [i, i+I+H)
```

Window eligibility requires no excluded observation in the complete span.
Checking only the target timestamps is not sufficient.

The same start, `I`, `H` and stride must be used for B1-B4. Window stride is
currently `10` in the active configuration and is a compute-saving setting,
not evidence that a stride of `1` is equivalent for every metric.

## Canonical P1/P2 fields

P1 should expose:

```text
forecast_start_index
input_start_index
input_end_index_exclusive
target_start_index
target_end_index_exclusive
forecast_start_utc
raw_mse
ema_score
threshold
predicted_anomaly
true_target_window_anomaly
valid
eligible_for_drift
mask_reason
```

P1 does not own P2's final streaming `DecisionPacket`. P2 should consume P1
fields and add its own:

```text
adwin_alarm
distribution_evaluated
ks_statistic
wasserstein_distance
js_divergence
severity
persistent_drift
degradation_delta_f05
action
reason
```

## Held-out test protocol

The validation split determines the anomaly threshold.

The test split:

- is scored with frozen B1 or the currently accepted P3 model;
- excludes target windows that cross retained boundary annotation intervals;
- uses exact `labels` only after predictions are generated;
- must never trigger candidate creation or candidate promotion.

P3 validation gate data must come from the validation split or a separately
labelled validation block. It must not use held-out test data.

## Point versus event metrics

P1 reports point/window metrics using each scored window as a binary
classification item.

P1 does not claim that point/window F0.5 is event-wise F0.5.

The existing event metric implementation uses a default tolerance of two
windows. This is retained as provisional in `config/config.yaml`. P2 and P3 must
not describe it as professor-approved until the team confirms the exact
tolerance and matching rule.

No tolerance should vary by scenario or mission without an explicit config
entry and test.

## Independent ESA B1 event evaluation

Use `python -m src.evaluation.esa_event_evaluation` for the static ESA B1
baseline. It reads only `results/<mission>/b1_scores.csv` and its matching
`b1_metrics.json`; it does not consume controlled-scenario or synthetic
experiment outputs. It writes separate mission JSON reports and refuses to
overwrite an existing report.

Final metrics are gated by both `evaluation.event_evaluation.approval_status`
and `source_verification_status`. Until the team approves the event protocol
and raw-source integrity is verified, each report records a pending status and
does not publish final TP/FP/FN or F0.5.

The proposed matching rule is maximum-cardinality one-to-one matching between
contiguous true and predicted scored-window events. A true event is expanded by
`event_tolerance_steps` scored windows on both sides; a predicted event matches
when its interval overlaps that expanded interval. This is a proposal, not an
approved convention. The report includes event intervals, matched pairs,
unmatched events, event counts, tolerance, precision, recall and F0.5 after the
approval gate is satisfied. Undefined precision/recall/F0.5 for empty
denominators remain `null` unless the approved config explicitly selects a
zero-value policy.

The existing general-purpose `event_metrics.py` helper is not the final ESA
protocol: it has a default two-window tolerance and can credit a prediction to
multiple true events. The independent evaluator avoids using it for the
approval-gated ESA baseline.

## Decisions requiring guide/team approval

1. Preprocessing margins before/after each annotation, in seconds, separately
   for each mission if cadence differences warrant it.
2. Streaming-mask margins and their units (seconds, observations, or scored
   windows); these are distinct from preprocessing margins.
3. Event tolerance in scored-window units, including whether the current value
   of two is retained.
4. Event matching rule (one-to-one maximum-cardinality matching versus the
   existing any-overlap/many-to-many behavior), and whether adjacent alarm
   windows form one event.
5. The policy for undefined precision/recall/F0.5 when there are no true or
   predicted events.
6. Whether candidate adaptation data must use a separate chronological
   adaptation split. The supplied arrays contain only train/validation/test;
   no separate adaptation split is present. The existing contract requires
   validation for candidate acceptance and prohibits test-driven updates, but
   does not itself require an additional split.
7. Whether unlabelled observations may be interpreted as normal, as current
   interval-label processing assumes.

Do not revise the saved observations, labels, masks, split bounds, scalers, or
checkpoints until any approved preprocessing-margin change is separately
reviewed and versioned.
