"""Unit tests and controlled synthetic-stream validation for Person 2 drift modules.

Tests:
1. Stable stream: checks false alarm rate.
2. Sudden shift: measures detection delay.
3. Gradual ramp shift: measures detection delay.
4. Anomaly masking: verifies anomaly spikes do not masquerade as distribution drift.
5. Three-way decision engine: tests NO_UPDATE, FINE_TUNE, STRONG_UPDATE transitions.
"""
from __future__ import annotations
import numpy as np
from src.drift.streaming_detectors import ADWIN, PageHinkley
from src.drift.distribution_measures import compute_all_distribution_measures
from src.drift.anomaly_masker import AnomalyMasker
from src.drift.severity import SeverityCalculator
from src.drift.persistence import PersistenceTracker
from src.drift.degradation import DegradationTracker
from src.drift.decision_engine import DecisionEngine, AdaptationAction


def run_tests():
    print("=" * 60)
    print("RUNNING PERSON 2 DRIFT & DECISION UNIT & SYNTHETIC TESTS")
    print("=" * 60)
    rng = np.random.default_rng(42)

    # 1. Stable synthetic stream
    n_stable = 1000
    stable_stream = rng.normal(loc=0.0, scale=0.1, size=n_stable)
    adwin = ADWIN(delta=0.01)
    ph = PageHinkley(delta=0.05, threshold=15.0)

    adwin_false_alarms = sum(adwin.update(val) for val in stable_stream)
    ph_false_alarms = sum(ph.update(val) for val in stable_stream)
    print(f"[TEST 1] Stable Stream ({n_stable} steps):")
    print(f"  ADWIN false alarms: {adwin_false_alarms} ({100*adwin_false_alarms/n_stable:.2f}%)")
    print(f"  Page-Hinkley false alarms: {ph_false_alarms} ({100*ph_false_alarms/n_stable:.2f}%)")
    assert adwin_false_alarms <= 2, "ADWIN false alarm rate too high on stable stream"

    # 2. Sudden shift stream
    n_pre = 400
    n_post = 400
    shift_magnitude = 1.0
    sudden_stream = np.concatenate([
        rng.normal(loc=0.0, scale=0.1, size=n_pre),
        rng.normal(loc=shift_magnitude, scale=0.1, size=n_post)
    ])
    adwin.reset()
    detected_step_adwin = None
    for step, val in enumerate(sudden_stream):
        if adwin.update(val) and step >= n_pre:
            detected_step_adwin = step
            break

    delay_adwin = (detected_step_adwin - n_pre) if detected_step_adwin else None
    print(f"\n[TEST 2] Sudden Shift Stream (shift at step {n_pre}, magnitude {shift_magnitude}):")
    print(f"  ADWIN detected at step: {detected_step_adwin} (Delay: {delay_adwin} steps)")
    assert delay_adwin is not None and delay_adwin < 100, "ADWIN failed to detect sudden shift in time"

    # 3. Gradual ramp shift stream
    ramp_length = 300
    ramp = np.linspace(0.0, 1.2, ramp_length)
    gradual_stream = np.concatenate([
        rng.normal(loc=0.0, scale=0.1, size=n_pre),
        ramp + rng.normal(loc=0.0, scale=0.1, size=ramp_length),
        rng.normal(loc=1.2, scale=0.1, size=200)
    ])
    adwin.reset()
    detected_step_gradual = None
    for step, val in enumerate(gradual_stream):
        if adwin.update(val) and step >= n_pre:
            detected_step_gradual = step
            break

    delay_gradual = (detected_step_gradual - n_pre) if detected_step_gradual else None
    print(f"\n[TEST 3] Gradual Ramp Shift (ramp starts at step {n_pre}):")
    print(f"  ADWIN detected at step: {detected_step_gradual} (Delay: {delay_gradual} steps)")
    assert delay_gradual is not None, "ADWIN failed to detect gradual shift"

    # 4. Anomaly masking test
    print(f"\n[TEST 4] Anomaly Masking & Exclusion Margin:")
    masker = AnomalyMasker(pre_margin_seconds=60, post_margin_seconds=60, nominal_step_seconds=30)
    flags = np.zeros(200, dtype=bool)
    flags[50:53] = True # 3-step anomaly spike
    exclusion = masker.build_exclusion_mask(flags)
    # pre_margin = 2 steps (48, 49), anomaly = (50, 51, 52), post_margin = 2 steps (53, 54)
    assert exclusion[48] and exclusion[54], "Pre/post margins not properly applied"
    assert not exclusion[45] and not exclusion[57], "Exclusion leaked outside margin"
    print("  Exclusion mask successfully spans safety margins before and after anomaly spike.")

    # Distribution measures with and without masking
    clean_ref = rng.normal(0, 0.1, 100)
    spike_window = clean_ref.copy()
    spike_window[50:53] += 5.0 # Anomaly spike
    unmasked_meas = compute_all_distribution_measures(clean_ref, spike_window)

    masked_sample = spike_window[~exclusion[:100]]
    masked_meas = compute_all_distribution_measures(clean_ref[:len(masked_sample)], masked_sample)
    print(f"  Unmasked spike KS: {unmasked_meas['ks']:.4f}, Wasserstein: {unmasked_meas['wasserstein']:.4f}")
    print(f"  Masked clean KS:   {masked_meas['ks']:.4f}, Wasserstein: {masked_meas['wasserstein']:.4f}")
    assert masked_meas['wasserstein'] < unmasked_meas['wasserstein'], "Masking did not suppress anomaly artifact"

    # 5. Three-way Decision Engine test
    print(f"\n[TEST 5] Three-Way Decision Logic:")
    engine = DecisionEngine(theta_low=0.35, theta_high=0.65, persistence_min=3, degradation_threshold=0.05)

    # Case A: Low severity -> NO_UPDATE
    d_a = engine.decide(
        step=1, timestamp_utc="2026-01-01T00:00:00Z", drift_detected=True,
        detector_name="ADWIN", raw_measures={"ks": 0.1}, normalized_measures={"ks": 0.2},
        severity=0.20, persistence_count=3, persistence_passed=True,
        degradation_value=0.01, degradation_passed=False
    )
    assert d_a.action == AdaptationAction.NO_UPDATE, "Low severity should result in NO_UPDATE"

    # Case B: High severity but not persistent -> NO_UPDATE
    d_b = engine.decide(
        step=2, timestamp_utc="2026-01-01T00:01:00Z", drift_detected=True,
        detector_name="ADWIN", raw_measures={"ks": 0.8}, normalized_measures={"ks": 0.8},
        severity=0.80, persistence_count=1, persistence_passed=False,
        degradation_value=0.10, degradation_passed=True
    )
    assert d_b.action == AdaptationAction.NO_UPDATE, "Non-persistent drift should result in NO_UPDATE"

    # Case C: Persistent moderate drift -> FINE_TUNE
    d_c = engine.decide(
        step=3, timestamp_utc="2026-01-01T00:02:00Z", drift_detected=True,
        detector_name="ADWIN", raw_measures={"ks": 0.4}, normalized_measures={"ks": 0.45},
        severity=0.45, persistence_count=3, persistence_passed=True,
        degradation_value=0.02, degradation_passed=False
    )
    assert d_c.action == AdaptationAction.FINE_TUNE, "Persistent moderate drift should FINE_TUNE"

    # Case D: Persistent severe drift + degradation -> STRONG_UPDATE
    d_d = engine.decide(
        step=4, timestamp_utc="2026-01-01T00:03:00Z", drift_detected=True,
        detector_name="ADWIN", raw_measures={"ks": 0.75}, normalized_measures={"ks": 0.75},
        severity=0.75, persistence_count=4, persistence_passed=True,
        degradation_value=0.08, degradation_passed=True
    )
    assert d_d.action == AdaptationAction.STRONG_UPDATE, "Severe drift + degradation should STRONG_UPDATE"

    print("  Decisions verified: NO_UPDATE, FINE_TUNE, STRONG_UPDATE all passed under correct conditions!")
    print("\nALL PERSON 2 UNIT TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_tests()

