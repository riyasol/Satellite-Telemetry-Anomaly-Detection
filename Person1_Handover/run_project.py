"""Master End-to-End Project Runner for ESA Mission 1 Spacecraft Telemetry.

Executes:
1. Syntax and module compilation check
2. Person 2 unit & synthetic drift tests
3. P3 replay, validation-gate, and experiment tests
4. Baseline B1 evaluation
5. Full B1-B4 benchmark across all 5 controlled drift scenarios
6. Ablation study & masking comparison
7. Multi-panel diagnostic timeline plot generation
8. Result summary table output
"""
import sys
import subprocess
import time
from pathlib import Path
import pandas as pd


def run_step(step_num: int, title: str, module_cmd: list):
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {title.upper()}")
    print("=" * 70)
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable] + module_cmd, check=True)
    elapsed = time.perf_counter() - t0
    print(f"---> Step {step_num} completed successfully in {elapsed:.1f}s")
    return elapsed


def main():
    root = Path(__file__).resolve().parent
    print("=" * 70)
    print("STARTING END-TO-END PROJECT RUN: ESA MISSION 1 ADAPTIVE ANOMALY DETECTION")
    print(f"Working Directory: {root}")
    print("=" * 70)

    total_start = time.perf_counter()

    # Step 1: Verify syntax
    run_step(1, "Verify Source Syntax", ["-m", "compileall", "-q", "src"])

    # Step 2: Run Person 2 unit & synthetic tests
    run_step(2, "Person 2 Drift & Decision Unit Tests", ["-m", "src.drift.synthetic_drift_tests"])

    # Step 3: Run P1 data-contract, P2 drift, and P3 continual-learning tests
    run_step(3, "Integrated P1-P3 Test Suite", [
        "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"
    ])

    # Step 4: Run Baseline B1 Evaluation
    run_step(4, "Baseline B1 Static Forecaster Evaluation", ["-m", "src.models.evaluate_b1"])

    # Step 5: Run B1-B4 across 5 scenarios and P3 component ablations
    run_step(5, "Full B1-B4 Benchmark & Ablation Study", [
        "-m", "src.experiments.run_full_experiments"
    ])

    # Step 6: Generate the controlled-stream diagnostic timeline
    run_step(6, "Generate Diagnostic Timeline Plot", [
        "-m", "src.experiments.plot_drift_timeline"
    ])

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print(f"ALL STEPS COMPLETED SUCCESSFULLY IN {total_time:.1f}s!")
    print("=" * 70)

    # Display final results summary
    comp_csv = root / "results" / "mission1" / "b1_b4_scenario_comparison.csv"
    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
        print("\n--- B1-B4 COMPARATIVE BENCHMARK ACROSS 5 SCENARIOS ---")
        summary_cols = [
            "scenario_name", "baseline", "f0.5_event", "precision", "recall",
            "fine_tune_updates", "strong_updates", "candidates_rejected",
            "candidate_retention_rejections",
            "candidate_adaptation_validation_rejections",
            "first_adaptation_delay_windows", "unnecessary_updates",
            "runtime_seconds", "runtime_training_seconds",
            "runtime_validation_seconds",
        ]
        print(df_comp[summary_cols].to_string(index=False))

    abl_csv = root / "results" / "mission1" / "ablation_study_results.csv"
    if abl_csv.exists():
        df_abl = pd.read_csv(abl_csv)
        print("\n--- ABLATION STUDY ON SCENARIO 3 (PERSISTENT HARMFUL DRIFT) ---")
        print(df_abl.to_string(index=False))

    plots_dir = root / "results" / "plots" / "mission1"
    if plots_dir.exists():
        print("\n--- GENERATED PUBLICATION-QUALITY PLOTS ---")
        for p in plots_dir.glob("*.png"):
            print(f"  * {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

