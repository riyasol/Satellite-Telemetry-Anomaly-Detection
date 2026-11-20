# Drift-Aware Adaptive Anomaly Detection for Non-Stationary Spacecraft Telemetry

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end framework for **drift-aware adaptive anomaly detection** in multivariate satellite housekeeping telemetry under non-stationary conditions. This repository extends the validation-gated continual learning baseline (Kuhn et al., *Acta Astronautica*, 2026) by introducing an intelligent multi-criteria **Decision Layer** (Drift Detection + Distributional Severity + Persistence Tracking + Detector Degradation) to prevent catastrophic forgetting and eliminate unnecessary model updates on benign fluctuations.

---

## 📌 Project Overview

* **Project**: Enhanced Validation-Gated Continual Learning for Satellite Telemetry Anomaly Detection
* **Dataset**: European Space Agency Telemetry Anomaly Dataset (ESA-ADB) — **Mission 1**
* **Target Channels**: Housekeeping telemetry channels `channel_41` through `channel_46`
* **Cadence / Span**: Nominal 30-second sampling, 112-day continuous observation block (338,689 time steps)

---

## 🎯 Objectives & Research Gaps Addressed

1. **Drift Detection $\neq$ Adaptation Decision**: Standard streaming drift detectors (ADWIN, Page-Hinkley) trigger alerts whenever data shifts, but cannot determine *whether* model adaptation is needed or *how strongly* the detector should be updated.
2. **Coupling Drift Magnitude to Update Strength**: Statistical divergence measures ($KS$, Wasserstein $W_1$, and Jensen–Shannon $JS$) are normalized and synthesized into a composite severity score $S_t \in [0, 1]$.
3. **Preventing Over-Updating on Transient / Harmless Shifts**:
   * **Persistence Tracking**: Short-lived telemetry fluctuations are prevented from triggering model updates.
   * **Degradation Gating**: Updates are only escalated if the anomaly detector's actual performance deteriorates.
4. **Safety Gating & Anomaly Masking**:
   * **Anomaly Masking**: Prevents genuine fault spikes from entering drift calculations or polluting replay update buffers.
   * **Validation Gate**: Candidate models are tested against the current accepted model on a separate validation set before deployment.

---

## 🏛️ System Architecture

```
Telemetry Stream (Channels 41–46)
        │
        ▼
┌───────────────────────────────┐
│ LSTM 2-Layer Forecaster (B1)  │  Input: 250 steps, Horizon: 10 steps
└──────────────┬────────────────┘
               │  Forecast Errors / Residuals: z_t = ||y_t - ŷ_t||²
               ▼
┌───────────────────────────────┐
│    Anomaly Masking Module     │  Filters fault spikes & pre/post margins
└──────────────┬────────────────┘
               │  Clean, Eligible Telemetry
               ▼
┌───────────────────────────────────────────────────────────────┐
│                     DECISION LAYER (Ours)                    │
│                                                               │
│  1. ADWIN / Page-Hinkley Streaming Drift Detector             │
│  2. Distributional Shift Metrics: KS, Wasserstein, JS         │
│  3. Severity Calculator: S_t = 0.4 KS + 0.3 W + 0.3 JS        │
│  4. Persistence Tracker: P_t = Σ D_i >= P_min                 │
│  5. Degradation Tracker: ΔQ = Q_ref - Q_t                     │
└──────────────┬────────────────────────────────────────────────┘
               │
               ▼  3-Way Decision
    ┌──────────────────────────┬──────────────────────────┐
    ▼                          ▼                          ▼
NO_UPDATE                  FINE_TUNE                STRONG_UPDATE
(Keep current model)     (3 epochs, small lr)     (6-8 epochs, full replay)
                               │                          │
                               └────────────┬─────────────┘
                                            ▼
                             ┌──────────────────────────────┐
                             │  Candidate Trainer (SER)     │
                             │  (Recent Data + Replay)      │
                             └──────────────┬───────────────┘
                                            ▼
                             ┌──────────────────────────────┐
                             │      Validation Gate         │
                             │  MSE_cand <= MSE_curr * 1.02 │
                             └──────────────┬───────────────┘
                                     ┌──────┴──────┐
                                     ▼             ▼
                                [Accepted]    [Rejected]
                               Promote Model   Roll Back
```

---

## 🔬 Four Compared Baselines

* **B1 — Static LSTM Baseline**: Two-layer LSTM forecaster trained on initial 16 weeks of normal telemetry; no online adaptation.
* **B2 — Kuhn Continual Learning Baseline**: Validation-gated continual learning with Strong Experience Replay (SER), triggered periodically without drift detection or severity logic.
* **B3 — Drift-Triggered Adaptation**: ADWIN detection immediately requests candidate update + validation gate without severity, persistence, or degradation logic.
* **B4 — Proposed Drift-Aware System**: Integrated Decision Layer (Anomaly Masking + Severity + Persistence + Degradation) governing SER candidate generation and validation gating.

---

## 📊 Experimental Results

The checked-in result files were generated by an earlier experimental version
and must not be used as evidence for the current P3 protocol. Regenerate results
after preparing the local ESA data and baseline checkpoint:

```powershell
python -m src.experiments.run_full_experiments
```

The current runner uses the first part of the P1 validation split for threshold
calibration and candidate validation, a disjoint clean validation segment for
controlled drift and model updates, and the untouched P1 test split for final
offline scoring. Test labels are used only after predictions are produced.

The regenerated scenario CSV records detection metrics, fine-tune and strong
update counts, rejected candidates, update delays, runtime breakdowns, seed,
and the full parameter snapshot for each run. Ablation results include injected
anomaly pulses near persistent drift and compare the complete system with
anomaly masking disabled. Do not claim that the proposed method is better than
the base paper until these results have been regenerated and compared on the
same streams.

---

## 📁 Repository Structure

```
Satellite-Telemetry-Anomaly-Detection/
├── config/
│   └── config.yaml              # Central experiment, model, and drift parameters
├── docs/
│   ├── final 6.pptx             # Comprehensive project research presentation
│   └── Task_Guide.pdf           # Detailed Person 1/2/3 technical specification guide
├── results/
│   ├── mission1/
│   │   ├── b1_b4_scenario_comparison.csv    # Quantitative results across 5 scenarios
│   │   ├── ablation_study_results.csv        # Ablation study metrics
│   │   ├── b1_metrics.json                   # Person 1 baseline evaluation metrics
│   │   └── b1_scores.csv                     # Per-window scores and EMA thresholds
│   ├── models/
│   │   └── mission1/
│   │       ├── scaler.json                   # Fitted normalizer parameters
│   │       └── training_history.csv          # Training loss progression
│   └── plots/
│       └── mission1/
│           ├── b1_b4_f05_comparison.png      # Event F0.5 bar chart
│           ├── unnecessary_updates_comparison.png # Update overhead analysis
│           ├── drift_severity_decision_timeline.png # 4-panel diagnostic telemetry trace
│           ├── ablation_study_f05.png         # Ablation component impact
│           └── masking_impact_comparison.png  # Masking vs unmasked comparison
├── src/
│   ├── continual/               # Person 3 continual learning & validation gate
│   │   ├── candidate_trainer.py
│   │   ├── replay_buffer.py
│   │   └── validation_gate.py
│   ├── data/                    # Dataset loaders, labels, and preprocessors
│   │   ├── esa_labels.py
│   │   ├── esa_loader.py
│   │   └── prepare_esa.py
│   ├── drift/                    # Person 2 drift detectors & decision engine
│   │   ├── anomaly_masker.py
│   │   ├── decision_engine.py
│   │   ├── degradation.py
│   │   ├── distribution_measures.py
│   │   ├── persistence.py
│   │   ├── severity.py
│   │   ├── streaming_detectors.py
│   │   └── synthetic_drift_tests.py
│   ├── evaluation/              # Event-wise metrics, benchmarks, and ablations
│   │   ├── ablation.py
│   │   ├── benchmark.py
│   │   └── event_metrics.py
│   ├── experiments/             # Experiment runners & plot generators
│   │   ├── plot_drift_timeline.py
│   │   └── run_full_experiments.py
│   ├── models/                  # LSTM architecture & scoring
│   │   ├── anomaly_scoring.py
│   │   ├── evaluate_b1.py
│   │   ├── lstm_forecaster.py
│   │   ├── train.py
│   │   └── window_dataset.py
│   └── utils/
│       └── config.py
├── run_project.py               # Master end-to-end execution script
├── requirements.txt             # Minimal verified dependencies
└── .gitignore                   # Excludes raw 4GB telemetry and model checkpoints
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment

```bash
# Clone the repository
git clone https://github.com/riyasol/Satellite-Telemetry-Anomaly-Detection.git
cd Satellite-Telemetry-Anomaly-Detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# or .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the official ESA-ADB Mission 1 archive (`ESA-Mission1.zip`) and place it under:
`data/raw/ESA-Mission1/`
Run preprocessing:
```bash
python -m src.data.prepare_esa
```

### 3. Execution

* **Run all tests & complete benchmark end-to-end**:
  ```bash
  python run_project.py
  ```
* **Run Person 2 unit & synthetic drift tests**:
  ```bash
  python -m src.drift.synthetic_drift_tests
  ```
* **Run B1 baseline evaluation**:
  ```bash
  python -m src.models.evaluate_b1
  ```
* **Run B1–B4 5-scenario benchmark & ablations**:
  ```bash
  python -m src.experiments.run_full_experiments
  ```
* **Generate diagnostic timeline plot**:
  ```bash
  python -m src.experiments.plot_drift_timeline
  ```

---

## 📚 References

1. M. Kuhn, B. Sánchez Gómez, N. Moreno Blasco, P. Caserman, and F. Antonello, *"Validation-gated continual learning for anomaly detection in satellite telemetry,"* **Acta Astronautica**, vol. 214, pp. 102–118, 2026.
2. M. Stevanoski, M. Gama, and P. Bizarro, *"Change detection and adaptation in multi-target regression on data streams,"* **Machine Learning**, vol. 113, no. 3, pp. 1245–1270, 2024.
3. N. Shakhovska and O. Pukach, *"Data stream drift severity estimation for machine learning model adaptation,"* **Journal of Big Data**, vol. 12, no. 1, 2025.
