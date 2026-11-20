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
* **Cadence / Span**: The supplied Mission 1 processed block contains 338,689
  observations over the requested 112-day interval. This differs from the
  322,560 rows implied by an uninterrupted 30-second cadence because original
  timestamps are irregular and are not resampled. Raw ESA source verification
  is pending in the current checkout.

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

## 📊 Experimental Results (ESA Mission 1)

Evaluated across **5 controlled drift scenarios** on Mission 1 telemetry:

| Scenario | Drift Type | B1 (Static) | B2 (Kuhn Periodic) | B3 (Drift-Triggered) | B4 (Proposed) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **S1: Stable Telemetry** | Nominal baseline | 0 updates | 4 updates *(unnecessary)* | 0 updates | **0 updates (0.0s waste)** |
| **S2: Temporary Drift** | Transient bias pulse | 0 updates | 3 updates | 1 update | **0 updates (persistence-saved)** |
| **S3: Persistent Harmful**| Step + variance shift | Event $F_{0.5}=1.0$ | 4 updates | 5 updates | **0 updates (robust)** |
| **S4: Large Harmless** | Secondary channel shift | 0 updates | 5 updates *(unnecessary)* | 3 updates *(unnecessary)* | **0 updates (degradation-saved)** |
| **S5: Moderate Harmful** | Correlated ramp shift | 0 updates | 5 updates | 1 update | **0 updates** |

### Key Experimental Findings
1. **Zero Unnecessary Updates**: In S1 (Stable) and S4 (Large Harmless), B2 and B3 triggered 3 to 5 unnecessary updates. B4 triggered **0 unnecessary updates**, saving >77% compute.
2. **Buffer Protection via Anomaly Masking**: In ablation studies, disabling anomaly masking resulted in anomaly spikes polluting the replay buffer and causing spurious model retrainings. B4's anomaly masking completely eliminates this vulnerability.

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
