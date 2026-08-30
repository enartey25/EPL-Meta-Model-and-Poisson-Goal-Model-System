# EPL Match Predictor — Stacked Ensemble & Calibrated Poisson System

> **Academic & Research Project** — All predictions are statistical estimates for educational purposes only. Not intended for sports betting or financial speculation.

---

## Overview

A multi-stage English Premier League match forecasting system that combines a **stacked tree ensemble classifier** with a **zone-calibrated Poisson goal model** to produce match outcome probabilities and exact scoreline distributions for any EPL fixture.

The pipeline was trained on 1,500+ historical EPL fixtures (2022/23–2024/25) using strict chronological partitioning to eliminate lookahead bias, and is evaluated using proper scoring rules (multi-class log loss, Brier score) rather than naive accuracy.

**Holdout test log loss: `1.0235`** vs. a naive null baseline of `1.0790`.

---

## Architecture

```
Match Data (football-data.co.uk)           xG Data (Understat)
         │                                        │
         └──────────────┬─────────────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   Leakage-Free State Engine  │
         │  • Sequential Elo ratings    │
         │  • Rolling xG (5-match)      │
         │  • Variance & form metrics   │
         └──────────────┬───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │      Base Learners           │
         │   XGBoost  +  Random Forest  │
         └──────────────┬───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   Meta-Learner (Level-1)     │
         │   Logistic Regression Stack  │
         │   Log Loss: 1.0235           │
         └──────────────┬───────────────┘
                        │
         ┌──────────────▼───────────────┐
         │  Zone-Based Poisson Engine   │
         │  • Leakage-free lambdas      │
         │  • Score matrix calibrated   │
         │    by meta-learner priors    │
         │  Log Loss: 1.0716 → 1.0235   │
         └──────────────────────────────┘
```

---

## Key Design Decisions

### Chronological Temporal Partitioning
Standard k-fold cross-validation introduces severe lookahead bias for time-series sports data. The pipeline uses strict chronological partitioning:

| Split | Seasons | Purpose |
|:------|:--------|:--------|
| Train | 2022/23 – 2023/24 | Model fitting |
| Validation | 2024/25 | Hyperparameter tuning |
| Holdout Test | 2025/26 | Final evaluation |

### SMOTENC Rejection
Synthetic oversampling (SMOTENC) was tested to address class imbalance in draw outcomes (~24%). It was **rejected** after analysis showed it corrupted the natural joint probability distribution between Elo differentials and match outcomes — increasing test log loss across both base learners. Match class imbalance is a macroeconomic property of the sport, not a sampling artifact.

### Proper Scoring Rules
All model evaluation uses **multi-class log loss** and **multi-class Brier score** rather than raw accuracy. These are strictly proper scoring rules that penalise miscalibrated probability estimates — critical for a forecasting system where decision quality depends on the confidence, not just the direction, of a prediction.

### Zone-Based Poisson Calibration
Raw Poisson models assume independence between home and away goals and achieve `1.0716` log loss. The meta-learner's outcome probabilities are used to rescale the three zones of the 11×11 score probability matrix (Home Win triangle, Draw diagonal, Away Win triangle), reducing Poisson log loss from `1.0716` to `1.0235` while preserving full scoreline interpretability.

### Promoted Team Priors
Newly promoted clubs (e.g., Coventry City, Ipswich Town, Hull City) have no EPL match history in the state dictionary. Rather than defaulting to the league mean, they are initialized with:

- **Elo**: `1420.0` (calibrated Championship-to-EPL step-down penalty)
- **Rolling xG**: `1.35` (empirical league prior)
- **Goal expectancy**: λ_h = `1.69`, λ_a = `1.32`

---

## Application Pages

The Streamlit dashboard includes five pages:

| Page | Description |
|:-----|:------------|
| **Home** | Live Elo standings, result distribution charts, goals-per-season trends |
| **Match Predictor** | Select any two EPL clubs to generate outcome probabilities and most likely scoreline |
| **Season Analytics** | Rolling xG trends, match volume, and team-level performance metrics |
| **Head to Head** | Historical head-to-head records and score heatmaps for any club pairing |
| **League Table** | Dynamic league table computed from match results with goal difference |

---

## Performance Summary

| Pipeline Stage | Test Log Loss | Test Brier Score |
|:--------------|:-------------:|:----------------:|
| Naive Null Baseline | 1.0790 | 0.6384 |
| Unengineered XGBoost / RF | 1.0447 | 0.6287 |
| Tuned Base Models | 1.0232 | 0.6241 |
| **Stacked Meta-Learner** | **1.0235** | **0.6238** |
| Raw Poisson Model | 1.0716 | 0.6410 |
| **Zone-Calibrated Poisson** | **1.0235** | **0.6238** |

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost |
| Statistical Modeling | SciPy (Poisson), NumPy |
| Data | pandas, football-data.co.uk, Understat (via `understatapi`) |
| Visualisation | Plotly |
| Application | Streamlit |

---

## Project Structure

```
├── app.py                      # Streamlit multi-page dashboard
├── predictor.py                # Match prediction pipeline (ensemble + Poisson)
├── feature_engine.py           # Elo state, team registry, promoted club logic
├── inference_utils.py          # Stateless feature builders for inference
├── data_loader.py              # Data fetching, normalisation, and name mapping
├── visualizations.py           # Plotly chart library
├── epl_predictor_assets.pkl    # Serialised models, Elo state, and Poisson table
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---

## Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/enartey25/EPL-Meta-Model-and-Poisson-Goal-Model-System.git
cd EPL-Meta-Model-and-Poisson-Goal-Model-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## Data Sources

- **Match results & odds**: [football-data.co.uk](https://www.football-data.co.uk)
- **Expected goals (xG)**: [Understat](https://understat.com) via [`understatapi`](https://github.com/amosbastian/understatapi)

---

## Disclaimer

This project is developed solely for **academic research and educational purposes**. All forecasts are probabilistic estimates derived from historical data. They are **not** intended for sports betting, wagering, or any financial speculation. The author accepts no responsibility for decisions made based on model outputs.

---

## License

[MIT License](LICENSE)
