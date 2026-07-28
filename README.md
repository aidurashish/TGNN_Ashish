# DiffATMGNN — Diffusion-based Attention Temporal Multi-resolution Graph Neural Network for Epidemic Forecasting

> Bachelor's Thesis Prototype · Ashish Aidur
> *Multi-Resolution Temporal Graph Attention Networks for Epidemic Forecasting: Biological Grounding and Diffusion over Dynamic Human Mobility Networks*

***

## Overview

This repository contains the full proof-of-concept implementation for the DiffATMGNN model: an epidemic forecasting architecture that combines:

- **Dynamic human mobility graphs**: regions are connected by real cross-regional movement data, updated at each time step
- **Multi-resolution spatial learning**: the mobility graph is coarsened into two additional levels (10 and 5 clusters) to capture both local and national-scale spread patterns
- **SEIR-based biological grounding**: estimated SEIR compartment values (Susceptible, Exposed, Infectious, Recovered) are injected as additional feature nodes, giving the model an understanding of the underlying disease dynamics
- **Diffusion-based probabilistic forecasting**: instead of a single point prediction, the model generates a full distribution over future case trajectories using a denoising diffusion decoder, enabling calibrated uncertainty estimation

The model was trained and evaluated on daily COVID-19 case counts and mobility data from four European countries, **England, France, Italy, and Spain**, covering the first pandemic wave (March–May 2020). Seven forecast horizons (shifts 0–6) were evaluated per country.

***

## Repository Structure

```
TGNN_Ashish/
│
├── src/
│   ├── models.py                  # DiffATMGNN, ATMGNN, ATMGNN+SEIR, BiLSTM, STAN model definitions
│   ├── ATMGNN_Diff_training.py    # Main training & evaluation loop for DiffATMGNN
│   ├── ATMGNN_training.py         # Training loop for baseline ATMGNN variants
│   ├── ablation.py                # Ablation study and model comparison runner
│   ├── optuna_hpo.py              # Hyperparameter optimisation via Optuna (TPE sampler)
│   ├── utils.py                   # Data loading, SEIR fitting, graph construction, metrics
│   │
│   └── eda/                       # Exploratory data analysis scripts
│       ├── case_distributions.py      # Distribution plots of regional case counts
│       ├── correlation_lag.py         # Cross-regional case correlation and lag analysis
│       ├── graph_structure.py         # Mobility graph topology visualisation
│       ├── integrity.py               # Dataset integrity and missing value checks
│       ├── mobility_distributions.py  # Mobility flow distribution analysis
│       ├── temporal_trends.py         # Temporal trend visualisation per country
│       └── uncertainty_analysis.py    # PICP, Spearman ρ, CRPS uncertainty evaluation
│
├── data/                          # Raw and preprocessed datasets (not tracked by Git)
├── predictions/                   # Model output CSVs (truth and prediction files per shift/country)
├── figures/                       # Generated plots and visualisation outputs
├── requirements.txt               # Python dependencies
├── skewness_check.py              # Validates right-skewness of England case distribution
├── symmetric_band_failure.py      # Demonstrates failure of symmetric uncertainty bands
└── .gitignore
```

***

## Models Implemented

| Model | Description |
|---|---|
| **DiffATMGNN** | Full proposed model with SEIR grounding + diffusion decoder |
| **ATMGNN + SEIR** | Attention temporal multi-resolution GNN with SEIR features, no diffusion |
| **ATMGNN** | Attention temporal multi-resolution GNN without SEIR features |
| **BiLSTM** | Bidirectional LSTM baseline |
| **STAN** | Spatio-Temporal Attention Network baseline |
| **ARIMA** | Classical statistical baseline |

***

## Setup

### Requirements

Python 3.8+ is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torch-geometric`, `networkx`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `optuna`, `seaborn`, `plotly`.

### Data

The dataset is derived from [Panagopoulos et al. (2021)](https://arxiv.org/abs/2009.08388) and uses Facebook's [Data for Good](https://dataforgood.facebook.com/dfg/tools/movement-maps) cross-regional mobility graphs. Place raw data files in the `data/` directory before running any training scripts.

***

## Usage

### Train DiffATMGNN

```bash
python src/ATMGNN_Diff_training.py
```

### Run Ablation / Model Comparison

```bash
python src/ablation.py
```

### Hyperparameter Optimisation

```bash
python src/optuna_hpo.py
```

### Uncertainty Analysis

```bash
python src/eda/uncertainty_analysis.py
```

### EDA Scripts

All scripts under `src/eda/` can be run independently and will output plots to the `figures/` directory.

***

## Uncertainty Evaluation Metrics

The uncertainty analysis evaluates three metrics:

- **PICP** (Prediction Interval Coverage Probability): what fraction of true values fall within the predicted band
- **Spearman ρ**: rank correlation between prediction error and band width. Positive values confirm the model widens its bands where it is less certain
- **CRPS** (Continuous Ranked Probability Score): measures the overall quality of the predicted distribution; lower is better

Post-hoc asymmetric scaling is applied after inference to correct for the right-skewed distribution of epidemic case counts.

***