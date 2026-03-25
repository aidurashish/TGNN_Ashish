"""
    Mobility–Case Lag Correlation
    ===============================================
    Figures produced (all PNG):
    - Lag-correlation bar chart: mean Pearson r(inflow[t], cases[t+k]) across regions, lags k=0..14, one panel per country
    - Best-lag scatter: total inflow vs new cases at the best lag per country, each dot = one (region, date), coloured by region
    - Region-to-region case correlation matrix heatmap: pairwise Pearson r of case time series across all 30 regions, per country (4-panel)
"""

# === IMPORTS === 

import csv
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve project root 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figures" / "correlation_lag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Country registry (w/ designated color scheme)
COUNTRIES = {
    "England": ("data/England [COVID-19]", "england_labels.csv", "EN"),
    "France":  ("data/France [COVID-19]",  "france_labels.csv",  "FR"),
    "Italy":   ("data/Italy [COVID-19]",   "italy_labels.csv",   "IT"),
    "Spain":   ("data/Spain [COVID-19]",   "spain_labels.csv",   "ES"),
}

PALETTE = {"England": "#1f77b4", "France": "#ff7f0e", "Italy": "#2ca02c",   "Spain": "#d62728"}
MAX_LAG = 14

# Helper functions
def _expected_dates():
    dates, d = [], date(2020, 3, 13)
    while d <= date(2020, 5, 12):
        dates.append(str(d))
        d += timedelta(days=1)
    return dates

DATES = _expected_dates()
N = len(DATES)


def load_labels(folder, label_file):
    path = PROJECT_ROOT / folder / label_file
    df = pd.read_csv(path, index_col=0)
    df.columns = pd.to_datetime(df.columns)
    return df.clip(lower=0).astype(float)


def load_inflow_per_region(folder, prefix, regions):
    """
        For each date, compute weighted in-flow to each region (sum of cross-region incoming edge weights).
        
        RETURNS: 
            np.ndarray shape (n_regions, n_dates)
    """
    region_idx = {r: i for i, r in enumerate(regions)}
    inflow = np.zeros((len(regions), N))
    for t, d in enumerate(DATES):
        p = PROJECT_ROOT / folder / "graphs" / f"{prefix}_{d}.csv"
        with open(p, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    src, dst = row[0].strip(), row[1].strip()
                    if src != dst and dst in region_idx:
                        try:
                            inflow[region_idx[dst], t] += float(row[2])
                        except ValueError:
                            pass
    return inflow


# Load all data ─
print("Loading data…")
all_cases   = {}
all_inflow  = {}
regions_map = {}

for country, (folder, lf, prefix) in COUNTRIES.items():
    print(f"  {country}…")
    df = load_labels(folder, lf)
    regions = list(df.index)
    regions_map[country] = regions
    all_cases[country]  = df.values.astype(float)   # (30, 61)
    all_inflow[country] = load_inflow_per_region(folder, prefix, regions)  # (30, 61)

country_names = list(COUNTRIES.keys())


# A) Lag-correlation bar chart
lags = list(range(MAX_LAG + 1))
best_lag_per_country = {}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    col    = PALETTE[country]
    cases  = all_cases[country]   # (30, 61)
    inflow = all_inflow[country]  # (30, 61)
    R      = len(regions_map[country])

    mean_r = []
    for k in lags:
        rs = []
        for r in range(R):
            x = inflow[r, : N - k]
            y = cases[r, k:]
            if np.std(x) > 0 and np.std(y) > 0:
                rval, _ = pearsonr(x, y)
                rs.append(rval)
        mean_r.append(np.mean(rs) if rs else 0.0)

    best_k = int(np.argmax(mean_r))
    best_lag_per_country[country] = best_k

    bars = ax.bar(lags, mean_r, color=col, alpha=0.8, edgecolor="white")
    bars[best_k].set_edgecolor("black")
    bars[best_k].set_linewidth(2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_xlabel("Lag k (days)")
    ax.set_ylabel("Mean Pearson r")
    ax.set_xticks(lags)
    ax.text(best_k, mean_r[best_k] + 0.01, f"k={best_k}",
            ha="center", fontsize=8, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Lag correlation: inflow(t) vs new cases(t+k) — mean across regions", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "A_lag_correlation_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: A_lag_correlation_bar.png")
print("Best lags:", best_lag_per_country)


# B) Best-lag scatter
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    col    = PALETTE[country]
    k      = best_lag_per_country[country]
    cases  = all_cases[country]
    inflow = all_inflow[country]
    regions = regions_map[country]
    R = len(regions)

    cmap = plt.cm.get_cmap("tab20", R)
    for r, region in enumerate(regions):
        x = inflow[r, : N - k]
        y = cases[r, k:]
        ax.scatter(x / 1e3, y, alpha=0.35, s=12, color=cmap(r), label=region)

    ax.set_title(f"{country}  (lag k={k})", fontsize=12, fontweight="bold")
    ax.set_xlabel("In-flow from neighbours (thousands)")
    ax.set_ylabel(f"New cases + {k} days later")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(linestyle="--", alpha=0.35)

fig.suptitle("Inflow vs lagged cases per region (dot = one region-day observation)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "B_best_lag_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: B_best_lag_scatter.png")


# C) Region-to-region case correlation matrix heatmaps
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    cases   = all_cases[country]   # (30, 61)
    regions = regions_map[country]
    R = len(regions)
    corr = np.corrcoef(cases)   # (30, 30)

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True   # upper triangle (optional, show full)
    mask[:] = False  # show full matrix

    sns.heatmap(
        corr, ax=ax,
        xticklabels=regions, yticklabels=regions,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        linewidths=0.3,
    )
    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

fig.suptitle("Pairwise case-count correlation across regions", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "C_region_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: C_region_correlation_matrix.png")

print(f"\nAll figures saved to: {OUT_DIR}")
