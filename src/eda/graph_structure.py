"""
    Daily Graph Structure 
    =========================================
    Figures produced (all PNG):
        - Mean/median weighted in-strength over time per country (4-panel)
        - Degree distribution histogram: averaged across all 61 days, per country (4-panel)
        - Top-5 hub regions by average in-strength: horizontal bar chart (4-panel)
        - Graph density over time: actual cross-edges / max possible (4-panel)
        - 30x30 adjacency heatmap: one representative date (log-scaled), per country (4-panel)
"""

# === IMPORTS === 

import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve project root 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figures" / "graph_structure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Country registry (w/ designated color scheme)
COUNTRIES = {
    "England": ("data/England [COVID-19]", "england_labels.csv", "EN"),
    "France":  ("data/France [COVID-19]",  "france_labels.csv",  "FR"),
    "Italy":   ("data/Italy [COVID-19]",   "italy_labels.csv",   "IT"),
    "Spain":   ("data/Spain [COVID-19]",   "spain_labels.csv",   "ES"),
}

PALETTE = {"England": "#1f77b4", "France": "#ff7f0e", "Italy": "#2ca02c",   "Spain": "#d62728"}
N_REGIONS = 30

# Helper functions
def _expected_dates():
    dates, d = [], date(2020, 3, 13)
    while d <= date(2020, 5, 12):
        dates.append(str(d))
        d += timedelta(days=1)
    return dates

DATES = _expected_dates()
N = len(DATES)
DATE_OBJS = [pd.Timestamp(d) for d in DATES]


def load_labels(folder, label_file):
    path = PROJECT_ROOT / folder / label_file
    df = pd.read_csv(path, index_col=0)
    return list(df.index)


def load_graph(folder, prefix, d, regions):
    """
    RETURNS:
        - W         : (N_REGIONS × N_REGIONS) np.ndarray, cross-region flows only
        - in_degree : (N_REGIONS,) — number of distinct in-neighbours (cross-region)
        - in_strength: (N_REGIONS,) — sum of incoming cross-region weights
    """
    idx = {r: i for i, r in enumerate(regions)}
    W = np.zeros((N_REGIONS, N_REGIONS))
    p = PROJECT_ROOT / folder / "graphs" / f"{prefix}_{d}.csv"
    with open(p, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                src, dst = row[0].strip(), row[1].strip()
                if src != dst and src in idx and dst in idx:
                    try:
                        W[idx[src], idx[dst]] += float(row[2])
                    except ValueError:
                        pass
    in_degree   = (W > 0).sum(axis=0).astype(float)
    in_strength = W.sum(axis=0)
    return W, in_degree, in_strength


# Compute per-day stats for all countries
print("Computing graph stats (this may take ~60 s)…")

data = {}   # { country: { "in_strength": (N_REGIONS, N),
            #               "in_degree":   (N_REGIONS, N),
            #               "density":     (N,),
            #               "regions":     [str],
            #               "W_repr":      (N_REGIONS, N_REGIONS)  # representative adj
            # } }

for country, (folder, lf, prefix) in COUNTRIES.items():
    print(f"  {country}…")
    regions = load_labels(folder, lf)
    in_str = np.zeros((N_REGIONS, N))
    in_deg = np.zeros((N_REGIONS, N))
    densities = np.zeros(N)
    max_edges = N_REGIONS * (N_REGIONS - 1)

    # Representative date: pick peak-case day and load labels to determine peak
    df_lab = pd.read_csv(PROJECT_ROOT / folder / lf, index_col=0)
    national = df_lab.sum(axis=0)
    peak_date = national.idxmax()   # string like '2020-04-01'

    W_repr = None
    for t, d in enumerate(DATES):
        W, deg, strength = load_graph(folder, prefix, d, regions)
        in_str[:, t] = strength
        in_deg[:, t] = deg
        actual_edges = int((W > 0).sum())
        densities[t] = actual_edges / max_edges if max_edges > 0 else 0
        if d == peak_date:
            W_repr = W.copy()

    if W_repr is None:
        W_repr = load_graph(folder, prefix, DATES[N // 2], regions)[0]

    data[country] = {
        "in_strength": in_str,
        "in_degree":   in_deg,
        "density":     densities,
        "regions":     regions,
        "W_repr":      W_repr,
        "peak_date":   peak_date,
    }

    # Save per-day stats CSV
    rows = []
    for t, d in enumerate(DATES):
        rows.append({
            "date":           d,
            "mean_in_strength": float(in_str[:, t].mean()),
            "median_in_strength": float(np.median(in_str[:, t])),
            "mean_in_degree":  float(in_deg[:, t].mean()),
            "graph_density":   float(densities[t]),
        })
    stats_df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"graph_stats_{country.lower()}.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"    Saved: graph_stats_{country.lower()}.csv")

country_names = list(COUNTRIES.keys())


# A) Mean/median in-strength over time
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    col = PALETTE[country]
    in_str = data[country]["in_strength"]   # (30, 61)

    mean_s   = in_str.mean(axis=0)
    median_s = np.median(in_str, axis=0)
    p25      = np.percentile(in_str, 25, axis=0)
    p75      = np.percentile(in_str, 75, axis=0)

    ax.fill_between(DATE_OBJS, p25 / 1e3, p75 / 1e3,
                    color=col, alpha=0.15, label="IQR")
    ax.plot(DATE_OBJS, mean_s / 1e3,   color=col, linewidth=2.2, label="Mean")
    ax.plot(DATE_OBJS, median_s / 1e3, color=col, linewidth=1.6,
            linestyle="--", alpha=0.75, label="Median")

    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_ylabel("In-strength (thousands)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.35)

fig.suptitle("Cross-region weighted in-strength over time (mean / median / IQR)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "A_instrength_over_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: A_instrength_over_time.png")


# B) Degree distribution histogram (averaged across all days)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    col    = PALETTE[country]
    in_deg = data[country]["in_degree"]   # (30, 61)
    # Average degree per region across all days
    avg_deg = in_deg.mean(axis=1)

    ax.bar(range(N_REGIONS), sorted(avg_deg, reverse=True), color=col, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_xlabel("Region rank (sorted by avg degree)")
    ax.set_ylabel("Avg in-degree")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Average cross-region in-degree per region (sorted descending)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "B_degree_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: B_degree_distribution.png")


# C) Top-5 hub regions by average in-strength
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    col     = PALETTE[country]
    in_str  = data[country]["in_strength"]
    regions = data[country]["regions"]
    avg_str = in_str.mean(axis=1)
    top5_idx = np.argsort(avg_str)[-5:][::-1]
    top5_names = [regions[i] for i in top5_idx]
    top5_vals  = [avg_str[i] / 1e3 for i in top5_idx]

    ax.barh(range(5), top5_vals, color=col, edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.set_yticks(range(5))
    ax.set_yticklabels(top5_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_xlabel("Average in-strength (thousands)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

fig.suptitle("Top-5 hub regions by average cross-region weighted in-strength", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "C_top5_hubs.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: C_top5_hubs.png")


# D) Graph density over time
fig, ax = plt.subplots(figsize=(14, 5))
for country in country_names:
    ax.plot(DATE_OBJS, data[country]["density"] * 100,
            label=country, linewidth=2, color=PALETTE[country])

ax.set_title("Graph density over time  (cross-region edges / max possible x 100%)", fontsize=13, fontweight="bold")
ax.set_ylabel("Density (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "D_graph_density_over_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: D_graph_density_over_time.png")


# E) 30×30 adjacency heatmap — representative date (peak)
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    W       = data[country]["W_repr"]
    regions = data[country]["regions"]
    peak    = data[country]["peak_date"]

    W_log = np.log1p(W)
    sns.heatmap(W_log, ax=ax,
                xticklabels=regions, yticklabels=regions,
                cmap="viridis",
                cbar_kws={"label": "log(1 + flow)", "shrink": 0.55},
                linewidths=0.2, square=False)
    ax.set_title(f"{country}  (peak: {peak})", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    ax.set_xlabel("Destination region")
    ax.set_ylabel("Source region")

fig.suptitle("Cross-region mobility adjacency matrix at peak case date (log scale)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "E_adjacency_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: E_adjacency_heatmap.png")

print(f"\nAll figures saved to: {OUT_DIR}")
