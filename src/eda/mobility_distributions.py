"""
    Mobility Distributions
    ========================================
    Figures produced (all PNG):
        - Edge weight histogram: cross-region flows, log-scale x, per country (4-panel)
        - Self-loop vs cross-region total flow over time: dual-line chart per country (4-panel)
        - Cross-region fraction over time: fraction of total flow that is inter-regional (4-panel)
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
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figures" / "mobility_distributions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Country registry (w/ designated color scheme)
COUNTRIES = {
    "England": ("data/England [COVID-19]", "england_labels.csv", "EN"),
    "France":  ("data/France [COVID-19]",  "france_labels.csv",  "FR"),
    "Italy":   ("data/Italy [COVID-19]",   "italy_labels.csv",   "IT"),
    "Spain":   ("data/Spain [COVID-19]",   "spain_labels.csv",   "ES"),
}

PALETTE = {"England": "#1f77b4", "France": "#ff7f0e", "Italy": "#2ca02c",   "Spain": "#d62728"}

# Helper functions
def load_graph_edges(folder, prefix, date_str):
    """Return list of (src, dst, weight) for one graph file, or [] if missing."""
    p = PROJECT_ROOT / folder / "graphs" / f"{prefix}_{date_str}.csv"
    if not p.exists():
        return []
    edges = []
    with open(p, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                try:
                    edges.append((row[0].strip(), row[1].strip(), float(row[2])))
                except ValueError:
                    pass
    return edges


def _get_dates(folder, label_file):
    """Return list of date strings (YYYY-MM-DD) from label CSV."""
    df = pd.read_csv(PROJECT_ROOT / folder / label_file, index_col=0)
    if "name" in df.columns:
        df = df.set_index("name")
    return [c for c in df.columns if str(c).startswith("20")]


# Pre-compute per-day mobility stats for all countries
print("Loading graph files…")

# Stores:  { country: { "cross": array, "self": array, "all_cross": array, "date_objs": list } }
mob = {}
for country, (folder, lf, prefix) in COUNTRIES.items():
    dates = _get_dates(folder, lf)
    cross_totals, self_totals, all_cross = [], [], []
    for d in dates:
        edges = load_graph_edges(folder, prefix, d)
        self_w  = sum(w for s, t, w in edges if s == t)
        cross_w = sum(w for s, t, w in edges if s != t)
        self_totals.append(self_w)
        cross_totals.append(cross_w)
        all_cross.extend(w for s, t, w in edges if s != t and w > 0)
    mob[country] = {
        "cross":     np.array(cross_totals),
        "self":      np.array(self_totals),
        "all_cross": np.array(all_cross),
        "date_objs": [pd.Timestamp(d) for d in dates],
    }
    print(f"  {country}: {len(all_cross):,} cross-region edge observations loaded")


# A) Edge weight histogram — log-scale x
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, country in zip(axes, list(COUNTRIES.keys())):
    vals = mob[country]["all_cross"]
    log_vals = np.log10(vals[vals > 0])
    bins = np.linspace(log_vals.min(), log_vals.max(), 60)
    ax.hist(log_vals, bins=bins, color=PALETTE[country],
            edgecolor="white", linewidth=0.4, alpha=0.85)

    ax.set_xlabel("log₁₀(cross-region flow)")
    ax.set_ylabel("Frequency")
    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    med = np.median(log_vals)
    ax.axvline(med, color="black", linewidth=1.2, linestyle="--")
    ax.text(med + 0.05, ax.get_ylim()[1] * 0.88, f"median=10^{med:.1f}", fontsize=8)

fig.suptitle("Cross-region mobility edge weight distribution (log₁₀ scale)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "A_edge_weight_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: A_edge_weight_histogram.png")


# B) Self-loop vs cross-region total flow over time
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
axes = axes.flatten()

for ax, country in zip(axes, list(COUNTRIES.keys())):
    col    = PALETTE[country]
    cross  = mob[country]["cross"]
    self_  = mob[country]["self"]
    date_objs = mob[country]["date_objs"]

    ax.plot(date_objs, cross / 1e6, label="Cross-region", color=col,
            linewidth=2)
    ax.plot(date_objs, self_ / 1e6, label="Self-loop (internal)",
            color=col, linewidth=2, linestyle="--", alpha=0.7)

    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_ylabel("Total flow (millions)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.35)

fig.suptitle("Total self-loop vs cross-region mobility flow over time", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "B_self_vs_cross_flow.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: B_self_vs_cross_flow.png")


# C) Cross-region fraction over time
fig, ax = plt.subplots(figsize=(14, 5))

for country in COUNTRIES:
    cross  = mob[country]["cross"]
    total  = mob[country]["cross"] + mob[country]["self"]
    frac   = np.where(total > 0, cross / total, np.nan)
    ax.plot(mob[country]["date_objs"], frac * 100, label=country, linewidth=2, color=PALETTE[country])

ax.set_title("Cross-region flow as % of total mobility (lockdown effect)", fontsize=13, fontweight="bold")
ax.set_ylabel("Cross-region share (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "C_cross_region_fraction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: C_cross_region_fraction.png")

print(f"\nAll figures saved to: {OUT_DIR}")
