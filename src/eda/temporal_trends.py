"""
    Temporal Trends
    =================================
    Figures produced:
        - Dual-axis time series (PNG + Plotly HTML) — left: daily new cases, right: total cross-region mobility, per country (4-panel)
        - Normalised overlay (PNG + Plotly HTML) — both series scaled 0-1 on the same axis, per country (4-panel)
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Resolve project root 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figures" / "temporal_trends"
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
def load_labels(folder, label_file):
    path = PROJECT_ROOT / folder / label_file
    df = pd.read_csv(path, index_col=0)
    if "name" in df.columns:          # Italy has extra 'name' and 'id' columns
        df = df.set_index("name")
    date_cols = [c for c in df.columns if str(c).startswith("20")]
    df = df[date_cols]
    df.columns = pd.to_datetime(df.columns)
    return df.clip(lower=0)


def load_cross_mobility(folder, prefix, dates):
    """Return array of total cross-region flow per day (length = len(dates))."""
    totals = []
    for d in dates:
        p = PROJECT_ROOT / folder / "graphs" / f"{prefix}_{d}.csv"
        if not p.exists():
            totals.append(0.0)
            continue
        total = 0.0
        with open(p, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0].strip() != row[1].strip():
                    try:
                        total += float(row[2])
                    except ValueError:
                        pass
        totals.append(total)
    return np.array(totals)


# Load data
print("Loading data…")
cases_series    = {}
mobility_series = {}
date_objs_by    = {}   # per-country list of Timestamps
date_str_by     = {}   # per-country list of date strings

for country, (folder, lf, prefix) in COUNTRIES.items():
    print(f"  Loading {country}…")
    df   = load_labels(folder, lf)
    dates_str = [str(d.date()) for d in df.columns]
    cases_series[country]    = df.sum(axis=0).values.astype(float)
    mobility_series[country] = load_cross_mobility(folder, prefix, dates_str)
    date_objs_by[country]    = list(df.columns)
    date_str_by[country]     = dates_str

country_names = list(COUNTRIES.keys())


def _minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)


# A) Dual-axis time series — PNG
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    col   = PALETTE[country]
    cases = cases_series[country]
    mob   = mobility_series[country]
    date_objs = date_objs_by[country]

    # Left axis — cases
    ax.plot(date_objs, cases, color=col, linewidth=2.2, label="Daily new cases")
    ax.set_ylabel("Daily new cases", color=col)
    ax.tick_params(axis="y", labelcolor=col)

    # Right axis — mobility
    ax2 = ax.twinx()
    ax2.plot(date_objs, mob / 1e6, color=col, linewidth=1.8, linestyle="--", alpha=0.65, label="Cross-region flow")
    ax2.set_ylabel("Cross-region flow (millions)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Combined legend
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")

fig.suptitle("Daily new cases (solid) vs total cross-region mobility (dashed)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "A_dual_axis_trends.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: A_dual_axis_trends.png")


# B) Normalised overlay — PNG
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    col    = PALETTE[country]
    date_objs = date_objs_by[country]
    c_norm = _minmax(cases_series[country])
    m_norm = _minmax(mobility_series[country])

    ax.plot(date_objs, c_norm, color=col, linewidth=2.2, label="Cases (normalised)")
    ax.plot(date_objs, m_norm, color=col, linewidth=1.8, linestyle="--",
            alpha=0.65, label="Mobility (normalised)")
    ax.fill_between(date_objs, c_norm, m_norm, alpha=0.08, color=col)

    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalised value (0–1)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.35)

fig.suptitle("Normalised cases (solid) vs normalised cross-region mobility (dashed)",fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "B_normalised_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: B_normalised_overlay.png")


# Interactive Plotly — dual-axis + normalised (combined HTML)
rc_map = {n: (r+1, c+1) for n, (r, c) in zip(country_names, [(0,0),(0,1),(1,0),(1,1)])}

# Figure 1: dual-axis
fig_p = make_subplots(
    rows=2, cols=2,
    subplot_titles=country_names,
    specs=[[{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]],
)

for country in country_names:
    r, c = rc_map[country]
    col  = PALETTE[country]
    cases = cases_series[country]
    mob   = mobility_series[country]
    date_str_list = date_str_by[country]

    fig_p.add_trace(go.Scatter(
        x=date_str_list, y=cases.tolist(),
        name=f"{country} cases",
        line=dict(color=col, width=2),
        hovertemplate="%{x}<br>Cases: %{y:,.0f}",
    ), row=r, col=c, secondary_y=False)

    fig_p.add_trace(go.Scatter(
        x=date_str_list, y=(mob / 1e6).tolist(),
        name=f"{country} mobility",
        line=dict(color=col, width=1.8, dash="dash"),
        opacity=0.65,
        hovertemplate="%{x}<br>Mobility: %{y:.2f}M",
    ), row=r, col=c, secondary_y=True)

fig_p.update_layout(
    title="Daily cases vs cross-region mobility (interactive)",
    height=700, template="plotly_white", width=1100,
)
fig_p.write_html(str(OUT_DIR / "A_dual_axis_trends.html"))
print("Saved: A_dual_axis_trends.html")

# Figure 2: normalised
fig_p2 = make_subplots(rows=2, cols=2, subplot_titles=country_names)
for country in country_names:
    r, c = rc_map[country]
    col  = PALETTE[country]
    c_n  = _minmax(cases_series[country])
    m_n  = _minmax(mobility_series[country])
    date_str_list = date_str_by[country]

    fig_p2.add_trace(go.Scatter(
        x=date_str_list, y=c_n.tolist(),
        name=f"{country} cases",
        line=dict(color=col, width=2),
    ), row=r, col=c)
    fig_p2.add_trace(go.Scatter(
        x=date_str_list, y=m_n.tolist(),
        name=f"{country} mobility",
        line=dict(color=col, width=1.8, dash="dash"),
        opacity=0.65,
    ), row=r, col=c)

fig_p2.update_layout(
    title="Normalised cases vs mobility (interactive)",
    height=700, template="plotly_white", width=1100,
)
fig_p2.write_html(str(OUT_DIR / "B_normalised_overlay.html"))
print("Saved: B_normalised_overlay.html")

print(f"\nAll figures saved to: {OUT_DIR}")
