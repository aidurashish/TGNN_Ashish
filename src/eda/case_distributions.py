"""
    Case Count Distributions
    ==========================================
    Figures produced (PNG + interactive HTML):
        - Regional epidemic curves: all 30 region lines (faint) + national sum (bold), 2×2 grid
        - Dual-scale national aggregate: linear left-axis/log right-axis, 4 subplots
        - Daily growth rate: (cases_t - cases_{t-1})/cases_{t-1}, one line per country
        - Case count histogram: log-scale x-axis, per country (4-panel)
        - Region x date heatmap: row-normalised 0-1 per country, shows spatial asynchrony (4-panel)
"""

# === IMPORTS === 

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

warnings.filterwarnings("ignore")

# Resolve project root 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figures" / "case_distributions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Country registry (w/ designated color scheme)
COUNTRIES = {
    "England": ("data/England [COVID-19]", "england_labels.csv", "EN"),
    "France":  ("data/France [COVID-19]",  "france_labels.csv",  "FR"),
    "Italy":   ("data/Italy [COVID-19]",   "italy_labels.csv",   "IT"),
    "Spain":   ("data/Spain [COVID-19]",   "spain_labels.csv",   "ES"),
}

PALETTE = {"England": "#1f77b4", "France": "#ff7f0e", "Italy": "#2ca02c",   "Spain": "#d62728"}

# Helper Function
def load_labels(folder, label_file):
    path = PROJECT_ROOT / folder / label_file
    df = pd.read_csv(path, index_col=0)
    df.columns = pd.to_datetime(df.columns)
    df = df.clip(lower=0)   # treat negatives as 0 for plotting
    return df


# Load all countries 
dfs = {}
for country, (folder, lf, _) in COUNTRIES.items():
    dfs[country] = load_labels(folder, lf)

country_names = list(COUNTRIES.keys())

# A) Regional epidemic curves  (Matplotlib PNG + Plotly HTML)
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    df = dfs[country]
    dates = df.columns
    national = df.sum(axis=0)
    color = PALETTE[country]

    for region in df.index:
        ax.plot(dates, df.loc[region], color=color, alpha=0.18, linewidth=0.8)

    ax.plot(dates, national, color=color, linewidth=2.5,
            label="National total", zorder=5)
    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_ylabel("Daily new cases")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Regional epidemic curves (faint) + national total (bold)", fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "A_regional_epidemic_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: A_regional_epidemic_curves.png")

# Plotly interactive version
fig_p = make_subplots(rows=2, cols=2, subplot_titles=country_names, shared_xaxes=False)
rc_map = {n: (r+1, c+1) for n, (r, c) in zip(country_names, [(0,0),(0,1),(1,0),(1,1)])}

for country in country_names:
    df = dfs[country]
    dates_str = [str(d.date()) for d in df.columns]
    national  = df.sum(axis=0).values
    r, c = rc_map[country]
    color = PALETTE[country]

    for region in df.index:
        fig_p.add_trace(go.Scatter(
            x=dates_str, y=df.loc[region].values,
            mode="lines", name=region,
            line=dict(color=color, width=0.7),
            opacity=0.25,
            legendgroup=country, showlegend=False,
            hovertemplate=f"{region}<br>%{{x}}: %{{y}}"
        ), row=r, col=c)

    fig_p.add_trace(go.Scatter(
        x=dates_str, y=national,
        mode="lines", name=f"{country} total",
        line=dict(color=color, width=3),
        legendgroup=country, showlegend=True,
        hovertemplate=f"{country} total<br>%{{x}}: %{{y}}"
    ), row=r, col=c)

fig_p.update_layout(
    title="Regional epidemic curves (interactive)",
    height=700, width=1100,
    template="plotly_white"
)
fig_p.write_html(str(OUT_DIR / "A_regional_epidemic_curves.html"))
print("Saved: A_regional_epidemic_curves.html")


# B) Dual-scale national aggregate (linear + log)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    df = dfs[country]
    national = df.sum(axis=0)
    color = PALETTE[country]

    ax.plot(df.columns, national, color=color, linewidth=2)
    ax.set_ylabel("Daily cases (linear)", color=color)
    ax.tick_params(axis="y", labelcolor=color)

    ax2 = ax.twinx()
    log_vals = np.log1p(national.values)
    ax2.plot(df.columns, log_vals, color=color, linewidth=1.5, linestyle="--", alpha=0.6)
    ax2.set_ylabel("log(1 + cases)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

fig.suptitle("National daily cases — linear (solid) vs log-scale (dashed)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "B_national_linear_vs_log.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: B_national_linear_vs_log.png")


# C) Daily growth rates
fig, ax = plt.subplots(figsize=(14, 5))
for country in country_names:
    national = dfs[country].sum(axis=0)
    growth = national.pct_change().replace([np.inf, -np.inf], np.nan)
    ax.plot(dfs[country].columns, growth,
            label=country, linewidth=1.6, color=PALETTE[country])

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Daily national case growth rate  (cases_t − cases_{t-1}) / cases_{t-1}", fontsize=13, fontweight="bold")
ax.set_ylabel("Growth rate")
ax.set_ylim(-1.5, 3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "C_daily_growth_rates.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: C_daily_growth_rates.png")


# D) Case count histogram (log-scale x)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    df = dfs[country]
    vals = df.values.flatten()
    vals = vals[vals > 0]   # skip zeros for log scale

    ax.hist(vals, bins=60, color=PALETTE[country], edgecolor="white", linewidth=0.4, alpha=0.85, log=False)
    ax.set_xscale("log")
    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_xlabel("Daily cases per region (log scale)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate median
    med = np.median(vals)
    ax.axvline(med, color="black", linewidth=1.2, linestyle="--")
    ax.text(med * 1.1, ax.get_ylim()[1] * 0.9, f"median={med:.0f}", fontsize=8, color="black")

fig.suptitle("Distribution of daily regional case counts (log x-axis, zeros excluded)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "D_case_count_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: D_case_count_histogram.png")


# E) Region x date heatmap (row-normalised)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for ax, country in zip(axes, country_names):
    df = dfs[country].copy()
    # Row-normalise each region 0–1
    row_max = df.max(axis=1).replace(0, 1)
    df_norm = df.div(row_max, axis=0)

    # X-tick labels: approx weekly
    date_strs = [str(d.date()) for d in df.columns]
    tick_idx   = list(range(0, len(date_strs), 7))
    tick_lbls  = [date_strs[i] for i in tick_idx]

    sns.heatmap(df_norm, ax=ax, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label": "Normalised cases (0–1)", "shrink": 0.6}, xticklabels=False, yticklabels=True, linewidths=0)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_lbls, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title(country, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Region")

fig.suptitle("Region x date heatmap — row-normalised (0=min, 1=peak per region)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "E_region_date_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: E_region_date_heatmap.png")

print(f"\nAll figures saved to: {OUT_DIR}")
