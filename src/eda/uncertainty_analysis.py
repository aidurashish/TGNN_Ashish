"""
    Uncertainty Analysis Script that loads the predictions, ground-truth, and uncertainty CSVs produced by
    ATMGNN_Diff_training.py and computes metrics that demonstrate whether the diffusion-based uncertainty estimates are meaningful
    or simply noise, that too, in comparison to ATMGNN (baseline) in terms of MAE-based accuracy.

    Computed Metrics:
        - PICP_1sigma     : Prediction Interval Coverage Probability at ±1σ  (target: 68%)
        - PICP_2sigma     : Prediction Interval Coverage Probability at ±2σ  (target: 95%)
        - PICP_1sigma_cal : PICP at ±1σ after conformal calibration (forced to ~68% by construction)
        - PICP_2sigma_cal : PICP at ±2σ after conformal calibration
        - spearman_rho    : Spearman correlation between per-node uncertainty and absolute error
        - spearman_p      : p-value of the Spearman correlation
        - mean_CRPS       : Mean interval score of the raw diffusion distribution
        - mean_CRPS_cal   : Mean interval score after conformal calibration (q-scaled σ)
        - baseline_CRPS   : Mean interval score of a naive rolling-std baseline
        - calib_factor    : Conformal calibration factor q (68th percentile of abs. error/σ)
        - fc_mae          : MAE of ATMGNN_Diff FC-head predictions
        - atmgnn_mae      : MAE of ATMGNN baseline predictions (for direct comparison)
"""

# === IMPORTS ===

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# Resolve project root regardless of where the script is called from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRED_DIR = PROJECT_ROOT / "predictions"
OUT_DIR  = PROJECT_ROOT / "figures" / "uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["IT", "EN", "FR", "ES"]
MODEL     = "ATMGNN_Diff"

# Number of days used to compute the rolling-std naive baseline per region.
BASELINE_WINDOW = 7


# === HELPERS ===

def _load_csv(path):
    """Load a predictions/truth/uncertainty CSV."""
    if not path.exists():
        return None
    return np.loadtxt(path, delimiter=',')


def _interval_score(lower, upper, y_true, alpha):
    """
        Generates Winkler interval score (mean over all observations).
        Lower alpha = narrower nominal interval (e.g. alpha=0.32 for ±1σ, alpha=0.05 for ±2σ).
        Penalises both overconfidence (coverage below 1-alpha) and underconfidence (wide intervals).
    """
    width     = upper - lower
    pen_below = np.maximum(lower - y_true, 0.0)
    pen_above = np.maximum(y_true - upper, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (pen_below + pen_above)))


def _baseline_uncertainty(y_true, window=BASELINE_WINDOW):
    """
        Naive baseline: rolling standard deviation of ground-truth case counts per region.
        Shape of y_true: [n_nodes, n_days].
        Returns rolling-std array of same shape 
    """
    _, n_days = y_true.shape
    baseline = np.zeros_like(y_true, dtype=float)
    for d in range(n_days):
        start = max(0, d - window + 1)
        baseline[:, d] = y_true[:, start:d + 1].std(axis=1)
    return baseline


def _compute_metrics(y_pred, y_true, y_uncert):
    """
        Compute all validation metrics for one (country, shift) pair.

        ARGS:
            - y_pred   : [n_nodes, n_days]  FC-head predictions
            - y_true   : [n_nodes, n_days]  ground-truth case counts
            - y_uncert : [n_nodes, n_days]  per-node uncertainty (±1σ half-width in case-count space)

        RETURNS:
            - dict of metric name -> value
    """
    metrics = {}

    # Prediction Interval Coverage Probability
    lower_1s = y_pred - y_uncert
    upper_1s = y_pred + y_uncert
    lower_2s = y_pred - 2.0 * y_uncert
    upper_2s = y_pred + 2.0 * y_uncert

    covered_1s = (y_true >= lower_1s) & (y_true <= upper_1s)
    covered_2s = (y_true >= lower_2s) & (y_true <= upper_2s)
    metrics["PICP_1sigma"] = float(covered_1s.mean())
    metrics["PICP_2sigma"] = float(covered_2s.mean())

    # Spearman correlation: uncertainty vs absolute error
    abs_err   = np.abs(y_pred - y_true).ravel()
    unc_flat  = y_uncert.ravel()
    rho, pval = spearmanr(unc_flat, abs_err)
    metrics["spearman_rho"] = float(rho)
    metrics["spearman_p"]   = float(pval)

    # Interval score (proxy for CRPS) at ±1σ nominal level
    # ±1σ corresponds to an 68% nominal interval, so alpha = 0.32.
    metrics["mean_CRPS"] = _interval_score(lower_1s, upper_1s, y_true, alpha=0.32)

    # Naive baseline: rolling-std of true case counts used as the uncertainty estimate.
    baseline_unc  = _baseline_uncertainty(y_true)
    base_lower_1s = y_pred - baseline_unc
    base_upper_1s = y_pred + baseline_unc
    metrics["baseline_CRPS"] = _interval_score(base_lower_1s, base_upper_1s, y_true, alpha=0.32)

    # Accuracy of FC head
    metrics["fc_mae"] = float(np.mean(np.abs(y_pred - y_true)))

    # Conformal calibration factor
    # q = 68th percentile of |error|/σ.  Calibrated ±q·σ interval covers ~68% by construction.
    valid = unc_flat > 0
    if valid.sum() > 10:
        ratios = abs_err[valid] / unc_flat[valid]
        q = float(np.percentile(ratios, 68.0))
    else:
        q = 1.0
    metrics["calib_factor"]   = q
    y_uncert_cal              = y_uncert * q
    lower_1s_cal              = y_pred - y_uncert_cal
    upper_1s_cal              = y_pred + y_uncert_cal
    lower_2s_cal              = y_pred - 2.0 * y_uncert_cal
    upper_2s_cal              = y_pred + 2.0 * y_uncert_cal
    metrics["PICP_1sigma_cal"] = float(((y_true >= lower_1s_cal) & (y_true <= upper_1s_cal)).mean())
    metrics["PICP_2sigma_cal"] = float(((y_true >= lower_2s_cal) & (y_true <= upper_2s_cal)).mean())
    metrics["mean_CRPS_cal"]   = _interval_score(lower_1s_cal, upper_1s_cal, y_true, alpha=0.32)

    return metrics


# === PLOTTING ===

def _plot_calibration(summary_df, country, out_dir):
    """Bar chart: expected vs actual coverage at ±1σ and ±2σ, one group of bars per shift."""
    df = summary_df[summary_df["country"] == country].copy()
    if df.empty:
        return

    shifts = sorted(df["shift"].unique())
    x      = np.arange(len(shifts))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(shifts) * 1.5), 4))

    ax.bar(x - width * 1.5, df["PICP_1sigma"].values,     width, label="Raw ±1σ",        color="steelblue",   alpha=0.6)
    ax.bar(x - width * 0.5, df["PICP_1sigma_cal"].values,  width, label="Calibrated ±1σ", color="steelblue")
    ax.bar(x + width * 0.5, df["PICP_2sigma"].values,     width, label="Raw ±2σ",        color="tomato",     alpha=0.6)
    ax.bar(x + width * 1.5, df["PICP_2sigma_cal"].values,  width, label="Calibrated ±2σ", color="tomato")
    ax.axhline(0.68, color="steelblue", linestyle="--", linewidth=1.2, label="Target 68%")
    ax.axhline(0.95, color="tomato",    linestyle="--", linewidth=1.2, label="Target 95%")

    ax.set_xticks(x)
    ax.set_xticklabels(["Shift +{}".format(s) for s in shifts])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Coverage rate")
    ax.set_title("{} — Uncertainty Calibration (PICP)".format(country))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "calibration_{}.png".format(country)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  [PLOT] Calibration plot saved to {}".format(path))


def _plot_error_vs_uncertainty(y_pred, y_true, y_uncert, country, shift, out_dir):
    """Scatter plot of per-node per-day absolute error vs uncertainty."""
    abs_err  = np.abs(y_pred - y_true).ravel()
    unc_flat = y_uncert.ravel()

    # Subsample for readability if very large.
    if len(abs_err) > 5000:
        idx      = np.random.choice(len(abs_err), 5000, replace=False)
        abs_err  = abs_err[idx]
        unc_flat = unc_flat[idx]

    rho, pval = spearmanr(unc_flat, abs_err)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(unc_flat, abs_err, alpha=0.3, s=10, color="steelblue")
    ax.set_xlabel("Uncertainty (±1σ half-width, cases)")
    ax.set_ylabel("Absolute forecast error (cases)")
    ax.set_title("{} — Shift +{} — Uncertainty vs Error\nSpearman ρ={:.3f}  p={:.3g}".format(
        country, shift, rho, pval))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "error_vs_uncertainty_{}_shift{}.png".format(country, shift)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  [PLOT] Error vs uncertainty saved to {}".format(path))


def _plot_crps_comparison(summary_df, out_dir):
    """Side-by-side bar chart: diffusion interval score vs naive baseline, per country."""
    countries  = summary_df["country"].unique()
    n          = len(countries)
    fig, axes  = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for i, country in enumerate(countries):
        ax  = axes[0][i]
        df  = summary_df[summary_df["country"] == country]
        shifts = sorted(df["shift"].unique())
        x      = np.arange(len(shifts))
        w      = 0.25
        ax.bar(x - w,       df["mean_CRPS"].values,     w, label="Diffusion (raw)",        color="steelblue", alpha=0.6)
        ax.bar(x,           df["mean_CRPS_cal"].values, w, label="Diffusion (calibrated)", color="steelblue")
        ax.bar(x + w,       df["baseline_CRPS"].values, w, label="Naive rolling-std",      color="lightcoral")
        ax.set_xticks(x)
        ax.set_xticklabels(["S+{}".format(s) for s in shifts], fontsize=8)
        ax.set_title(country)
        ax.set_ylabel("Interval score (lower = better)")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Diffusion Uncertainty vs Naive Baseline — Interval Score", fontsize=11)
    fig.tight_layout()
    path = out_dir / "crps_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  [PLOT] CRPS comparison saved to {}".format(path))


def _plot_fc_vs_diff_accuracy(summary_df, out_dir):
    """Line plot comparing ATMGNN baseline MAE vs ATMGNN_Diff FC-head MAE per shift per country."""
    countries = summary_df["country"].unique()
    n         = len(countries)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for i, country in enumerate(countries):
        ax = axes[0][i]
        df = summary_df[summary_df["country"] == country].sort_values("shift")
        ax.plot(df["shift"], df["fc_mae"], marker="o", label="ATMGNN_Diff (FC head)", color="steelblue")
        bm = df["atmgnn_mae"].values
        if not np.all(np.isnan(bm)):
            ax.plot(df["shift"], bm, marker="s", label="ATMGNN (baseline)",
                    color="tomato", linestyle="--")
        ax.set_title(country)
        ax.set_xlabel("Shift (days ahead)")
        ax.set_ylabel("MAE (cases)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("ATMGNN Baseline vs ATMGNN_Diff — Forecast Accuracy (MAE)", fontsize=11)
    fig.tight_layout()
    path = out_dir / "fc_vs_diff_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  [PLOT] Accuracy comparison saved to {}".format(path))


# === MAIN ===

if __name__ == "__main__":
    np.random.seed(0)
    rows = []

    for country in COUNTRIES:
        print("\n--- {} ---".format(country))

        # Detect available shifts by scanning for uncertainty files.
        shift = 0
        picp_1s_list, picp_2s_list, shifts_found = [], [], []

        while True:
            unc_path  = PRED_DIR / "uncertainty_{}_shift{}_{}.csv".format(MODEL, shift, country)
            pred_path = PRED_DIR / "predict_{}_shift{}_{}.csv".format(MODEL, shift, country)
            true_path = PRED_DIR / "truth_{}_shift{}_{}.csv".format(MODEL, shift, country)

            if not (unc_path.exists() and pred_path.exists() and true_path.exists()):
                break   # no more shifts

            y_pred   = _load_csv(pred_path)
            y_true   = _load_csv(true_path)
            y_uncert = _load_csv(unc_path)

            if y_pred is None or y_true is None or y_uncert is None:
                shift += 1
                continue

            # Guard: ensure shapes match before proceeding.
            if not (y_pred.shape == y_true.shape == y_uncert.shape):
                print("  [WARN] Shape mismatch for shift {}, skipping.".format(shift))
                shift += 1
                continue

            metrics = _compute_metrics(y_pred, y_true, y_uncert)
            row     = {"country": country, "shift": shift}
            row.update(metrics)

            # Load baseline ATMGNN predictions for accuracy comparison.
            base_pred_path = PRED_DIR / "predict_ATMGNN_shift{}_{}.csv".format(shift, country)
            base_true_path = PRED_DIR / "truth_ATMGNN_shift{}_{}.csv".format(shift, country)
            y_base_pred = _load_csv(base_pred_path)
            y_base_true = _load_csv(base_true_path)
            if (y_base_pred is not None and y_base_true is not None
                    and y_base_pred.shape == y_base_true.shape):
                row["atmgnn_mae"] = float(np.mean(np.abs(y_base_pred - y_base_true)))
            else:
                row["atmgnn_mae"] = float("nan")

            rows.append(row)
            shifts_found.append(shift)
            picp_1s_list.append(metrics["PICP_1sigma"])
            picp_2s_list.append(metrics["PICP_2sigma"])

            # Save calibrated uncertainty CSV.
            y_uncert_cal = y_uncert * metrics["calib_factor"]
            cal_path = PRED_DIR / "uncertainty_calibrated_{}_shift{}_{}.csv".format(MODEL, shift, country)
            np.savetxt(cal_path, y_uncert_cal, fmt="%.5f", delimiter=',')

            print("  Shift +{}: PICP_1σ {:.1%}→{:.1%}(cal)  PICP_2σ {:.1%}→{:.1%}(cal)  "
                "ρ={:.3f} (p={:.3g})  q={:.3f}  fc_mae={:.1f}".format(
                    shift,
                    metrics["PICP_1sigma"],     metrics["PICP_1sigma_cal"],
                    metrics["PICP_2sigma"],     metrics["PICP_2sigma_cal"],
                    metrics["spearman_rho"],    metrics["spearman_p"],
                    metrics["calib_factor"],    metrics["fc_mae"]))

            _plot_error_vs_uncertainty(y_pred, y_true, y_uncert, country, shift, OUT_DIR)
            shift += 1

        if not shifts_found:
            print("  No uncertainty files found — run ATMGNN_Diff_training.py first.")

    if not rows:
        print("\n[INFO] No data found. Run ATMGNN_Diff_training.py first, then re-run this script.")
        sys.exit(0)

    summary_df = pd.DataFrame(rows)

    # Save CSV summary.
    csv_path = OUT_DIR / "uncertainty_validation.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.5f")
    print("\n[OUTPUT] Summary saved to {}".format(csv_path))

    # Generate plots.
    for country in summary_df["country"].unique():
        _plot_calibration(summary_df, country, OUT_DIR)

    _plot_crps_comparison(summary_df, OUT_DIR)
    _plot_fc_vs_diff_accuracy(summary_df, OUT_DIR)

    # Print overall summary.
    print("\n=== Overall Calibration Summary ===")
    for _, row in summary_df.iterrows():
        calibrated = abs(row["PICP_1sigma_cal"] - 0.68) < 0.08 and abs(row["PICP_2sigma_cal"] - 0.95) < 0.06
        informed   = row["spearman_rho"] > 0.0 and row["spearman_p"] < 0.05
        beats_base = row["mean_CRPS_cal"] < row["baseline_CRPS"]
        verdict    = "PASS" if (calibrated and informed) else "REVIEW"
        print("  {} shift+{}: PICP_1σ {:.1%}→{:.1%}(cal)  ρ={:.3f}  q={:.3f}  CRPS_cal<baseline={}  [{}]".format(
            row["country"], int(row["shift"]),
            row["PICP_1sigma"], row["PICP_1sigma_cal"],
            row["spearman_rho"], row["calib_factor"], beats_base, verdict))
