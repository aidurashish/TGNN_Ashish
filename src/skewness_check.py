"""
    Computes descriptive statistics and skewness of actual COVID-19 case counts for England (shift 0) as an example.
"""

import sys
import pandas as pd
import numpy as np
from scipy.stats import skew

truth = pd.read_csv("predictions/truth_ATMGNN_Diff_shift0_EN.csv", header=None)
vals = truth.values.flatten().astype(float)
vals = vals[vals > 0]

print("--- DISTRIBUTION OF ACTUAL CASE COUNTS (England, shift=0) ---")
print("N observations       :", len(vals))
print("Mean                 :", round(vals.mean(), 2))
print("Median               :", round(float(np.median(vals)), 2))
print("Mean/Median ratio    :", round(vals.mean() / np.median(vals), 3))
print("Skewness coefficient :", round(float(skew(vals)), 3))
p95 = np.percentile(vals, 95)
top5_share = vals[vals >= p95].sum() / vals.sum() * 100
print("Top 5% share of total:", round(float(top5_share), 1), "%")
print("Max value            :", vals.max())
sys.stdout.flush()