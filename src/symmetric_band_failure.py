"""
    Simple script to check whether a naive symmetric ±1.96σ prediction interval provides balanced tail coverage for England (shift 0) as an example.
"""

import sys
import pandas as pd
import numpy as np

truth = pd.read_csv("predictions/truth_ATMGNN_Diff_shift0_EN.csv", header=None)
pred  = pd.read_csv("predictions/predict_ATMGNN_Diff_shift0_EN.csv", header=None)

t = truth.values.flatten().astype(float)
p = pred.values.flatten().astype(float)
n = min(len(t), len(p))
t, p = t[:n], p[:n]

residuals = t - p
sigma = np.std(residuals)

upper_sym = p + 1.96 * sigma
lower_sym = p - 1.96 * sigma

print("--- SYMMETRIC BAND FAILURE (England, shift=0) ---")
print("Upper bound exceeded:", round((t > upper_sym).mean()*100, 1), "%  (nominal = 2.5%)")
print("Lower bound exceeded:", round((t < lower_sym).mean()*100, 1), "%  (nominal = 2.5%)")

q975 = np.percentile(residuals, 97.5)
q025 = np.percentile(residuals,  2.5)
q50  = np.percentile(residuals, 50)
print("2.5th  pct of residuals:", round(float(q025), 3))
print("50th   pct of residuals:", round(float(q50), 3))
print("97.5th pct of residuals:", round(float(q975), 3))
print("Asymmetry ratio        :", round((q975 - q50) / abs(q50 - q025), 3))
sys.stdout.flush()