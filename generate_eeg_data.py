"""
generate_eeg_data.py — Run this FIRST to create eeg_data.csv
Usage: python generate_eeg_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N_NORMAL, N_SEIZURE = 2000, 500

def make_normal(n):
    return pd.DataFrame({
        "delta_power":    np.random.normal(80, 12, n).clip(10, 120),
        "theta_power":    np.random.normal(60, 10, n).clip(5, 100),
        "alpha_power":    np.random.normal(75, 11, n).clip(10, 110),
        "beta_power":     np.random.normal(42,  9, n).clip(5, 80),
        "gamma_power":    np.random.normal(28,  7, n).clip(2, 60),
        "line_length":    np.random.normal(0.45, 0.08, n).clip(0.1, 0.9),
        "sample_entropy": np.random.normal(1.8, 0.3,  n).clip(0.5, 3.5),
        "kurtosis":       np.random.normal(3.1, 0.5,  n).clip(1.0, 6.0),
        "skewness":       np.random.normal(0.1, 0.3,  n).clip(-1.5, 1.5),
        "variance":       np.random.normal(120, 25,   n).clip(20, 300),
        "zero_crossings": np.random.normal(55,  10,   n).clip(10, 120),
        "peak_frequency": np.random.normal(10,  2,    n).clip(2, 30),
        "label": "Normal",
    })

def make_seizure(n):
    return pd.DataFrame({
        "delta_power":    np.random.normal(35, 10, n).clip(5, 80),
        "theta_power":    np.random.normal(28,  9, n).clip(3, 70),
        "alpha_power":    np.random.normal(22,  8, n).clip(3, 60),
        "beta_power":     np.random.normal(95, 15, n).clip(30, 160),
        "gamma_power":    np.random.normal(88, 13, n).clip(30, 140),
        "line_length":    np.random.normal(0.82, 0.07, n).clip(0.3, 1.2),
        "sample_entropy": np.random.normal(0.7, 0.2,  n).clip(0.1, 1.5),
        "kurtosis":       np.random.normal(6.8, 1.1,  n).clip(2.0, 12.0),
        "skewness":       np.random.normal(1.2, 0.5,  n).clip(-0.5, 3.0),
        "variance":       np.random.normal(480, 80,   n).clip(100, 800),
        "zero_crossings": np.random.normal(120, 20,   n).clip(40, 200),
        "peak_frequency": np.random.normal(25,  4,    n).clip(8, 45),
        "label": "Seizure",
    })

df = pd.concat([make_normal(N_NORMAL), make_seizure(N_SEIZURE)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("eeg_data.csv", index=False)
print(f"✅  eeg_data.csv created — {len(df)} rows ({N_NORMAL} Normal, {N_SEIZURE} Seizure)")
