"""Supplementary File S1
Reproducible calculations for the secondary-data scenario model reported in
'Thermally Induced Larval Mortality in Hybrid Clarias Catfish'.

This script reproduces: (i) deterministic hatchery outputs in Table 2,
(ii) the Q10 multiplier reported in Results, and (iii) the illustrative
seasonal/multiyear climate envelope used for Figure 3.

The script is intentionally transparent: long-term economic/genomic trajectories
are scenario analyses and are not calibrated forecasts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20260929
N_MONTE_CARLO = 10000
rng = np.random.default_rng(SEED)

# Source-study biological inputs (Rey et al. [9])
BIO = {
    29: {"fertilization": 0.677, "hatching": 0.434, "survival72": 0.449},
    32: {"fertilization": 0.590, "hatching": 0.266, "survival72": 0.000},
}

# Deterministic hatchery/economic scenario inputs
N0 = 5_000_000
FRY_PRICE = 1.50
BASELINE_COST = 100_000
COOLING_COST = 80_000


def hatchery_output(temp_c: int, cooling: bool = False):
    b = BIO[temp_c]
    fertilized = N0 * b["fertilization"]
    hatched = fertilized * b["hatching"]
    viable = hatched * b["survival72"]
    revenue = viable * FRY_PRICE
    total_cost = BASELINE_COST + (COOLING_COST if cooling else 0)
    net = revenue - total_cost
    return fertilized, hatched, viable, revenue, total_cost, net


scenario_a = hatchery_output(32, cooling=False)
scenario_b = hatchery_output(29, cooling=True)
print("Scenario A (32C):", scenario_a)
print("Scenario B (29C + cooling):", scenario_b)
print("Net-outcome contrast (B-A):", scenario_b[-1] - scenario_a[-1])

# Q10 sensitivity for a 3C increase (29 -> 32C)
for q10 in (2.0, 3.0):
    print(f"Q10={q10:.1f}: multiplier={q10 ** (3/10):.3f}")

# Climate/scenario parameters used in Figure 3
TMEAN = 28.5
AMP0 = 3.5
DPEAK = 115
ALPHA = 0.08          # C/year; within Table 1 range 0.05-0.10
BETA = 0.03           # C/year seasonal-amplitude change
E_SD = 0.30           # C
TOPT = 29.0
SIGMA = 1.0
SMAX = 44.9           # percent, source-study 72-h survival at 29C
TCRIT0 = 30.75
DELTA_G = 0.15        # C/year; sensitivity range 0.10-0.20

# Panel A: representative seasonal cycle
day = np.arange(1, 366)
Tday = TMEAN + AMP0 * np.cos(2*np.pi*(day-DPEAK)/365.0)
Sday = SMAX * np.exp(-0.5*((Tday-TOPT)/SIGMA)**2)

# Panel B: Monte Carlo envelope for annual anomalies E(y)
years = np.arange(2026, 2037)
nd = len(years) * 365
time_year = 2026 + np.arange(nd)/365.0
yindex = np.minimum((np.arange(nd)//365), len(years)-1)
doy = (np.arange(nd) % 365) + 1
base = (TMEAN + ALPHA*yindex + (AMP0 + BETA*yindex) *
        np.cos(2*np.pi*(doy-DPEAK)/365.0))
# One annual anomaly per simulated year, shared by all days in that year.
anom = rng.normal(0.0, E_SD, size=(N_MONTE_CARLO, len(years)))
# Quantiles can be obtained analytically from the annual anomalies after adding base.
q025_year = np.quantile(anom, 0.025, axis=0)
q975_year = np.quantile(anom, 0.975, axis=0)
lo = base + q025_year[yindex]
hi = base + q975_year[yindex]
mean = base
Tcrit_genomic = TCRIT0 + DELTA_G * (time_year-2026)

# Sensitivity summaries reported in text
print("Delta_g sensitivity: Tcrit after 10 years")
for dg in (0.10, 0.15, 0.20):
    print(dg, TCRIT0 + 10*dg)
print("E(y) SD sensitivity: approximate 95% annual-anomaly half-width")
for sd in (0.20, 0.30, 0.50):
    print(sd, 1.96*sd)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
ax = axes[0]
ax2 = ax.twinx()
ax.plot(day, Tday, label="T(d): Temperature")
ax2.plot(day, Sday, linestyle="--", label="S(T): 72-h survival")
ax.axhline(TCRIT0, linestyle=":", label="Threshold (Tcrit)")
ax.set_xlim(1, 365)
ax.set_xlabel("Day of the year")
ax.set_ylabel("Water temperature (°C)")
ax2.set_ylabel("72-h larval survival S(T) (%)")
ax.set_title("a | Annual vulnerability cycle", loc="left", fontweight="bold")
handles = ax.get_lines() + ax2.get_lines()
ax.legend(handles, [h.get_label() for h in handles], loc="lower center")

ax = axes[1]
ax.plot(time_year, mean, label="Mean T(d,y)")
ax.fill_between(time_year, lo, hi, alpha=0.15, label="95% simulation interval")
ax.axhline(TCRIT0, linestyle="--", label="Static Tcrit")
ax.plot(time_year, Tcrit_genomic, label="Hypothetical Tcrit(y), Δg=0.15°C/year")
ax.set_xlim(2026, 2036)
ax.set_xlabel("Projection year")
ax.set_ylabel("Water temperature (°C)")
ax.set_title("b | 10-year stochastic climate scenario", loc="left", fontweight="bold")
ax.legend(loc="upper left")

fig.savefig('figure3_reproducible.png', dpi=220)
