"""Step 4 - agro-climatic indicators per site and year, from CHIRPS daily rainfall.

  Onset        Sivakumar (1988), standard in the Sahel: first day after 1 May with
               >= 20 mm over 3 consecutive days, not followed by a dry spell of more
               than 7 days within the next 30 days.
  Cessation    simple water balance (100 mm capacity, 5 mm/day evapotranspiration):
               first day after 1 September (and after onset) when the soil store is empty.
  Length       cessation - onset (days)
  Dry spell    longest run of days < 1 mm between onset and cessation
  SPI-3        Standardised Precipitation Index of July-September, gamma distribution
               fitted on the 1991-2020 normal (McKee et al. 1993); < -1 = drought.
  Trends       Mann-Kendall test (Kendall's tau against year) and Theil-Sen slope.
Outputs: data/processed/indicators_sites.csv, data/processed/trends_sites.csv,
         data/processed/spi_jas_grid.npz, figures/indicators_*.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES, NORMAL, PROCESSED, SITES


def onset(r):                                   # r: daily rainfall from 1 Jan, indexed by day of year - 1
    for d in range(120, 290):                   # 1 May .. mid-October
        if r[d:d + 3].sum() >= 20:
            dry = (r[d + 1:d + 31] < 1).astype(int)
            longest = max((len(x) for x in "".join(map(str, dry)).split("0")), default=0)
            if longest <= 7:
                return d + 1
    return np.nan


def cessation(r, onset_doy):
    sw = 0.0
    for d, rain in enumerate(r):
        sw = min(100.0, max(0.0, sw + rain - 5.0))
        if d >= max(243, onset_doy) and sw == 0:   # from 1 September, and never before onset
            return d + 1
    return np.nan


def longest_dry(r):
    runs = "".join("1" if x < 1 else "0" for x in r).split("0")
    return max(map(len, runs)) if len(r) else np.nan


def spi(values, normal_mask):
    """Gamma-based SPI with the zero-rain probability handled explicitly."""
    ref = values[normal_mask]
    q = (ref == 0).mean()
    a, _, scale = stats.gamma.fit(ref[ref > 0], floc=0)
    cdf = q + (1 - q) * stats.gamma.cdf(values, a, scale=scale)
    return stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))


df = pd.read_csv(PROCESSED / "daily_clean.csv", parse_dates=["date"])
ch = df[df.source == "CHIRPS"].set_index("date")
sites = [s for s in SITES if s in set(df.site)]

rows = []
for site in sites:
    s = ch[ch.site == site].precip_mm.fillna(0)
    for year, r in s.groupby(s.index.year):
        r = r.to_numpy()
        on = onset(r)
        end = cessation(r, on) if np.isfinite(on) else np.nan
        season = r[int(on) - 1:int(end)] if np.isfinite(on) and np.isfinite(end) and end > on else np.array([])
        rows.append(dict(site=site, year=year, onset_doy=on, cessation_doy=end,
                         length_days=end - on if len(season) else np.nan,
                         max_dry_spell_days=longest_dry(season), season_total_mm=season.sum() if len(season) else np.nan,
                         jas_total_mm=r[181:273].sum()))
ind = pd.DataFrame(rows)
ind["spi3_jas"] = np.nan
for site, g in ind.groupby("site"):
    ind.loc[g.index, "spi3_jas"] = spi(g.jas_total_mm.to_numpy(), g.year.between(*NORMAL).to_numpy())
ind.round(2).to_csv(PROCESSED / "indicators_sites.csv", index=False)

trend_rows = []
for site, g in ind.groupby("site"):
    for col in ["onset_doy", "length_days", "jas_total_mm", "max_dry_spell_days"]:
        g2 = g[["year", col]].dropna()
        tau, pval = stats.kendalltau(g2.year, g2[col])
        trend_rows.append(dict(site=site, indicator=col, theil_sen_per_decade=10 * stats.theilslopes(g2[col], g2.year)[0],
                               kendall_tau=tau, p_value=pval, significant_5pct=pval < 0.05))
trends = pd.DataFrame(trend_rows).round(3)
trends.to_csv(PROCESSED / "trends_sites.csv", index=False)

# gridded SPI-3 (JAS) from the CHIRPS monthly cube, for maps
cube = np.load(PROCESSED / "chirps_monthly_senegal.npz")
years = np.array([int(t[:4]) for t in cube["time"]])
months = np.array([int(t[5:]) for t in cube["time"]])
jas_grid = np.stack([cube["rain"][(years == y) & np.isin(months, [7, 8, 9])].sum(0) for y in np.unique(years)])
yrs = np.unique(years)
spi_grid = np.full(jas_grid.shape, np.nan, dtype="float32")
nmask = (yrs >= NORMAL[0]) & (yrs <= NORMAL[1])
land = np.isfinite(cube["rain"]).all(0)
for i, j in zip(*np.where(land)):
    spi_grid[:, i, j] = spi(jas_grid[:, i, j], nmask)
np.savez_compressed(PROCESSED / "spi_jas_grid.npz", spi=spi_grid, jas=jas_grid, year=yrs,
                    lat=cube["lat"], lon=cube["lon"])

print(ind.groupby("site")[["onset_doy", "length_days", "jas_total_mm", "max_dry_spell_days"]].mean().round(0))
print(trends[trends.significant_5pct])

# figures: SPI heat map of sites x years, and the national drought maps of selected years
order = sorted(sites, key=lambda k: -SITES[k][0])          # north to south
h = ind.pivot(index="site", columns="year", values="spi3_jas").loc[order]
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(h, cmap="BrBG", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(0, h.shape[1], 2), h.columns[::2], rotation=90); ax.set_yticks(range(len(order)), order)
fig.colorbar(im, label="SPI-3 (JAS)  < -1 drought, > 1 wet")
ax.set_title("Rainy-season drought index (SPI-3, July-September) by region, CHIRPS 1991-2025, north to south")
fig.tight_layout(); fig.savefig(FIGURES / "indicators_spi_heatmap.png", dpi=150)

worst = yrs[np.argsort(np.nanmean(spi_grid, axis=(1, 2)))[:3]]
show = list(worst) + [yrs[-1]]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, y in zip(axes, show):
    k = list(yrs).index(y)
    im = ax.imshow(spi_grid[k], cmap="BrBG", vmin=-2.5, vmax=2.5,
                   extent=[cube["lon"][0], cube["lon"][-1], cube["lat"][-1], cube["lat"][0]])
    for name, (la, lo, _) in SITES.items():
        ax.plot(lo, la, "k.", ms=3)
    ax.set_title(f"{y}" + ("  (most recent)" if y == yrs[-1] else "  (among driest)"))
fig.colorbar(im, ax=axes, shrink=0.8, label="SPI-3 JAS")
fig.suptitle("Gridded drought index over Senegal (CHIRPS, 0.05 deg)")
fig.savefig(FIGURES / "indicators_spi_maps.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 4.5))
for site in ["Saint-Louis", "Kaolack", "Tambacounda", "Ziguinchor"]:
    g = ind[ind.site == site]
    ax.plot(g.year, g.onset_doy, marker="o", ms=3, label=site)
ax.set_ylabel("Onset (day of year)"); ax.grid(alpha=0.3); ax.legend()
ax.set_title("Rainy-season onset (Sivakumar criterion), north to south")
fig.tight_layout(); fig.savefig(FIGURES / "indicators_onset.png", dpi=150)
