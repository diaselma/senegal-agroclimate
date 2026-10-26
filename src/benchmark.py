"""Step 3 - benchmarking ERA5 and NASA POWER rainfall against CHIRPS.

CHIRPS is used as the reference because it blends satellite estimates with rain-gauge
observations; it is not ground truth. Without access to national station data, this
is an inter-product comparison, which is what the metrics below should be read as.

  Daily      : rain-day detection at 1 mm - POD, FAR, CSI (May-Oct)
  Monthly    : bias (%), RMSE (mm), Pearson r on rainy-season months (May-Oct)
  Seasonal   : bias (%) and correlation of July-September totals across 35 years
               (the interannual signal that drought monitoring depends on)
Outputs: data/processed/benchmark_sites.csv, figures/benchmark_*.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES, PROCESSED, SITES

df = pd.read_csv(PROCESSED / "daily_clean.csv", parse_dates=["date"])
sites = [s for s in SITES if s in set(df.site)]
p = df.pivot_table(index=["site", "date"], columns="source", values="precip_mm").dropna()
p = p[p.index.get_level_values("date").month.isin(range(5, 11))]
monthly = p.groupby([p.index.get_level_values("site"), p.index.get_level_values("date").to_period("M")]).sum()
jas = monthly[monthly.index.get_level_values(1).month.isin([7, 8, 9])]
jas = jas.groupby([jas.index.get_level_values(0), jas.index.get_level_values(1).year]).sum()

rows = []
for site in sites:
    d, m, s = p.loc[site], monthly.loc[site], jas.loc[site]
    ref_wet = d.CHIRPS >= 1
    for prod in ["ERA5", "NASA_POWER"]:
        wet = d[prod] >= 1
        hits, misses, false = (wet & ref_wet).sum(), (~wet & ref_wet).sum(), (wet & ~ref_wet).sum()
        rows.append(dict(
            site=site, product=prod,
            pod=hits / (hits + misses), far=false / (hits + false), csi=hits / (hits + misses + false),
            monthly_bias_pct=100 * (m[prod].sum() / m.CHIRPS.sum() - 1),
            monthly_rmse_mm=np.sqrt(((m[prod] - m.CHIRPS) ** 2).mean()),
            monthly_r=m[prod].corr(m.CHIRPS),
            jas_bias_pct=100 * (s[prod].mean() / s.CHIRPS.mean() - 1),
            jas_interannual_r=s[prod].corr(s.CHIRPS)))

bench = pd.DataFrame(rows).round(3)
bench.to_csv(PROCESSED / "benchmark_sites.csv", index=False)
print(bench.groupby("product").mean(numeric_only=True).round(2).T)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
order = sites[::-1]                                  # barh draws bottom-up: keeps config order top-down
for ax, col, title in zip(axes, ["jas_bias_pct", "jas_interannual_r", "csi"],
                          ["JAS total bias vs CHIRPS (%)", "Interannual correlation of JAS totals",
                           "Rain-day detection (CSI, 1 mm)"]):
    t = bench.pivot(index="site", columns="product", values=col).loc[order]
    t.plot.barh(ax=ax, color=["#2E7D4F", "#C98A1B"], width=0.75, legend=ax is axes[0])
    ax.set_title(title); ax.set_ylabel(""); ax.grid(axis="x", alpha=0.3)
    if col == "jas_bias_pct":
        ax.axvline(0, color="k", lw=0.8)
fig.suptitle("Rainfall products over Senegal's 14 regional capitals, 1991-2025 (reference: CHIRPS)")
fig.tight_layout()
fig.savefig(FIGURES / "benchmark_products.png", dpi=150)
