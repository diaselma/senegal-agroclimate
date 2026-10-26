"""Step 2 - quality control of the daily site data.

Rules (a failed value is set to missing, never silently corrected):
  - missing values and missing dates in the expected daily calendar
  - precipitation < 0 mm or > 250 mm/day (above the Senegal daily records)
  - tmax / tmin outside 5-50 degC, or tmin > tmax
  - duplicated (date, site, source) rows
Outputs: data/processed/daily_clean.csv, data/processed/qc_report.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import END_YEAR, PROCESSED, START_YEAR

df = pd.read_csv(PROCESSED / "daily_sites.csv", parse_dates=["date"])
df = df.replace([-999.0, -9999.0], np.nan)                 # fill values (NASA POWER, CHIRPS over sea)
expected = len(pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))

rows = []
for (site, source), g in df.groupby(["site", "source"]):
    dup = g.duplicated("date").sum()
    g = g.drop_duplicates("date")
    bad_p = (g.precip_mm < 0) | (g.precip_mm > 250)
    bad_t = (g.tmax_c < 5) | (g.tmax_c > 50) | (g.tmin_c < 5) | (g.tmin_c > 50) | (g.tmin_c > g.tmax_c)
    rows.append(dict(site=site, source=source, days=len(g), missing_dates=expected - len(g),
                     duplicates=dup, precip_missing=int(g.precip_mm.isna().sum()),
                     precip_out_of_range=int(bad_p.sum()),
                     temp_flagged=int(bad_t.sum()) if source != "CHIRPS" else np.nan))
    df.loc[g.index[bad_p], "precip_mm"] = np.nan
    df.loc[g.index[bad_t], ["tmax_c", "tmin_c"]] = np.nan

report = pd.DataFrame(rows)
report.to_csv(PROCESSED / "qc_report.csv", index=False)
df.drop_duplicates(["date", "site", "source"]).to_csv(PROCESSED / "daily_clean.csv", index=False)
print(report.groupby("source")[["missing_dates", "duplicates", "precip_missing",
                                "precip_out_of_range", "temp_flagged"]].sum())
