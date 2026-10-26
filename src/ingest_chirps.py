"""Step 1a - CHIRPS v2.0 monthly rainfall over Senegal.

Each African monthly GeoTIFF (~4.5 MB compressed, 1600 x 1500 cells at 0.05 deg) is
downloaded into memory, cropped to the Senegal box and discarded: nothing but the
Senegal cube is ever written to disk (~10 MB for 35 years).
Output: data/processed/chirps_monthly_senegal.npz  (rain[t, lat, lon] in mm/month)
"""
import gzip
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import END_YEAR, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PROCESSED, START_YEAR

URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs/chirps-v2.0.{y}.{m:02d}.tif.gz"
RES, WEST, NORTH = 0.05, -20.0, 40.0           # grid of the Africa product
rows = slice(int((NORTH - LAT_MAX) / RES), int((NORTH - LAT_MIN) / RES))
cols = slice(int((LON_MIN - WEST) / RES), int((LON_MAX - WEST) / RES))
lats = NORTH - RES * (np.arange(rows.start, rows.stop) + 0.5)
lons = WEST + RES * (np.arange(cols.start, cols.stop) + 0.5)


def fetch(ym):
    y, m = ym
    for attempt in range(4):
        try:
            r = requests.get(URL.format(y=y, m=m), timeout=120)
            r.raise_for_status()
            grid = tifffile.imread(io.BytesIO(gzip.decompress(r.content)))[rows, cols]
            return np.where(grid < 0, np.nan, grid).astype("float32")
        except Exception as e:                  # network hiccups: retry
            err = e
    raise RuntimeError(f"{y}-{m:02d}: {err}")


if __name__ == "__main__":
    months = [(y, m) for y in range(START_YEAR, END_YEAR + 1) for m in range(1, 13)]
    with ThreadPoolExecutor(6) as pool:
        cube = []
        for i, grid in enumerate(pool.map(fetch, months), 1):
            cube.append(grid)
            if i % 60 == 0:
                print(f"{i}/{len(months)} months", flush=True)
    rain = np.stack(cube)
    time = np.array([f"{y}-{m:02d}" for y, m in months])
    np.savez_compressed(PROCESSED / "chirps_monthly_senegal.npz", rain=rain, time=time, lat=lats, lon=lons)
    print("saved", rain.shape, f"missing cells: {np.isnan(rain).mean():.1%}")
