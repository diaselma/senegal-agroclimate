"""Step 1b - daily weather at the 14 regional capitals from three independent products.

  CHIRPS v2.0  satellite + gauge rainfall      ClimateSERV API (NASA/USAID SERVIR)
  ERA5         ECMWF reanalysis                Open-Meteo archive API
  NASA POWER   MERRA-2 based, bias-corrected   NASA POWER daily API

Output: data/processed/daily_sites.csv  (date, site, source, precip_mm, tmax_c, tmin_c)
Raw API responses are cached in data/raw/ so the script can be resumed.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import END_YEAR, PROCESSED, RAW, SITES, START_YEAR

START, END = f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"


def cached(name, fetch):
    path = RAW / name
    if not path.exists():
        path.write_text(json.dumps(fetch()))
    return json.loads(path.read_text())


def chirps_year(lat, lon, year):
    base = "https://climateserv.servirglobal.net/api/"
    geom = json.dumps({"type": "Point", "coordinates": [lon, lat]})
    params = dict(datatype=0, intervaltype=0, operationtype=5, geometry=geom,
                  begintime=f"01/01/{year}", endtime=f"12/31/{year}")
    for attempt in range(6):                    # the API answers with an empty page when busy
        try:
            job = requests.get(base + "submitDataRequest/", params=params, timeout=120).json()[0]
            while requests.get(base + "getDataRequestProgress/", params={"id": job}, timeout=60).json()[0] < 100:
                time.sleep(2)
            return requests.get(base + "getDataFromRequest/", params={"id": job}, timeout=300).json()["data"]
        except (ValueError, requests.RequestException):
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"CHIRPS {year} at {lat},{lon} failed")


def chirps(lat, lon):
    # one request per year, in parallel: the API is fast on short periods, very slow on 35 years
    with ThreadPoolExecutor(4) as pool:
        years = pool.map(lambda y: chirps_year(lat, lon, y), range(START_YEAR, END_YEAR + 1))
        return {"data": [d for year in years for d in year]}


def era5(lat, lon):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=300, params=dict(
        latitude=lat, longitude=lon, start_date=START, end_date=END, timezone="GMT",
        daily="precipitation_sum,temperature_2m_max,temperature_2m_min", models="era5"))
    r.raise_for_status()
    return r.json()


def power(lat, lon):
    r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", timeout=300, params=dict(
        parameters="PRECTOTCORR,T2M_MAX,T2M_MIN", community="AG", latitude=lat, longitude=lon,
        start=START.replace("-", ""), end=END.replace("-", ""), format="JSON"))
    r.raise_for_status()
    return r.json()


def frames(site, lat, lon):
    c = cached(f"chirps_{site}.json", lambda: chirps(lat, lon))
    yield pd.DataFrame({"date": pd.to_datetime([d["date"] for d in c["data"]], format="%m/%d/%Y"),
                        "precip_mm": [d["value"]["avg"] for d in c["data"]]}).assign(source="CHIRPS")
    e = cached(f"era5_{site}.json", lambda: era5(lat, lon))["daily"]
    yield pd.DataFrame({"date": pd.to_datetime(e["time"]), "precip_mm": e["precipitation_sum"],
                        "tmax_c": e["temperature_2m_max"], "tmin_c": e["temperature_2m_min"]}).assign(source="ERA5")
    p = cached(f"power_{site}.json", lambda: power(lat, lon))["properties"]["parameter"]
    yield pd.DataFrame({"date": pd.to_datetime(list(p["PRECTOTCORR"]), format="%Y%m%d"),
                        "precip_mm": list(p["PRECTOTCORR"].values()), "tmax_c": list(p["T2M_MAX"].values()),
                        "tmin_c": list(p["T2M_MIN"].values())}).assign(source="NASA_POWER")


if __name__ == "__main__":
    parts = []
    for site, (lat, lon, _) in SITES.items():
        for df in frames(site, lat, lon):
            parts.append(df.assign(site=site))
        print("done", site, flush=True)
    daily = pd.concat(parts)[["date", "site", "source", "precip_mm", "tmax_c", "tmin_c"]]
    daily.to_csv(PROCESSED / "daily_sites.csv", index=False)
    print(daily.groupby("source").size())
