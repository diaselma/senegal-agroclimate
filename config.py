"""Shared settings: study area, period, sites and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "figures"
for d in (RAW, PROCESSED, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

START_YEAR, END_YEAR = 1991, 2025          # 35 full years; WMO normal 1991-2020 inside
NORMAL = (1991, 2020)

# Senegal bounding box with a small margin (degrees)
LAT_MIN, LAT_MAX = 12.0, 17.0
LON_MIN, LON_MAX = -17.8, -11.2

# The 14 regional capitals of Senegal, with their main agricultural profile
SITES = {
    "Dakar":       (14.72, -17.47, "peri-urban horticulture"),
    "Thies":       (14.79, -16.93, "horticulture, groundnut"),
    "Diourbel":    (14.65, -16.23, "groundnut basin"),
    "Fatick":      (14.34, -16.41, "groundnut basin"),
    "Kaolack":     (14.15, -16.07, "groundnut basin"),
    "Kaffrine":    (14.11, -15.55, "groundnut basin"),
    "Louga":       (15.62, -16.22, "pastoral zone"),
    "Saint-Louis": (16.03, -16.42, "irrigated rice, river valley"),
    "Matam":       (15.66, -13.26, "irrigated rice, river valley"),
    "Tambacounda": (13.77, -13.67, "cotton, cereals, livestock"),
    "Kedougou":    (12.56, -12.18, "cereals"),
    "Kolda":       (12.89, -14.94, "cereals, rainfed rice"),
    "Sedhiou":     (12.71, -15.56, "rainfed rice, cereals"),
    "Ziguinchor":  (12.58, -16.27, "rainfed lowland rice"),
}
