"""Step 6 - decision-support dashboard.   Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED, SITES

st.set_page_config(page_title="Senegal Agro-Climate Monitor", layout="wide")


@st.cache_data
def load():
    ind = pd.read_csv(PROCESSED / "indicators_sites.csv")
    return (ind, pd.read_csv(PROCESSED / "trends_sites.csv"), pd.read_csv(PROCESSED / "benchmark_sites.csv"),
            pd.read_csv(PROCESSED / "qc_report.csv"), dict(np.load(PROCESSED / "spi_jas_grid.npz")))


ind, trends, bench, qc, grid = load()
years = grid["year"].tolist()

st.title("Senegal Agro-Climate Monitor")
st.caption("Rainy-season indicators for the 14 regional capitals and gridded drought monitoring, "
           "1991-2025. Sources: CHIRPS v2.0, ERA5, NASA POWER.")

site = st.sidebar.selectbox("Region", list(SITES), index=list(SITES).index("Kaolack"))
year = st.sidebar.selectbox("Season", years[::-1])
st.sidebar.markdown(f"**Agricultural profile:** {SITES[site][2]}")

tab_site, tab_map, tab_bench, tab_qc = st.tabs(["Region", "Drought map", "Product benchmark", "Data quality"])

with tab_site:
    g = ind[ind.site == site].set_index("year")
    row, normal = g.loc[year], g.loc[1991:2020]
    doy = lambda d: "n/a" if pd.isna(d) else pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(d) - 1)
    c = st.columns(5)
    c[0].metric("Onset", "n/a" if pd.isna(row.onset_doy) else doy(row.onset_doy).strftime("%d %b"),
                None if pd.isna(row.onset_doy) else f"{row.onset_doy - normal.onset_doy.mean():+.0f} days vs normal",
                delta_color="inverse")
    c[1].metric("Season length", f"{row.length_days:.0f} d" if pd.notna(row.length_days) else "n/a",
                f"{row.length_days - normal.length_days.mean():+.0f} d" if pd.notna(row.length_days) else None)
    c[2].metric("Jul-Sep rainfall", f"{row.jas_total_mm:.0f} mm",
                f"{100 * (row.jas_total_mm / normal.jas_total_mm.mean() - 1):+.0f} % vs normal")
    c[3].metric("Longest dry spell", f"{row.max_dry_spell_days:.0f} d" if pd.notna(row.max_dry_spell_days) else "n/a")
    c[4].metric("SPI-3 (JAS)", f"{row.spi3_jas:+.2f}",
                "drought" if row.spi3_jas < -1 else ("wet" if row.spi3_jas > 1 else "near normal"), delta_color="off")

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    ax[0].bar(g.index, g.jas_total_mm, color=np.where(g.spi3_jas < -1, "#B5651D", "#2E7D4F"))
    ax[0].axhline(normal.jas_total_mm.mean(), color="k", lw=0.8, ls="--", label="1991-2020 normal")
    ax[0].set_title("July-September rainfall (mm); brown = drought (SPI < -1)"); ax[0].legend()
    ax[1].plot(g.index, g.onset_doy, marker="o", ms=3, label="onset")
    ax[1].plot(g.index, g.cessation_doy, marker="o", ms=3, label="cessation")
    ax[1].set_title("Season onset and cessation (day of year)"); ax[1].legend()
    for a in ax:
        a.axvline(year, color="grey", alpha=0.4, lw=6); a.grid(alpha=0.3)
    st.pyplot(fig)
    t = trends[trends.site == site].set_index("indicator")
    st.markdown("**Trends 1991-2025** (Theil-Sen slope per decade, Mann-Kendall p-value)")
    st.dataframe(t[["theil_sen_per_decade", "kendall_tau", "p_value", "significant_5pct"]], width="stretch")

with tab_map:
    k = years.index(year)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid["spi"][k], cmap="BrBG", vmin=-2.5, vmax=2.5,
                   extent=[grid["lon"][0], grid["lon"][-1], grid["lat"][-1], grid["lat"][0]])
    for name, (la, lo, _) in SITES.items():
        ax.plot(lo, la, "k.", ms=4); ax.annotate(name, (lo, la), fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(im, label="SPI-3, July-September")
    ax.set_title(f"Rainy-season drought index {year} (CHIRPS 0.05 deg)")
    st.pyplot(fig)
    share = np.nanmean(grid["spi"][k] < -1) / np.mean(np.isfinite(grid["spi"][k]))
    st.write(f"Share of Senegal in drought (SPI < -1) in {year}: **{share:.0%}**")

with tab_bench:
    st.markdown("ERA5 and NASA POWER compared with CHIRPS, which blends satellite and rain-gauge data. "
                "This is an inter-product comparison, not a validation against national stations.")
    st.dataframe(bench[bench.site == site].set_index("product"), width="stretch")
    st.markdown("**Average over the 14 regional capitals**")
    st.dataframe(bench.groupby("product").mean(numeric_only=True).round(2), width="stretch")

with tab_qc:
    st.dataframe(qc, width="stretch", hide_index=True)
