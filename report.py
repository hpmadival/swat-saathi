"""
Two things for your thesis or paper:
  - a ready-to-paste methods paragraph (with references) describing exactly the data
    and settings used for this download
  - a short PDF data report: maps of elevation, land use and soil, land use and soil
    tables, the monthly climate, and the methods paragraph
"""

import datetime
import io
import textwrap

import numpy as np

import swat_maps as M
import swat_weather as S

REFERENCES = {
    "dem": "European Space Agency (2021). Copernicus Global Digital Elevation Model (GLO-30 / GLO-90). "
           "Distributed by OpenTopography and the AWS Registry of Open Data.",
    "lu": "Zanaga, D., Van De Kerchove, R., Daems, D., et al. (2022). ESA WorldCover 10 m 2021 v200. "
          "Zenodo. https://doi.org/10.5281/zenodo.7254221",
    "lu2020": "Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., et al. (2021). ESA WorldCover 10 m 2020 "
              "v100. Zenodo. https://doi.org/10.5281/zenodo.5571936",
    "soil": "Poggio, L., de Sousa, L. M., Batjes, N. H., et al. (2021). SoilGrids 2.0: producing soil "
            "information for the globe with quantified spatial uncertainty. SOIL, 7, 217-240. "
            "https://doi.org/10.5194/soil-7-217-2021",
    "sr": "Saxton, K. E., and Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic "
          "matter for hydrologic solutions. Soil Science Society of America Journal, 70, 1569-1578.",
    "williams": "Williams, J. R. (1995). The EPIC model. In V. P. Singh (Ed.), Computer Models of Watershed "
                "Hydrology (pp. 909-1000). Water Resources Publications.",
    "nrcs": "USDA Natural Resources Conservation Service (2009). National Engineering Handbook, Part 630 "
            "Hydrology, Chapter 7: Hydrologic Soil Groups.",
    "rain": "Pai, D. S., Sridhar, L., Rajeevan, M., et al. (2014). Development of a new high spatial resolution "
            "(0.25 x 0.25 degree) long period (1901-2010) daily gridded rainfall data set over India and its "
            "comparison with existing data sets over the region. Mausam, 65(1), 1-18.",
    "temp": "Srivastava, A. K., Rajeevan, M., and Kshirsagar, S. R. (2009). Development of a high resolution "
            "daily gridded temperature data set (1969-2005) for the Indian region. Atmospheric Science Letters, "
            "10, 249-254.",
    "power": "NASA Langley Research Center (2024). Prediction Of Worldwide Energy Resources (POWER) Project. "
             "https://power.larc.nasa.gov",
    "osm": "OpenStreetMap contributors (2024). OpenStreetMap. https://www.openstreetmap.org",
    "flw": "Eilander, D., van Verseveld, W., Yamazaki, D., Weerts, A., Winsemius, H. C., and Ward, P. J. "
           "(2021). A hydrography upscaling method for scale-invariant parametrization of distributed "
           "hydrological models. Hydrology and Earth System Sciences, 25, 5287-5313 (pyflwdir).",
    "app": "Madival, H. P. (2026). SWAT Bharat: SWAT input data preparation for India. "
           "Built on the IMD Data Puller, https://doi.org/10.5281/zenodo.24352511",
}


def methods_text(info, ms=None, ws=None):
    """A methods paragraph and its reference list, from what was actually made.

    info: dict with area_km2, epsg, places (text), margin (text), traced (text or None),
          rivers (number of lines or None), lu_year, res
    """
    zone = info["epsg"] - 32600
    used = ["app"]
    p = []
    own = info.get("own_km2", info["area_km2"])
    if info["area_km2"] > own + 0.5 and "margin" in info["margin"]:
        area = (f"The study area covered {own:,.0f} km² ({info['places']}). All input data were prepared for a "
                f"{info['area_km2']:,.0f} km² download area that adds {info['margin']}, so that watershed "
                f"delineation has terrain beyond the boundary and weather grid cells near the edge are included.")
    else:
        area = f"The study area covered {own:,.0f} km² ({info['places']}); data were prepared for this area."
    if info.get("traced"):
        tres = info.get("traced_res", 90)
        how = ("the Copernicus 90 m DEM to find its extent and then the 30 m DEM for the final boundary"
               if tres == 30 else "the Copernicus 90 m DEM")
        area += (f" The catchment boundary was traced from the outlet with D8 flow directions from {how}, "
                 f"using pyflwdir (Eilander et al., 2021); {info['traced']}")
        used.append("flw")
    p.append(area)
    if ms is not None:
        res = info["res"]
        if ms.dem is not None:
            p.append(f"Elevation was taken from the Copernicus GLO-{30 if res == 30 else 90} digital elevation "
                     f"model (ESA, 2021) and projected to WGS 84 / UTM zone {zone}N (EPSG:{info['epsg']}) at "
                     f"{res} m resolution with bilinear resampling.")
            used.append("dem")
        if ms.landuse_lookup is not None:
            yr = info.get("lu_year", 2021)
            n_codes = ms.landuse_lookup["SWAT code"].nunique()
            p.append(f"Land use and land cover came from ESA WorldCover {yr} at 10 m (Zanaga et al., "
                     f"{2022 if yr == 2021 else 2021}), aggregated to the {res} m grid by majority and "
                     f"reclassified into {n_codes} SWAT land use classes "
                     f"({', '.join(sorted(ms.landuse_lookup['SWAT code'].unique()))}).")
            used.append("lu" if yr == 2021 else "lu2020")
        if ms.soil_units is not None:
            p.append(f"Soil properties (sand, silt, clay, bulk density, organic carbon, coarse fragments and pH) "
                     f"were taken from SoilGrids 2.0 at 250 m (Poggio et al., 2021). In total, {len(ms.soil_units)} "
                     f"soil types were defined by the USDA texture class of the topsoil (0-30 cm) and subsoil "
                     f"(30-100 cm), each described with four layers down to 1 m. Available water capacity and "
                     f"saturated hydraulic conductivity were estimated with the pedotransfer functions of Saxton "
                     f"and Rawls (2006), soil erodibility (USLE K) following Williams (1995), and hydrologic soil "
                     f"groups from the saturated conductivity of the least permeable layer (NRCS, 2009).")
            used += ["soil", "sr", "williams", "nrcs"]
    if info.get("rivers"):
        p.append(f"River lines from OpenStreetMap (OpenStreetMap contributors, 2024; {info['rivers']} segments) "
                 f"were burned into the DEM before watershed delineation.")
        used.append("osm")
    if ws is not None:
        n_p, n_t = len(ws.stations.get("pcp", [])), len(ws.stations.get("tmp", []))
        how = ("each grid cell over the study area was used as a virtual weather station"
               if ws.mode == "grid" else "area-weighted averages over each subbasin were used as virtual stations")
        w = (f"Daily rainfall for {ws.start_date:%d %B %Y} to {ws.end_date:%d %B %Y} came from the India "
             f"Meteorological Department (IMD) 0.25° gridded dataset (Pai et al., 2014), and daily maximum and "
             f"minimum temperature from the IMD 1° gridded dataset (Srivastava et al., 2009); {how} "
             f"({n_p} rainfall and {n_t} temperature stations).")
        used += ["rain", "temp"]
        if ws.stations.get("hmd"):
            w += (" Daily relative humidity, solar radiation and wind speed at 2 m came from NASA POWER "
                  "(NASA Langley Research Center, 2024).")
            used.append("power")
        hh = ws.settings.get("Half-hour rain factor (largest daily rain x factor)", S.HALF_HOUR_FACTOR)
        w += (f" Weather generator parameters were calculated from the same daily series; the maximum "
              f"half-hour rainfall was estimated as {hh} times the maximum daily rainfall of each month, "
              f"following the IMD empirical reduction formula P(t) = P(24 h) x (t/24)^(1/3).")
        p.append(w)
    p.append("All inputs were prepared with SWAT Bharat (Madival, 2026).")
    refs = sorted({REFERENCES[k] for k in used})
    return " ".join(p), refs


def methods_file(info, ms=None, ws=None):
    text, refs = methods_text(info, ms, ws)
    return ("METHODS PARAGRAPH (edit freely)\r\n" + "=" * 31 + "\r\n\r\n" + "\r\n".join(textwrap.wrap(text, 95))
            + "\r\n\r\nREFERENCES\r\n" + "=" * 10 + "\r\n\r\n" + "\r\n\r\n".join(refs) + "\r\n")


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def _to_utm(geoms, epsg):
    import geopandas as gpd
    return list(gpd.GeoSeries(geoms, crs=4326).to_crs(epsg)) if geoms else []


def _draw_outline(ax, geom, **kw):
    for g in getattr(geom, "geoms", [geom]):
        if g.geom_type == "Polygon":
            x, y = g.exterior.xy
            ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, **kw)
        elif g.geom_type == "Point":
            ax.plot(g.x / 1000, g.y / 1000, "o", ms=6, mfc="#eb6834", mec="white", mew=1.2, zorder=6)
        elif g.geom_type == "LineString":
            x, y = g.xy
            ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, **kw)


def _map_panel(ax, arr, grid, which, area_utm, places_utm, rivers_utm, n_soil=1, cax=None):
    step = max(1, int(np.ceil(max(arr.shape) / 700)))
    a = arr[::step, ::step]
    x0, y0, x1, y1 = grid.bounds
    ext = [x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000]
    rgba = M._colour(which, a.astype("float32") if which == "dem" else a, n_soil)
    ax.imshow(rgba, extent=ext, interpolation="nearest")
    if which == "dem":
        import matplotlib.pyplot as plt
        ok = a[a != M.DEM_NODATA]
        if ok.size:
            lo, hi = np.percentile(ok, [1, 99])
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("terrain"),
                                       norm=plt.Normalize(lo - (hi - lo) * 0.15 / 0.85, hi))
            cb = plt.colorbar(sm, cax=cax) if cax is not None else plt.colorbar(sm, ax=ax, fraction=0.04)
            cb.set_label("Elevation (m)", fontsize=7)
            cb.ax.tick_params(labelsize=6)
    for r in rivers_utm:
        _draw_outline(ax, r, color="#1565c0", lw=0.7, zorder=4)
    _draw_outline(ax, area_utm, color="#222222", lw=1.0, ls="--", zorder=5)
    for g in places_utm:
        _draw_outline(ax, g, color="#eb6834", lw=1.4, zorder=6)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Easting (km)", fontsize=7)
    ax.set_ylabel("Northing (km)", fontsize=7)


def _advice_page(pdf, adv):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, "Health check and setup suggestions", fontsize=14, weight="bold", color="#1b2a3a")
    h = adv["health"]
    colours = {"OK": "#d8f0dc", "Check": "#fdf0c8", "Fix": "#f9d4d0"}
    rows = [[r.Status, "\n".join(textwrap.wrap(f"{r.Area}: {r.Check}", 34)), "\n".join(textwrap.wrap(r.Result, 48)),
             "\n".join(textwrap.wrap(r._4, 52))] for r in h.itertuples(index=False)]
    n_lines = sum(max(c.count("\n") + 1 for c in row) for row in rows) + 1
    hgt = min(0.55, 0.016 * n_lines)
    ax = fig.add_axes([0.06, 0.93 - hgt, 0.88, hgt])
    ax.set_axis_off()
    t = ax.table(cellText=rows, colLabels=["Status", "Check", "Result", "What to do"], loc="upper left",
                 cellLoc="left", colLoc="left", colWidths=[0.08, 0.25, 0.32, 0.35])
    t.auto_set_font_size(False)
    t.set_fontsize(6.5)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            continue
        lines = max(x.count("\n") + 1 for x in rows[r - 1])
        cell.set_height(cell.get_height() * lines * 1.05)
        if c == 0:
            cell.set_facecolor(colours.get(rows[r - 1][0], "white"))
    y = 0.93 - hgt - 0.04
    tips = []
    if "stream_km2" in adv:
        tips.append(("Stream threshold", f"about {adv['stream_km2']:g} km² (gives roughly 20 to 40 subbasins "
                                         f"for this {adv['catchment_km2']:,.0f} km² catchment)"))
    hru = adv.get("hru", {})
    for key, label in (("landuse", "Land use threshold"), ("soil", "Soil threshold"), ("slope", "Slope threshold")):
        if key in hru:
            tt = hru[key]["table"]
            row = tt[tt["Threshold (%)"] == hru[key]["pick"]].iloc[0]
            txt = f"{hru[key]['pick']}%  (drops: {row['Classes dropped']})"
            if hru[key].get("exempt"):
                txt += f"; exempt {', '.join(hru[key]['exempt'])}"
            tips.append((label, txt))
    if "slope" in hru:
        sc = hru["slope"]["classes"]
        tips.append(("Slope classes", f"{sc['qswat']}  (" + ", ".join(
            f"{a}: {b:.0f}%" for a, b in zip(sc["table"]["Slope class"], sc["table"]["Share of catchment (%)"]))
            + ")"))
    e = adv.get("elevation")
    if e:
        tips.append(("Elevation bands", (f"yes, {e['bands']} bands (relief {e['relief']:,.0f} m); TLAPS "
                                         f"{e['tlaps']} °C/km, PLAPS {e['plaps']:g} mm/km"
                                         if e["use_bands"] else f"not needed (relief {e['relief']:,.0f} m)")))
    fig.text(0.06, y, "Suggested settings", fontsize=11, weight="bold", color="#1b2a3a")
    y -= 0.03
    for label, txt in tips:
        fig.text(0.06, y, label, fontsize=8, weight="bold", color="#1b2a3a")
        wrapped = textwrap.wrap(txt, 95)
        for part in wrapped:
            fig.text(0.27, y, part, fontsize=8, color="#1b2a3a")
            y -= 0.016
        y -= 0.006
    fig.text(0.06, max(y - 0.01, 0.03), "\n".join(textwrap.wrap(
        "HRU thresholds are applied by SWAT inside each subbasin; the suggestions are worked out for the whole "
        "catchment, so treat them as a starting point. Each suggestion moves at most 5% of the area into other "
        "classes. Full tables: Report/Setup suggestions.txt.", 120)), fontsize=7, color="#52606d", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def pdf_report(info, sites, area_wgs84, ms=None, ws=None, rivers=None, adv=None):
    """A short A4 PDF (bytes): maps, tables and climate, health check and suggestions, methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch
    epsg = info["epsg"]
    area_utm = _to_utm([area_wgs84], epsg)[0]
    places_utm = _to_utm([s.geometry for s in sites], epsg)
    rivers_utm = _to_utm([g for g, _, _ in (rivers or [])], epsg)
    buf = io.BytesIO()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    with PdfPages(buf) as pdf:
        # ---------------- page 1: summary and maps ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.965, "SWAT Bharat · data report", fontsize=16, weight="bold", color="#1b2a3a")
        fig.text(0.06, 0.945, f"Made on {datetime.date.today():%d %B %Y}", fontsize=8, color="#52606d")
        own = info.get("own_km2", info["area_km2"])
        lines = [f"Your area: {own:,.0f} km²  ·  download area: {info['area_km2']:,.0f} km² "
                 f"({info['margin']})  ·  WGS 84 / UTM zone {epsg - 32600}N (EPSG:{epsg})",
                 f"Places: {info['places']}"]
        if info.get("traced"):
            lines.append("Catchment traced from the outlet: " + info["traced"])
        if ms is not None:
            lines.append(f"Maps: {info['res']} m cells, {ms.grid.width:,} x {ms.grid.height:,}")
        if ws is not None:
            lines.append(f"Weather: {ws.start_date:%d-%m-%Y} to {ws.end_date:%d-%m-%Y}, "
                         f"{len(ws.stations.get('pcp', []))} rainfall and {len(ws.stations.get('tmp', []))} "
                         f"temperature stations")
        if rivers:
            lines.append(f"River lines for burn-in: {len(rivers)} segments (OpenStreetMap)")
        yy = 0.918
        for ln in lines:
            for part in textwrap.wrap(ln, 115):
                fig.text(0.06, yy, part, fontsize=8, color="#1b2a3a")
                yy -= 0.015
        top = yy - 0.02
        panels = []
        if ms is not None:
            if ms.dem is not None:
                panels.append(("dem", "Elevation (Copernicus DEM)"))
            if ms.landuse is not None:
                panels.append(("landuse", "Land use (ESA WorldCover)"))
            if ms.soil is not None:
                panels.append(("soil", "Soil types (from SoilGrids)"))
        if panels:
            h = (top - 0.05) / len(panels)
            for n, (k, title) in enumerate(panels):
                yb = top - (n + 1) * h + 0.01
                ax = fig.add_axes([0.08, yb, 0.56, h - 0.05])
                arr = getattr(ms, k)
                n_soil = int(ms.soil.max()) if k == "soil" else 1
                _map_panel(ax, arr, ms.grid, k, area_utm, places_utm, rivers_utm, n_soil,
                           cax=fig.add_axes([0.67, yb + 0.02, 0.015, h - 0.08]) if k == "dem" else None)
                ax.set_title(title, fontsize=9, weight="bold", loc="left")
                lax = fig.add_axes([0.67, yb, 0.3, h - 0.05])
                lax.set_axis_off()
                handles = []
                if k == "landuse":
                    for v, name, code, pct in ms.landuse_lookup[["Value", "Land cover (ESA WorldCover)",
                                                                  "SWAT code", "Area (%)"]].itertuples(index=False):
                        handles.append(Patch(color=M.WORLDCOVER_COLOURS.get(v, "#999"),
                                             label=f"{name} → {code} ({pct:.1f}%)"))
                elif k == "soil":
                    cols = M.soil_colours(ms)
                    handles = [Patch(color=cols[c], label=f"{nm} ({p:.1f}%)\n" + "\n".join(textwrap.wrap(w, 30)))
                               for c, nm, w, p in
                               ms.soil_units[["Soil code", "Soil name", "What it is", "Area (%)"]].head(14)
                               .itertuples(index=False)]
                else:
                    lax.set_position([0.75, yb, 0.22, h - 0.05])
                    v = arr[arr != M.DEM_NODATA]
                    if v.size:
                        lax.text(0, 0.9, f"Lowest: {v.min():,.0f} m\nHighest: {v.max():,.0f} m\n"
                                         f"Average: {v.mean():,.0f} m", fontsize=7.5, va="top")
                if handles:
                    lax.legend(handles=handles, loc="center left", fontsize=6.8, frameon=False)
        fig.text(0.06, 0.02, "Orange: your area (what SWAT models). Dashed line: download area (your area + safety "
                 "margin). Blue lines: rivers.",
                 fontsize=7, color="#52606d")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- page 2: tables, climate, methods ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.96
        if ms is not None and ms.soil_units is not None:
            fig.text(0.06, y, "Soil types", fontsize=11, weight="bold", color="#1b2a3a")
            su = ms.soil_units.head(12)
            cols = ["Soil name", "What it is", "Area (%)", "Hydrologic group", "Topsoil clay (%)",
                    "Topsoil AWC (mm/mm)", "Topsoil Ksat (mm/h)", "USLE K"]
            ax = fig.add_axes([0.06, y - 0.03 - 0.02 * (len(su) + 1), 0.88, 0.02 * (len(su) + 1)])
            ax.set_axis_off()
            t = ax.table(cellText=su[cols].astype(str).values, colLabels=[c.replace(" (", "\n(") for c in cols],
                         loc="upper left", cellLoc="left", colLoc="left",
                         colWidths=[0.13, 0.27, 0.08, 0.1, 0.1, 0.11, 0.11, 0.08])
            t.auto_set_font_size(False)
            t.set_fontsize(6.5)
            t.scale(1, 1.25)
            y -= 0.07 + 0.02 * (len(su) + 1)
        if ws is not None and ws.wgn:
            fig.text(0.06, y, "Monthly climate (average of all weather generator stations)", fontsize=11,
                     weight="bold", color="#1b2a3a")
            st = sum(w.stats for w in ws.wgn) / len(ws.wgn)
            ax = fig.add_axes([0.09, y - 0.26, 0.8, 0.22])
            ax.bar(S.MONTHS, st["pcp_ave"], color="#2a78d6", alpha=0.8, label="Rainfall")
            ax.set_ylabel("Average monthly rainfall (mm)")
            ax2 = ax.twinx()
            ax2.plot(S.MONTHS, st["tmp_max_ave"], "-o", color="#eb6834", ms=3, label="Max temperature")
            ax2.plot(S.MONTHS, st["tmp_min_ave"], "-o", color="#f2a541", ms=3, label="Min temperature")
            ax2.set_ylabel("Temperature (°C)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", frameon=False)
            ax.set_title(f"Average yearly rainfall: {st['pcp_ave'].sum():,.0f} mm", fontsize=8, loc="right")
            y -= 0.31
        if ms is not None and ms.landuse_lookup is not None:
            fig.text(0.06, y, "Land use classes", fontsize=11, weight="bold", color="#1b2a3a")
            lk = ms.landuse_lookup
            ax = fig.add_axes([0.06, y - 0.03 - 0.02 * (len(lk) + 1), 0.6, 0.02 * (len(lk) + 1)])
            ax.set_axis_off()
            t = ax.table(cellText=lk.astype(str).values, colLabels=list(lk.columns), loc="upper left",
                         cellLoc="left", colLoc="left", colWidths=[0.12, 0.5, 0.18, 0.2])
            t.auto_set_font_size(False)
            t.set_fontsize(7)
            t.scale(1, 1.25)
        pdf.savefig(fig)
        plt.close(fig)

        if adv is not None:
            _advice_page(pdf, adv)

        # ---------------- last page: methods and references ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.96
        text, refs = methods_text(info, ms, ws)
        fig.text(0.06, y, "Methods paragraph (also in the zip as a text file)", fontsize=11, weight="bold",
                 color="#1b2a3a")
        body = "\n".join(textwrap.wrap(text, 105))
        fig.text(0.06, y - 0.02, body, fontsize=8, va="top", color="#1b2a3a", linespacing=1.45)
        n_lines = body.count("\n") + 1
        y2 = y - 0.05 - n_lines * 0.0142
        if True:
            fig.text(0.06, y2, "References", fontsize=9, weight="bold", color="#1b2a3a")
            rtxt = "\n\n".join("\n".join(textwrap.wrap(r, 120)) for r in refs)
            fig.text(0.06, y2 - 0.015, rtxt, fontsize=7, va="top", color="#3d4652", linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()
