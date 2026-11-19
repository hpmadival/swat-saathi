"""
Advice before you start SWAT: a health check of the downloaded data, and suggested
settings for the steps people most often guess at.

In plain words:
  - Health check: looks for gaps and oddities (empty DEM cells, missing rain days,
    tiny soil types, a short weather period...) and says what to do about each.
  - HRU thresholds: SWAT drops land uses, soils and slope classes that cover less
    than a set percentage. For each candidate threshold, this shows what would be
    dropped, and suggests one that keeps the model simple without losing much.
  - Slope classes: break points that split your catchment's slopes into sensible groups.
  - Elevation bands: whether your catchment is hilly enough to need them, how many,
    and temperature / rainfall lapse rates.

Everything is worked out inside your catchment (your shapes, or the traced catchment),
not the extra margin around it, because that is what SWAT will model.
"""

import math

import numpy as np
import pandas as pd

import swat_maps as M

THRESHOLDS = [0, 5, 10, 15, 20]
NICE_BREAKS = [0.5, 1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
OK, CHECK, FIX = "OK", "Check", "Fix"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def catchment_mask(ms, sites):
    """Cells inside your shapes (not the margin). Falls back to the study area for points only."""
    import geopandas as gpd
    from rasterio.features import geometry_mask
    shapes = [s.geometry for s in sites if not s.is_point]
    g = ms.grid
    if not shapes:
        geoms = [ms.area_utm]
    else:
        geoms = list(gpd.GeoSeries(shapes, crs=4326).to_crs(g.epsg))
    return geometry_mask(geoms, out_shape=(g.height, g.width), transform=g.transform, invert=True,
                         all_touched=False)


def slope_percent(dem, res):
    """Slope in percent from a projected DEM (metres)."""
    d = dem.astype("float64")
    bad = d == M.DEM_NODATA
    d[bad] = np.nan
    gy, gx = np.gradient(d, res)
    s = np.sqrt(gx ** 2 + gy ** 2) * 100
    s[bad] = np.nan
    return s


def _shares(values, labels=None):
    """{label: % of cells} from an array of class values."""
    v, c = np.unique(values, return_counts=True)
    tot = c.sum() or 1
    out = {}
    for vv, cc in zip(v, c):
        name = labels.get(int(vv), str(vv)) if labels else str(vv)
        out[name] = out.get(name, 0) + 100.0 * cc / tot
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _nice(x):
    return min(NICE_BREAKS, key=lambda b: abs(b - x))


# ---------------------------------------------------------------------------
# Slope classes
# ---------------------------------------------------------------------------
def slope_classes(slope_in):
    """Suggested slope class breaks (%) and the share of the catchment in each class."""
    s = slope_in[np.isfinite(slope_in)]
    if s.size == 0:
        return None
    p33, p67 = np.percentile(s, [33, 67])
    breaks = sorted({_nice(p33), _nice(p67)})
    if len(breaks) == 1 or breaks[0] == breaks[-1]:
        breaks = sorted({_nice(np.percentile(s, 50))})
    edges = [0] + breaks + [9999]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        share = 100.0 * ((s >= lo) & (s < hi)).mean()
        rows.append({"Slope class": f"{lo:g} to {hi:g} %" if hi < 9999 else f"over {lo:g} %",
                     "Share of catchment (%)": round(share, 1)})
    return {"breaks": breaks, "qswat": "0-" + "-".join(f"{b:g}" for b in breaks) + "-9999",
            "table": pd.DataFrame(rows), "median": float(np.median(s)), "p90": float(np.percentile(s, 90)),
            "flat_share": float(100 * (s < 0.5).mean())}


# ---------------------------------------------------------------------------
# HRU thresholds
# ---------------------------------------------------------------------------
def threshold_table(shares, keep_anyway=()):
    """For each candidate threshold, which classes fall below it and how much area they cover.
    keep_anyway: classes you would exempt (e.g. water, towns), so they never count as dropped."""
    rows = []
    for t in THRESHOLDS:
        dropped = {k: v for k, v in shares.items() if v < t and k not in keep_anyway}
        rows.append({"Threshold (%)": t, "Classes kept": len(shares) - len(dropped),
                     "Classes dropped": ", ".join(f"{k} ({v:.1f}%)" for k, v in dropped.items()) or "none",
                     "Area moved to other classes (%)": round(sum(dropped.values()), 1)})
    return pd.DataFrame(rows)


def pick_threshold(table, max_loss=5.0):
    """Suggested threshold: drop as many small classes as possible while moving no more than
    max_loss % of the catchment. When several thresholds drop the same classes, take the lowest,
    because inside single subbasins a higher threshold drops more. If nothing small exists,
    10% keeps the number of HRUs down without losing anything."""
    ok = table[table["Area moved to other classes (%)"] <= max_loss]
    if not len(ok):
        return 0
    best = ok["Area moved to other classes (%)"].max()
    if best == 0:
        return int(min(10, ok["Threshold (%)"].max()))
    return int(ok.loc[ok["Area moved to other classes (%)"] == best, "Threshold (%)"].min())


def hru_advice(ms, cmask, slope):
    """Suggested land use / soil / slope thresholds with the tables behind them."""
    out = {}
    if ms.landuse is not None:
        lu_codes = dict(zip(ms.landuse_lookup["Value"], ms.landuse_lookup["SWAT code"]))
        vals = ms.landuse[cmask & (ms.landuse > 0)]
        shares = _shares(vals, lu_codes)
        exempt = [k for k in shares if k in ("WATR", "URMD", "URHD", "URLD", "UCOM") and shares[k] < 10]
        t = threshold_table(shares, exempt)
        out["landuse"] = {"shares": shares, "table": t, "pick": pick_threshold(t), "exempt": exempt}
    if ms.soil is not None:
        names = dict(zip(ms.soil_units["Soil code"], ms.soil_units["Soil name"]))
        vals = ms.soil[cmask & (ms.soil > 0)]
        shares = _shares(vals, names)
        t = threshold_table(shares)
        out["soil"] = {"shares": shares, "table": t, "pick": pick_threshold(t)}
    if slope is not None:
        sc = slope_classes(slope[cmask])
        if sc is not None:
            shares = dict(zip(sc["table"]["Slope class"], sc["table"]["Share of catchment (%)"]))
            t = threshold_table(shares)
            out["slope"] = {"shares": shares, "table": t, "pick": pick_threshold(t), "classes": sc}
    return out


# ---------------------------------------------------------------------------
# Elevation bands and lapse rates
# ---------------------------------------------------------------------------
def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) < 150:
        return None, None
    a, b = np.polyfit(x[ok], y[ok], 1)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return a, r


def elevation_advice(dem_in, ws=None):
    v = dem_in[np.isfinite(dem_in) & (dem_in != M.DEM_NODATA)]
    if v.size == 0:
        return None
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    relief = hi - lo
    need = relief >= 500
    n = int(min(10, max(3, math.ceil(relief / 250)))) if need else 0
    advice = {"lowest": float(v.min()), "highest": float(v.max()), "mean": float(v.mean()),
              "relief": relief, "use_bands": need, "bands": n,
              "tlaps": -6.5, "tlaps_note": "the usual rate at which air cools with height",
              "plaps": 0.0, "plaps_note": "no clear pattern, so 0 (SWAT's default) is safest"}
    if ws is not None:
        pcp = ws.stations.get("pcp", [])
        yrs = max((ws.end_date - ws.start_date).days / 365.25, 1)
        annual = [float(st.data.iloc[:, 0].sum(skipna=True) / yrs) for st in pcp]
        a, r = _fit([st.elev for st in pcp], annual)
        if a is not None and abs(r) >= 0.5:
            advice["plaps"] = round(a * 1000, 0)
            advice["plaps_note"] = (f"from the {len(pcp)} IMD rainfall stations: yearly rain changes by "
                                    f"{a * 1000:+,.0f} mm per km of height (correlation {r:.2f})")
        elif a is not None:
            advice["plaps_note"] = (f"the {len(pcp)} rainfall stations show only a weak link between height "
                                    f"and rain (correlation {r:.2f}), so 0 is safest")
    return advice


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def health_check(ms=None, ws=None, cmask=None, slope=None, rivers=None, traced=None, sites=()):
    rows = []

    def add(area, check, status, result, todo=""):
        rows.append({"Area": area, "Check": check, "Status": status, "Result": result, "What to do": todo})

    if traced is not None:
        if traced.cut_off:
            add("Catchment", "Traced boundary complete", FIX, "The traced catchment touches the edge of the "
                "search square, so part of it may be missing.", "Raise the largest search size, or upload the "
                "boundary instead.")
        else:
            add("Catchment", "Traced boundary complete", OK, f"{traced.area_km2:,.0f} km² at {traced.res} m.")
        st = OK if traced.moved_m <= 500 else CHECK
        add("Catchment", "Outlet on the river", st, f"The outlet was moved {traced.moved_m:,.0f} m onto the "
            f"main river.", "" if st == OK else "Check on the satellite map that the moved outlet is on the "
            "right river, above or below any confluence you care about.")

    if ms is not None and cmask is not None:
        n_in = int(cmask.sum()) or 1
        if ms.dem is not None:
            gap = 100.0 * ((ms.dem == M.DEM_NODATA) & cmask).sum() / n_in
            add("Elevation", "Empty cells inside the catchment", OK if gap == 0 else FIX,
                f"{gap:.2f}% of cells have no elevation.",
                "" if gap == 0 else "Usually sea or a missing tile: check the outline is on land.")
            if slope is not None:
                s = slope[cmask & np.isfinite(slope)]
                flat = float(100 * (s < 0.5).mean()) if s.size else 0
                st = OK if flat < 20 else CHECK
                add("Elevation", "Flat land (slope under 0.5%)", st, f"{flat:.0f}% of the catchment is very flat.",
                    "" if st == OK else "Streams from the DEM may wander in flat land: use the river lines "
                    "(Maps/rivers.shp) to burn in the stream network.")
        if ms.landuse is not None:
            gap = 100.0 * ((ms.landuse == 0) & cmask).sum() / n_in
            add("Land use", "Empty cells inside the catchment", OK if gap < 0.5 else CHECK,
                f"{gap:.2f}% of cells have no land use.", "" if gap < 0.5 else "Check that the land use map "
                "covers the whole catchment.")
            wat = ms.landuse_lookup.loc[ms.landuse_lookup["SWAT code"] == "WATR", "Area (%)"].sum()
            if wat > 5:
                add("Land use", "Water bodies", CHECK, f"{wat:.1f}% is water.", "Large reservoirs are better "
                    "added as SWAT reservoirs than left as water land use.")
        if ms.soil_units is not None:
            tiny = ms.soil_units[ms.soil_units["Area (%)"] < 1]
            add("Soil", "Number of soil types", OK if len(ms.soil_units) <= 15 else CHECK,
                f"{len(ms.soil_units)} soil types, {len(tiny)} of them under 1% of the area.",
                "" if len(tiny) == 0 else "Tiny soil types are dropped by the soil threshold anyway.")
            dg = ms.soil_units.loc[ms.soil_units["Hydrologic group"] == "D", "Area (%)"].sum()
            if dg > 60:
                add("Soil", "Runoff-prone soils", CHECK, f"{dg:.0f}% of the area is hydrologic group D "
                    "(high runoff).", "Normal for clayey soils; keep it in mind when calibrating CN2.")

    if ws is not None:
        years = ws.end_date.year - ws.start_date.year + 1
        add("Weather", "Length of record", OK if years >= 8 else CHECK, f"{years} years.",
            "" if years >= 8 else "Use at least 8 to 10 years: 2 to 3 for warm-up, the rest for calibration "
            "and validation.")
        for kind, label in (("pcp", "Rainfall"), ("tmp", "Temperature"), ("hmd", "Humidity, solar, wind")):
            stations = ws.stations.get(kind, [])
            if not stations:
                if kind != "hmd":
                    add("Weather", f"{label} stations", FIX, "None.", "SWAT would make all of it up from the "
                        "weather generator.")
                continue
            miss = float(np.mean([100 * st.data.isna().any(axis=1).mean() for st in stations]))
            st = OK if miss < 1 else (CHECK if miss < 10 else FIX)
            add("Weather", f"{label}: missing days", st, f"{miss:.1f}% of days are -99 (missing).",
                "" if st == OK else ("Before 1981 NASA POWER has no data, so SWAT generates those days."
                                     if kind == "hmd" else "Press the button again to fetch years IMD's server "
                                     "missed."))
        empty = [w.id for w in ws.wgn if w.stats.isna().any().any()]
        add("Weather", "Weather generator complete", OK if not empty else FIX,
            "All 14 monthly values filled." if not empty else f"Empty values in {', '.join(empty)}.",
            "" if not empty else "SWAT will not run with empty weather generator values: fill them first.")
        if ws.mode == "grid" and len(ws.stations.get("pcp", [])) < 2:
            add("Weather", "Rainfall detail", CHECK, "Only one rainfall station.", "Fine for a small catchment. "
                "For more spatial detail, add a margin so more IMD cells are included.")
    if rivers is not None:
        add("Rivers", "River lines for burn-in", OK if rivers else CHECK,
            f"{len(rivers)} river segments." if rivers else "No mapped rivers.",
            "" if rivers else "Burn-in is optional; the DEM alone is usually fine in hilly land.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Everything together, and the text version
# ---------------------------------------------------------------------------
def build_advice(ms=None, ws=None, sites=(), rivers=None, traced=None):
    cmask = catchment_mask(ms, sites) if ms is not None else None
    slope = slope_percent(ms.dem, ms.grid.res) if (ms is not None and ms.dem is not None) else None
    adv = {"health": health_check(ms, ws, cmask, slope, rivers, traced, sites)}
    if ms is not None:
        adv["hru"] = hru_advice(ms, cmask, slope)
        if ms.dem is not None:
            adv["elevation"] = elevation_advice(np.where(cmask, ms.dem, M.DEM_NODATA), ws)
        catch_km2 = float(cmask.sum()) * ms.grid.res ** 2 / 1e6
        adv["catchment_km2"] = catch_km2
        # stream threshold: aim for roughly 20 to 40 subbasins
        adv["stream_km2"] = round(max(catch_km2 / 30, 0.5), 1 if catch_km2 < 300 else 0)
    return adv


def advice_text(adv):
    out = ["SETUP SUGGESTIONS", "=================", ""]
    h = adv["health"]
    out += ["HEALTH CHECK", "------------"]
    for r in h.itertuples(index=False):
        out.append(f"[{r.Status}] {r.Area} - {r.Check}: {r.Result}" + (f" -> {r._4}" if r._4 else ""))
    out.append("")
    if "stream_km2" in adv:
        out += ["WATERSHED DELINEATION", "---------------------",
                f"Catchment: {adv['catchment_km2']:,.0f} km2. A stream threshold of about {adv['stream_km2']:g} km2 "
                f"gives roughly 20 to 40 subbasins (QSWAT: 'Threshold' in km2, or the same area in cells).", ""]
    hru = adv.get("hru", {})
    if hru:
        out += ["HRU THRESHOLDS (percent, as in QSWAT / ArcSWAT 'Multiple HRUs')",
                "----------------------------------------------------------------",
                "SWAT applies these inside each subbasin; the numbers here are for the whole catchment,",
                "so treat them as a guide. Each suggestion moves at most 5% of the area into other classes.", ""]
        for key, label in (("landuse", "Land use"), ("soil", "Soil"), ("slope", "Slope")):
            if key in hru:
                out.append(f"{label}: suggested threshold {hru[key]['pick']}%")
                if hru[key].get("exempt"):
                    out.append(f"  Exempt (keep even if small): {', '.join(hru[key]['exempt'])}")
                out.append(hru[key]["table"].to_string(index=False))
                out.append("")
        if "slope" in hru:
            sc = hru["slope"]["classes"]
            out += ["SLOPE CLASSES", "-------------",
                    f"Suggested breaks: {', '.join(f'{b:g}%' for b in sc['breaks'])}  (QSWAT: {sc['qswat']})",
                    f"Median slope {sc['median']:.1f}%, 90% of the land is below {sc['p90']:.1f}%.",
                    sc["table"].to_string(index=False), ""]
    e = adv.get("elevation")
    if e:
        out += ["ELEVATION BANDS", "---------------",
                f"Elevation {e['lowest']:,.0f} to {e['highest']:,.0f} m (relief {e['relief']:,.0f} m)."]
        if e["use_bands"]:
            out.append(f"Use elevation bands: yes, {e['bands']} bands. TLAPS = {e['tlaps']} C/km "
                       f"({e['tlaps_note']}); PLAPS = {e['plaps']:g} mm/km ({e['plaps_note']}).")
        else:
            out.append("Use elevation bands: not needed (relief under 500 m).")
        out.append("")
    return "\r\n".join(out).replace("°", " deg").replace("²", "2")
