"""
The SWAT side of the app: turns daily weather into SWAT 2012 and SWAT+ input files.

What this file does, in plain words:
  1. Gets IMD rainfall and temperature (using imd_tools, the same engine as the
     IMD Data Puller app) and decides where the "weather stations" are:
       - "grid" way:   every IMD grid cell over your area becomes a station
       - "places" way: every shape (or point) you give becomes one station,
                       holding the area-weighted average over that shape
  2. Adds relative humidity, solar radiation and wind from NASA POWER
     (at the temperature stations), and the ground elevation of every station.
  3. Works out the weather generator (WGEN) statistics: 14 monthly values that
     SWAT uses to fill gaps and to make up rain intensity.
  4. Writes the files in both formats and packs them into one zip:
       SWAT 2012 (ArcSWAT / QSWAT):  pcp.txt, tmp.txt ... + one .txt per station
                                     + WGEN_user table (CSV) for the database
       SWAT+ (SWAT+ Editor):         pcp.cli, tmp.cli ... + .pcp/.tmp/... files
                                     + weather-wgn.cli + the two WGEN CSV files

The app screen (app.py) only calls build_weather() and build_zip(), so the same
steps can be run from a notebook too.
"""

import datetime
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import imd_tools as T
import power_tools as P

APP_NAME = "SWAT Saathi"
MISSING = -99.0          # SWAT (2012 and +) treats -99 as "no value, generate it"

RAIN_COL = T.VARIABLES["rain"]["column"]
TMAX_COL = T.VARIABLES["tmax"]["column"]
TMIN_COL = T.VARIABLES["tmin"]["column"]

# The five kinds of weather files SWAT reads
KINDS = {
    "pcp": {"name": "Rainfall", "unit": "mm", "swat2012": "pcp", "plus": "pcp",
            "plus_title": "Precipitation data"},
    "tmp": {"name": "Temperature", "unit": "°C", "swat2012": "tmp", "plus": "tmp",
            "plus_title": "Temperature data"},
    "hmd": {"name": "Relative humidity", "unit": "fraction", "swat2012": "rh", "plus": "hmd",
            "plus_title": "Relative humidity data"},
    "slr": {"name": "Solar radiation", "unit": "MJ/m²/day", "swat2012": "solar", "plus": "slr",
            "plus_title": "Solar radiation data"},
    "wnd": {"name": "Wind speed", "unit": "m/s", "swat2012": "wind", "plus": "wnd",
            "plus_title": "Wind speed data"},
}
POWER_COLUMN = {"hmd": P.RH, "slr": P.SOLAR, "wnd": P.WIND}
DECIMALS = {"pcp": 2, "tmp": 2, "hmd": 3, "slr": 3, "wnd": 3}

# Weather generator: SWAT+ names, SWAT 2012 names, and what each one means
WGN_COLUMNS = [
    ("tmp_max_ave", "TMPMX", "Average daily maximum temperature (°C)"),
    ("tmp_min_ave", "TMPMN", "Average daily minimum temperature (°C)"),
    ("tmp_max_sd", "TMPSTDMX", "Spread (standard deviation) of daily maximum temperature (°C)"),
    ("tmp_min_sd", "TMPSTDMN", "Spread (standard deviation) of daily minimum temperature (°C)"),
    ("pcp_ave", "PCPMM", "Average total rainfall in the month (mm)"),
    ("pcp_sd", "PCPSTD", "Spread (standard deviation) of daily rainfall (mm)"),
    ("pcp_skew", "PCPSKW", "Skew of daily rainfall"),
    ("wet_dry", "PR_W1_", "Chance of a wet day after a dry day"),
    ("wet_wet", "PR_W2_", "Chance of a wet day after a wet day"),
    ("pcp_days", "PCPD", "Average number of wet days in the month"),
    ("pcp_hhr", "RAINHHMX", "Largest half-hour rainfall (mm), estimated"),
    ("slr_ave", "SOLARAV", "Average daily solar radiation (MJ/m²/day)"),
    ("dew_ave", "DEWPT", "Average daily dew point (°C)"),
    ("wnd_ave", "WNDAV", "Average daily wind speed (m/s)"),
]
WGN_ROUND = {"wet_dry": 3, "wet_wet": 3, "pcp_skew": 3}
MONTH_DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# IMD's rule of thumb for short rain from daily rain: P(t hours) = P(24 h) x (t / 24) ^ (1/3)
HALF_HOUR_FACTOR = round((0.5 / 24) ** (1 / 3), 3)          # = 0.275


# ---------------------------------------------------------------------------
# The containers
# ---------------------------------------------------------------------------

@dataclass
class Station:
    id: str                      # e.g. pcp001 (also the file name)
    kind: str                    # pcp, tmp, hmd, slr or wnd
    lat: float
    lon: float
    data: pd.DataFrame           # daily values; tmp has two columns (max, min)
    source: str = ""             # where the numbers come from, in words
    place: str = ""              # which of your places it belongs to
    elev: float = float("nan")
    elev_source: str = ""


@dataclass
class WgnStation:
    id: str
    lat: float
    lon: float
    elev: float
    rain_yrs: int
    stats: pd.DataFrame          # 12 rows (months) x 14 columns (SWAT+ names)
    based_on: str = ""           # which stations its numbers come from
    estimated: list = field(default_factory=list)   # values estimated, not measured


@dataclass
class WeatherSet:
    mode: str
    start_date: datetime.date
    end_date: datetime.date
    stations: dict = field(default_factory=dict)      # kind -> list of Station
    wgn: list = field(default_factory=list)           # list of WgnStation
    problems: list = field(default_factory=list)      # data that could not be got
    notes: list = field(default_factory=list)         # cautions
    settings: dict = field(default_factory=dict)
    imd: object = None                                 # the imd_tools Result

    def all_stations(self):
        return [s for k in KINDS for s in self.stations.get(k, [])]


# ---------------------------------------------------------------------------
# 1. Stations from IMD data
# ---------------------------------------------------------------------------

def _parse_label(label):
    """'12.75 N, 75.25 E' -> (12.75, 75.25)"""
    la, lo = label.replace(" N", "").replace(" E", "").split(",")
    return float(la), float(lo)


def _grid_stations(res):
    """Every IMD grid cell with data under any of the places becomes a station."""
    rain, temp = {}, {}
    for s in res.sites:
        cv = res.cell_values.get((s.name, "rain"))
        if cv is not None:
            for label in cv.columns:
                if label not in rain and cv[label].notna().any():
                    rain[label] = (cv[label], s.name)
        tx, tn = res.cell_values.get((s.name, "tmax")), res.cell_values.get((s.name, "tmin"))
        if tx is not None and tn is not None:
            for label in tx.columns:
                if label not in temp and label in tn and tx[label].notna().any():
                    temp[label] = (tx[label], tn[label], s.name)
    pcp, tmp = [], []
    for label, (series, place) in rain.items():
        la, lo = _parse_label(label)
        pcp.append(Station("", "pcp", la, lo, series.to_frame(RAIN_COL),
                           f"IMD rainfall grid cell (0.25 degree) centred at {label}", place))
    for label, (tx, tn, place) in temp.items():
        la, lo = _parse_label(label)
        tmp.append(Station("", "tmp", la, lo, pd.DataFrame({TMAX_COL: tx, TMIN_COL: tn}),
                           f"IMD temperature grid cell (1 degree) centred at {label}", place))
    return pcp, tmp


def _place_stations(res):
    """One station per place, holding the area-weighted average over it."""
    pcp, tmp = [], []
    for s in res.sites:
        df = res.daily.get(s.name)
        if df is None:
            continue
        la, lo = s.centre
        how = "nearest grid cell to the point" if s.is_point else "area-weighted average over the shape"
        if RAIN_COL in df and df[RAIN_COL].notna().any():
            pcp.append(Station("", "pcp", la, lo, df[[RAIN_COL]].copy(),
                               f"IMD rainfall, {how} '{s.name}'", s.name))
        if TMAX_COL in df and TMIN_COL in df and df[TMAX_COL].notna().any():
            tmp.append(Station("", "tmp", la, lo, df[[TMAX_COL, TMIN_COL]].copy(),
                               f"IMD temperature, {how} '{s.name}'", s.name))
    return pcp, tmp


def _number(stations, kind):
    """Sort north to south, then west to east, and give IDs like pcp001."""
    stations.sort(key=lambda st: (-round(st.lat, 4), round(st.lon, 4)))
    width = max(3, len(str(len(stations))))
    for i, st in enumerate(stations, 1):
        st.id = f"{kind}{i:0{width}d}"
    return stations


def _distance_km(la1, lo1, la2, lo2):
    return math.hypot(la1 - la2, (lo1 - lo2) * math.cos(math.radians((la1 + la2) / 2))) * T.KM_PER_DEGREE


def _nearest(stations, lat, lon):
    return min(stations, key=lambda st: _distance_km(st.lat, st.lon, lat, lon))


# ---------------------------------------------------------------------------
# 2. Weather generator statistics
# ---------------------------------------------------------------------------

def wgn_statistics(rain, tmax, tmin, solar, dew, wind, wet_mm=0.1, half_hour_factor=HALF_HOUR_FACTOR):
    """The 14 monthly weather generator values (SWAT+ names), from daily series.

    All series share one daily date index; missing days are NaN and skipped.
    Rainfall statistics use every day (dry days count as 0 mm), as SWAT's
    own tools (pcpSTAT, WGNmaker) do.
    """
    idx = rain.index
    month = idx.month
    out = pd.DataFrame(index=range(1, 13), columns=[c[0] for c in WGN_COLUMNS], dtype="float64")

    r = rain.astype("float64")
    wet = (r >= wet_mm).where(r.notna())                       # True / False / NaN
    prev = wet.shift(1)
    for m in range(1, 13):
        sel = month == m
        rm = r[sel].dropna()
        out.loc[m, "tmp_max_ave"] = tmax[sel].mean()
        out.loc[m, "tmp_min_ave"] = tmin[sel].mean()
        out.loc[m, "tmp_max_sd"] = tmax[sel].std()
        out.loc[m, "tmp_min_sd"] = tmin[sel].std()
        if len(rm):
            out.loc[m, "pcp_ave"] = rm.mean() * MONTH_DAYS[m - 1]
            out.loc[m, "pcp_sd"] = rm.std()
            sk = rm.skew()
            out.loc[m, "pcp_skew"] = sk if np.isfinite(sk) else 0.0
            w, p = wet[sel], prev[sel]
            both = w.notna() & p.notna()
            w, p = w[both].astype(bool), p[both].astype(bool)
            out.loc[m, "wet_dry"] = (w & ~p).sum() / max((~p).sum(), 1)
            out.loc[m, "wet_wet"] = (w & p).sum() / max(p.sum(), 1)
            out.loc[m, "pcp_days"] = (rm >= wet_mm).mean() * MONTH_DAYS[m - 1]
            out.loc[m, "pcp_hhr"] = rm.max() * half_hour_factor
        out.loc[m, "slr_ave"] = solar[sel].mean()
        out.loc[m, "dew_ave"] = dew[sel].mean()
        out.loc[m, "wnd_ave"] = wind[sel].mean()
    out = out.astype("float64")
    for c in out.columns:
        out[c] = out[c].round(WGN_ROUND.get(c, 2))
    out.index.name = "Month"
    return out


def estimated_solar(tmax, tmin, lat, k_rs=0.16):
    """Hargreaves-Samani solar radiation (FAO-56 eq. 50), MJ/m²/day, used only when
    NASA POWER is switched off: Rs = k x square root of (Tmax - Tmin) x Ra."""
    ra = T.extraterrestrial_radiation(lat, tmax.index)
    return pd.Series(k_rs * np.sqrt(np.clip(tmax - tmin, 0, None)) * ra, index=tmax.index)


# ---------------------------------------------------------------------------
# 3. The main job
# ---------------------------------------------------------------------------

def build_weather(sites, start_date, end_date, data_dir, mode="grid", use_power=True,
                  wet_mm=0.1, half_hour_factor=HALF_HOUR_FACTOR, backup_dir=None,
                  progress=None, power_fetch=None, elevation_fetch=None):
    """Get the data and build every station.

    sites:      list of imd_tools.Site (shapes or points)
    mode:       "grid" (every IMD cell is a station) or "places" (one station per place)
    use_power:  add humidity, solar radiation and wind from NASA POWER
    progress:   optional function progress(fraction 0-1, message)
    power_fetch / elevation_fetch: stand-ins for testing without internet
    """
    say = progress or (lambda f, m: None)
    power_fetch = power_fetch or P.fetch_power_many
    elevation_fetch = elevation_fetch or P.fetch_elevations
    start_date, end_date = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
    ws = WeatherSet(mode=mode, start_date=start_date, end_date=end_date)
    dates = pd.date_range(start_date, end_date, freq="D")

    # --- IMD rainfall and temperature ---
    res = T.pull_data(sites, ["rain", "tmax", "tmin"], start_date.year, end_date.year, data_dir,
                      keep_cell_values=(mode == "grid"), backup_dir=backup_dir,
                      start_date=start_date, end_date=end_date,
                      progress=lambda f, m: say(0.75 * f, m))
    ws.imd = res
    ws.problems += res.problems
    single = [n for n in res.notes if "from a single grid cell" in n]
    ws.notes += [n for n in res.notes if "don't cover whole years" not in n and n not in single]
    if single and mode == "places":
        ws.notes.append(f"{len(single)} of your places get their temperature from a single 1 degree IMD cell "
                        f"(about 110 km across), so nearby places can share the same temperature.")
    pcp, tmp = _grid_stations(res) if mode == "grid" else _place_stations(res)
    if not pcp:
        raise ValueError("No rainfall data was found for these places. They may be over the sea, "
                         "or IMD's server could not be reached.")
    for st in pcp + tmp:
        st.data = st.data.reindex(dates)
    ws.stations["pcp"] = _number(pcp, "pcp")
    ws.stations["tmp"] = _number(tmp, "tmp")
    if not tmp:
        ws.problems.append("No temperature data was found (IMD temperature starts in 1951). "
                           "SWAT will generate temperature from the weather generator, which then "
                           "needs a weather generator station from another source.")

    # --- humidity, solar radiation and wind from NASA POWER, at the temperature stations ---
    # (in the "places" way, or if there is no temperature, at the rainfall stations)
    anchors = tmp if tmp else pcp
    power_tables, power_elev = {}, {}
    if use_power:
        period = P.power_period(start_date, end_date)
        if period is None:
            ws.problems.append("NASA POWER starts on 1 January 1981, so humidity, solar radiation "
                               "and wind could not be added for this period. SWAT will generate them.")
        else:
            points = [(round(st.lat, 4), round(st.lon, 4)) for st in anchors]
            power_tables, power_elev, probs = power_fetch(
                points, start_date, end_date, progress=lambda f, m: say(0.75 + 0.15 * f, m))
            ws.problems += probs
            if period[0] > start_date:
                ws.notes.append("NASA POWER starts on 1 January 1981, so humidity, solar radiation and "
                                "wind are -99 (missing) before that. SWAT fills those days from the "
                                "weather generator.")
            for kind in ("hmd", "slr", "wnd"):
                made = []
                for st in anchors:
                    table = power_tables.get((round(st.lat, 4), round(st.lon, 4)))
                    if table is None:
                        continue
                    col = POWER_COLUMN[kind]
                    made.append(Station("", kind, st.lat, st.lon, table[[col]].reindex(dates),
                                        f"NASA POWER daily data at {st.lat:.2f} N, {st.lon:.2f} E",
                                        st.place))
                ws.stations[kind] = _number(made, kind)

    # --- elevations ---
    say(0.92, "Finding the elevation of each station")
    points = list(dict.fromkeys((round(st.lat, 4), round(st.lon, 4)) for st in ws.all_stations()))
    heights = elevation_fetch(points)
    missing = 0
    for st in ws.all_stations():
        key = (round(st.lat, 4), round(st.lon, 4))
        if key in heights:
            st.elev, st.elev_source = heights[key], "Copernicus 90 m DEM (via Open-Meteo)"
        elif power_elev:
            near = min(power_elev, key=lambda p: _distance_km(p[0], p[1], st.lat, st.lon))
            if power_elev[near] is not None:
                st.elev, st.elev_source = power_elev[near], "NASA POWER grid elevation (rough)"
        if not np.isfinite(st.elev):
            st.elev, st.elev_source = 0.0, "not found, set to 0"
            missing += 1
    if missing:
        ws.problems.append(f"The elevation of {missing} station(s) could not be found, so it was set to "
                           f"0 m. Elevation only matters in SWAT if you use elevation bands; you can type "
                           f"the right values into the station lists.")

    # --- weather generator: one per temperature station (or rainfall station if no temperature) ---
    say(0.95, "Working out the weather generator statistics")
    for i, anchor in enumerate(anchors, 1):
        rain_st = _nearest(pcp, anchor.lat, anchor.lon)
        rain = rain_st.data[RAIN_COL]
        if tmp:
            temp_st = anchor if anchor.kind == "tmp" else _nearest(tmp, anchor.lat, anchor.lon)
            tx, tn = temp_st.data[TMAX_COL], temp_st.data[TMIN_COL]
        else:
            tx = tn = pd.Series(np.nan, index=dates)
        table = power_tables.get((round(anchor.lat, 4), round(anchor.lon, 4)))
        estimated = []
        if table is not None:
            table = table.reindex(dates)
            solar, dew, wind = table[P.SOLAR], table[P.DEW], table[P.WIND]
        else:
            solar = estimated_solar(tx, tn, anchor.lat)
            dew = tn.copy()                                   # FAO-56: dew point is close to Tmin
            wind = pd.Series(2.0, index=dates)                # FAO-56 world average
            estimated = ["solar radiation (Hargreaves-Samani from temperature)",
                         "dew point (taken as the minimum temperature)", "wind speed (set to 2 m/s)"]
        stats = wgn_statistics(rain, tx, tn, solar, dew, wind, wet_mm, half_hour_factor)
        years = int(rain.dropna().index.year.nunique())
        based = f"rainfall from {rain_st.id}"
        if tmp:
            based += f", temperature from {temp_st.id}"
        based += ", humidity / solar / wind from NASA POWER" if table is not None else ""
        ws.wgn.append(WgnStation(f"wgn{i:03d}", anchor.lat, anchor.lon, anchor.elev, years, stats,
                                 based, estimated))
        if stats.isna().any().any():
            ws.problems.append(f"Some weather generator values for wgn{i:03d} are empty because there "
                               f"was no data for them; fill them in before using the station.")
    if any(w.estimated for w in ws.wgn):
        ws.notes.append("Without NASA POWER, the weather generator's solar radiation, dew point and "
                        "wind are rough estimates from temperature.")

    ws.settings = {
        "Station layout": ("Every IMD grid cell is a station" if mode == "grid"
                           else "One station per place (area-weighted average)"),
        "Period": f"{start_date:%d-%m-%Y} to {end_date:%d-%m-%Y}",
        "Humidity, solar radiation and wind": "NASA POWER" if use_power else "Left to SWAT's weather generator",
        "Wet day threshold for the weather generator": f"{wet_mm} mm",
        "Half-hour rain factor (largest daily rain x factor)": f"{half_hour_factor}",
        "Missing value written as": f"{MISSING:g}",
    }
    say(1.0, "Done")
    return ws


# ---------------------------------------------------------------------------
# 4. Tables for checking
# ---------------------------------------------------------------------------

def station_table(ws):
    rows = []
    for st in ws.all_stations():
        n_ok = int(st.data.notna().all(axis=1).sum())
        rows.append({
            "Station": st.id, "Data": KINDS[st.kind]["name"],
            "Latitude": round(st.lat, 4), "Longitude": round(st.lon, 4),
            "Elevation (m)": round(st.elev, 1), "Elevation from": st.elev_source,
            "Days with data (%)": round(100 * n_ok / max(len(st.data), 1), 1),
            "Place": st.place, "Source": st.source})
    for w in ws.wgn:
        rows.append({"Station": w.id, "Data": "Weather generator", "Latitude": round(w.lat, 4),
                     "Longitude": round(w.lon, 4), "Elevation (m)": round(w.elev, 1),
                     "Elevation from": "", "Days with data (%)": np.nan, "Place": "",
                     "Source": w.based_on})
    return pd.DataFrame(rows)


def wgn_long_table(ws):
    """All weather generator stations, one row per station and month, plain names."""
    parts = []
    for w in ws.wgn:
        t = w.stats.copy()
        t.columns = [c[2] for c in WGN_COLUMNS]
        t.insert(0, "Month", MONTHS)
        t.insert(0, "Station", w.id)
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def daily_table(ws, kind):
    """Every station of one kind side by side (one column per station)."""
    cols = {}
    for st in ws.stations.get(kind, []):
        if kind == "tmp":
            cols[f"{st.id} max"] = st.data[TMAX_COL]
            cols[f"{st.id} min"] = st.data[TMIN_COL]
        else:
            cols[st.id] = st.data.iloc[:, 0]
    df = pd.DataFrame(cols)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# 5. Writing the files
# ---------------------------------------------------------------------------

def _fmt(v, d):
    return f"{MISSING:.1f}" if (v is None or not np.isfinite(v)) else f"{v:.{d}f}"


def _values(st):
    """The station's daily numbers as a 2-D float array (days x 1 or 2)."""
    return st.data.to_numpy(dtype="float64")


def swat2012_files(ws):
    """{path inside the zip: text} for ArcSWAT / QSWAT (SWAT 2012)."""
    files = {}
    base = "SWAT2012/ObservedWeather/"
    for kind, info in KINDS.items():
        stations = ws.stations.get(kind, [])
        if not stations:
            continue
        lines = ["ID,NAME,LAT,LONG,ELEVATION"]
        for i, st in enumerate(stations, 1):
            lines.append(f"{i},{st.id},{st.lat:.4f},{st.lon:.4f},{st.elev:.1f}")
            first = st.data.index[0]
            vals, d = _values(st), DECIMALS[kind]
            body = [f"{first:%Y%m%d}"]
            if kind == "tmp":
                body += [f"{_fmt(a, d)},{_fmt(b, d)}" for a, b in vals]
            else:
                body += [_fmt(a, d) for a in vals[:, 0]]
            files[base + f"{st.id}.txt"] = "\r\n".join(body) + "\r\n"
        files[base + f"{info['swat2012']}.txt"] = "\r\n".join(lines) + "\r\n"

    # weather generator table, laid out like the WGEN_user table in SWAT2012.mdb
    rows = []
    for n, w in enumerate(ws.wgn, 1):
        row = {"OBJECTID": n, "STATE": "India", "STATION": w.id, "WLATITUDE": round(w.lat, 4),
               "WLONGITUDE": round(w.lon, 4), "WELEV": round(w.elev, 1), "RAIN_YRS": w.rain_yrs}
        for plus, old, _ in WGN_COLUMNS:
            for m in range(1, 13):
                row[f"{old}{m}"] = w.stats.loc[m, plus]
        rows.append(row)
    if rows:
        files["SWAT2012/WeatherGenerator/WGEN_user.csv"] = pd.DataFrame(rows).to_csv(index=False,
                                                                                       lineterminator="\r\n")
    return files


def swatplus_files(ws, today=None):
    """{path inside the zip: text} for SWAT+ (import into SWAT+ Editor)."""
    today = today or datetime.date.today()
    stamp = f"written by {APP_NAME} on {today:%Y-%m-%d}"
    files = {}
    base = "SWATplus/ObservedWeather/"
    for kind, info in KINDS.items():
        stations = ws.stations.get(kind, [])
        if not stations:
            continue
        ext = info["plus"]
        names = []
        for st in stations:
            fname = f"{st.id}.{ext}"
            names.append(fname)
            idx = st.data.index
            nbyr = int(idx[-1].year - idx[0].year + 1)
            vals, d = _values(st), DECIMALS[kind]
            years, days = idx.year.to_numpy(), idx.dayofyear.to_numpy()
            lines = [f"{fname}: {info['plus_title']} - {_ascii(st.source)} - {stamp}",
                     "nbyr     tstep       lat       lon      elev",
                     f"{nbyr:>4}{0:>10}{st.lat:>10.3f}{st.lon:>10.3f}{st.elev:>10.3f}"]
            if kind == "tmp":
                lines += [f"{y:>4}{j:>5}{_fmt(a, d):>10}{_fmt(b, d):>10}"
                          for y, j, (a, b) in zip(years, days, vals)]
            else:
                lines += [f"{y:>4}{j:>5}{_fmt(a, d):>10}" for y, j, a in zip(years, days, vals[:, 0])]
            files[base + fname] = "\n".join(lines) + "\n"
        files[base + f"{ext}.cli"] = "\n".join(
            [f"{ext}.cli: {info['plus_title'].replace(' data', '')} file names - {stamp}", "filename"]
            + names) + "\n"

    if ws.wgn:
        head = "  ".join(c[0] for c in WGN_COLUMNS)
        lines = [f"weather-wgn.cli: {stamp}"]
        for w in ws.wgn:
            lines.append(f"{w.id:<12}{w.lat:>10.3f}{w.lon:>10.3f}{w.elev:>10.2f}{w.rain_yrs:>6}")
            lines.append(head)
            for m in range(1, 13):
                lines.append("".join(f"{_fmt(w.stats.loc[m, c[0]], WGN_ROUND.get(c[0], 2)):>12}"
                                     for c in WGN_COLUMNS))
        files["SWATplus/WeatherGenerator/weather-wgn.cli"] = "\n".join(lines) + "\n"

        stations = pd.DataFrame([{"id": n, "name": w.id, "lat": round(w.lat, 4), "lon": round(w.lon, 4),
                                  "elev": round(w.elev, 1), "rain_yrs": w.rain_yrs}
                                 for n, w in enumerate(ws.wgn, 1)])
        monthly, k = [], 0
        for n, w in enumerate(ws.wgn, 1):
            for m in range(1, 13):
                k += 1
                row = {"id": k, "wgn_id": n, "month": m}
                row.update({c[0]: w.stats.loc[m, c[0]] for c in WGN_COLUMNS})
                monthly.append(row)
        files["SWATplus/WeatherGenerator/wgn-stations.csv"] = stations.to_csv(index=False)
        files["SWATplus/WeatherGenerator/wgn-monthly-values.csv"] = pd.DataFrame(monthly).to_csv(index=False)
    return files


def _ascii(text):
    return (str(text).replace("°", " deg").replace("²", "2").replace("'", "")
            .encode("ascii", "ignore").decode())


def weather_summary(ws):
    """Lines for the zip's README: stations, cautions, settings."""
    counts = ", ".join(f"{KINDS[k]['name'].lower()} {len(v)}" for k, v in ws.stations.items() if v)
    counts += f", weather generator {len(ws.wgn)}"
    st = station_table(ws)[["Station", "Data", "Latitude", "Longitude", "Elevation (m)"]]
    return {"period": f"{ws.start_date:%d-%m-%Y} to {ws.end_date:%d-%m-%Y}", "counts": counts,
            "cautions": [_ascii(x) for x in ws.problems + ws.notes],
            "settings": ws.settings, "stations": _ascii(st.to_string(index=False))}


CHECK_NAMES = {"pcp": "rainfall (mm)", "tmp": "temperature (deg C)", "hmd": "relative humidity (fraction)",
               "slr": "solar radiation (MJ per m2 per day)", "wnd": "wind speed (m per s)"}


def check_files(ws):
    files = {"CheckTables/Weather station list.csv": station_table(ws).to_csv(index=False)}
    if ws.wgn:
        files["CheckTables/Weather generator values.csv"] = wgn_long_table(ws).to_csv(index=False)
    for kind in KINDS:
        if ws.stations.get(kind):
            files[f"CheckTables/Daily {CHECK_NAMES[kind]}.csv"] = \
                daily_table(ws, kind).round(DECIMALS[kind]).to_csv()
    return files


def weather_files(ws, swat2012=True, swatplus=True):
    """All weather files: {path inside the zip: text}."""
    files = {}
    if swat2012:
        files.update(swat2012_files(ws))
    if swatplus:
        files.update(swatplus_files(ws))
    files.update(check_files(ws))
    return files
