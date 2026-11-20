"""
SWAT Saathi: watershed delineation and SWAT input builder for India.

Makes ready-to-load inputs for SWAT 2012 (ArcSWAT / QSWAT) and SWAT+ (QSWAT+ / SWAT+ Editor):
  - maps: elevation (DEM), land use and soil, on one UTM grid, with lookup tables
  - a soil database (usersoil) for the soil types in your area
  - weather: IMD rainfall and temperature, NASA POWER humidity, solar radiation and
    wind, and the weather generator table

Layout (like the IMD Data Puller app):
  left   - the control panel: what to make, period, stations, options
  right  - the map: your places, the safety margin (buffer) around them, the IMD grid
  below  - the results: map previews, soil types, weather stations and the download

Run it with:   streamlit run app.py
"""

import datetime
import json
import os

import altair as alt
import folium
import numpy as np
import pandas as pd
import shapely
import streamlit as st
from folium.plugins import Draw, Fullscreen, Geocoder, MousePosition
from shapely.geometry import Point, box, mapping, shape
from streamlit_folium import st_folium

import bundle as B
import catchment as C
import imd_tools as T
import report as R
import rivers as RV
import setup_advice as SA
import study_area as A
import swat_maps as M
import swat_weather as S

st.set_page_config(page_title="SWAT Saathi · Watershed delineation and SWAT input builder for India", page_icon="🏞️", layout="wide",
                   initial_sidebar_state="collapsed")

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = datetime.date.today()
LAST_YEAR = TODAY.year - 1 if TODAY.month >= 6 else TODAY.year - 2   # latest year IMD has usually published
INDIA_CENTRE, INDIA_ZOOM = [22.5, 80.5], 5
MAX_KM2 = {30: 15000, 90: 150000}          # largest study area per map cell size (free server memory)
KIND_COLOURS = {"pcp": "#2a78d6", "tmp": "#eb6834"}
ss = st.session_state
ss.setdefault("places_by_mode", {"point": [], "shape": [], "delineate": []})   # each tab keeps its own places
ss.setdefault("area_mode", "📍 Point + area")
ss.setdefault("map_version", 0)
ss.setdefault("result", None)
ss.setdefault("f_years_choice", "1991 to 2020")
ss.setdefault("f_mode", "Every IMD grid cell is a station")
AREA_DEFAULTS = {"sa_shape_km": 5.0, "sa_use_margin": True, "sa_point_km": 10.0, "sa_point_shape": "Circle",
                 "sa_shape_style": "Follow the outline", "trace_snap": 1000, "trace_max": 300, "trace_fine": True}


def keep_settings():
    """Streamlit forgets a widget's value when the widget is not on screen (for example the point size
    while the Delineate tab is open). Saving each value again at the start of every run keeps it."""
    for k, v in AREA_DEFAULTS.items():
        ss[k] = ss[k] if k in ss else v


keep_settings()
ss.setdefault("traced_obj", {})      # catchment name -> the delineation (only the latest is kept, to save memory)
ss.setdefault("traced", {})          # catchment name -> how it was traced (for the methods text)

st.markdown("""
<style>
footer {visibility: hidden; height: 0;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1500px;}   /* start below Streamlit's top bar */
div[data-testid="stMetric"] {background: rgba(42,120,214,0.07); border: 1px solid rgba(128,140,155,0.28);
                             border-radius: 10px; padding: 10px 14px;}
div[data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.75;}
.panel-title {font-size: 0.8rem; letter-spacing: .04em; opacity: 0.65; margin: 0.6rem 0 0.1rem 0;}
.hint {font-size: 0.92rem; opacity: 0.8; margin-bottom: 0.3rem;}
.tagline {font-size: 1.05rem; opacity: 0.9; margin-top: -0.6rem;}
.step {font-size: 1.3rem; font-weight: 700; margin: 1.4rem 0 0.5rem 0; display: flex; align-items: center;}
.step-n {display: inline-flex; width: 1.9rem; height: 1.9rem; border-radius: 50%; background: #2a78d6;
         color: #ffffff; align-items: center; justify-content: center; margin-right: 0.6rem; font-size: 1.05rem;}
.step-sub {font-weight: 400; opacity: 0.6; font-size: 0.95rem; margin-left: 0.7rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def india_layers():
    with open(os.path.join(HERE, "india_boundary.geojson"), encoding="utf-8") as f:
        india = json.load(f)
    area = shapely.simplify(shape(india["geometry"]).buffer(0.05), 0.05)
    grids = {}
    for key, grid in (("rain", T.RAIN_GRID), ("tmax", T.TEMP_GRID)):
        x0, y0, x1, y1 = grid.bounds
        xs = np.arange(x0, x1 + 1e-9, grid.step)
        ys = np.arange(y0, y1 + 1e-9, grid.step)
        lines = ([shapely.LineString([(x, y0), (x, y1)]) for x in xs] +
                 [shapely.LineString([(x0, y), (x1, y)]) for y in ys])
        clipped = shapely.set_precision(shapely.intersection(shapely.MultiLineString(lines), area), 0.001)
        grids[key] = {"type": "Feature", "properties": {}, "geometry": mapping(clipped)}
    return india, grids


@st.cache_data(show_spinner=False, max_entries=500)
def preview_cells(wkb, var):
    from shapely import wkb as W
    return T.cells_for_site(T.Site("x", W.loads(wkb)), var)


@st.cache_data(show_spinner=False, max_entries=1000)
def area_of(wkb):
    from shapely import wkb as W
    return T.area_km2(W.loads(wkb))


@st.cache_data(show_spinner=False, max_entries=200)
def study_area_cached(places_json, shape_km, point_km, point_shape, shape_style):
    # an outlet whose catchment was traced needs no circle of its own: the catchment's margin covers it
    sites = [T.Site(p["name"], shape(p["geojson"])) for p in json.loads(places_json) if not p.get("outlet")]
    area, epsg = A.study_area(sites, shape_km, point_km, point_shape, shape_style)
    return (area.wkb if area is not None else None), epsg


@st.cache_data(show_spinner=False, max_entries=10,
               hash_funcs={C.Traced: lambda t: (t.polygon.wkb, t.stream_km2)})
def delineation_zip(name, tr):
    import io as _io
    import zipfile as _zip
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w", _zip.ZIP_DEFLATED) as z:
        for f, b in C.delineation_files(tr, name).items():
            z.writestr(f, b)
    return buf.getvalue()


def dark_tiles():
    return folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ", name="Dark map", max_zoom=16)


def dark_mode():
    """True when the visitor is viewing the app in dark mode."""
    try:
        return (st.context.theme.type or "light") == "dark"
    except Exception:
        return False


MODES = {"📍 Point + area": "point", "🗺️ Your shape": "shape", "🌊 Delineate": "delineate"}
MODE_HELP = {
    "point": "Click 📍 on the map (or type coordinates below) to add points, such as a gauge or a site. Each point "
             "gets a circle or square of the size you choose.",
    "shape": "Upload your watershed or subbasins (shapefile, GeoJSON, GeoPackage or KML), or draw an outline ⬠ or a "
             "box ▭ on the map. A safety margin can be added around it.",
    "delineate": "Click 📍 on the map (or type coordinates below) to place your outlet or gauge, then press "
                 "Delineate: the app finds the catchment, streams and subbasins from the Copernicus DEM.",
}


def mode_key():
    return MODES.get(ss.get("area_mode"), "point")


def places_list():
    """The places of the tab that is open (each tab keeps its own)."""
    return ss.places_by_mode[mode_key()]


def current_sites():
    used = set()
    return [T.Site(T.clean_name(p["name"], used), shape(p["geojson"])) for p in places_list()]


def margin_km():
    """The safety margin around shapes, or 0 if switched off."""
    return float(ss.sa_shape_km) if ss.sa_use_margin else 0.0


def current_area(with_margin=True):
    """The download area (WGS 84 geometry) and its UTM EPSG code.
    with_margin=False gives 'your area': your shapes as they are, plus the areas around points."""
    pl = places_list()
    if not pl or all(p.get("outlet") for p in pl):
        return None, None
    if mode_key() == "delineate" and current_delineation()[1] is None:
        return None, None                      # nothing to download until the catchment is delineated
    from shapely import wkb as W
    wkb, epsg = study_area_cached(json.dumps(pl, sort_keys=True), margin_km() if with_margin else 0.0,
                                  float(ss.sa_point_km), ss.sa_point_shape,
                                  ss.sa_shape_style if with_margin else "Follow the outline")
    return (W.loads(wkb) if wkb else None), epsg


def add_place(name, geom, **extra):
    pl = places_list()
    used = {p["name"] for p in pl}
    pl.append({"name": T.clean_name(name, used), "geojson": mapping(geom), **extra})
    ss.map_version += 1


def refresh_map_area():
    from streamlit.errors import StreamlitInvalidLayoutContextError
    try:
        st.rerun(scope="fragment")
    except StreamlitInvalidLayoutContextError:
        st.rerun()


def next_name(kind):
    n = sum(1 for p in places_list() if p["name"].startswith(kind)) + 1
    return f"{kind} {n}"


def set_outlet(geom, name="Outlet"):
    """Delineate tab: a new outlet replaces the old one and clears the old delineation."""
    ss.places_by_mode["delineate"] = [{"name": name, "geojson": mapping(geom)}]
    ss.traced_obj, ss.traced = {}, {}
    ss.map_version += 1


def current_delineation():
    """(name, delineation) of the Delineate tab, or (None, None)."""
    if mode_key() != "delineate":
        return None, None
    names = {p["name"] for p in places_list()}
    for k, t in ss.traced_obj.items():
        if k in names:
            return k, t
    return None, None


class Tracker:
    """One progress bar for a job made of several steps.

    Each step has a rough expected length in seconds (its weight). The bar shows the step,
    what is being downloaded, the overall percentage and the time left.

    Time left = the rest of the current step (judged from how fast this step is going)
              + the steps still to come (their expected lengths, corrected by how much
                faster or slower the finished steps were than expected)."""

    def __init__(self, steps, where=None):
        import time
        self.steps = [(k, label, max(float(sec), 1.0)) for k, label, sec in steps]
        self.total = sum(sec for _, _, sec in self.steps)
        self.start = time.time()
        self.bar = (where or st).progress(0.0, text="Starting…")
        self.last = 0.0
        self.current, self.step_start = None, time.time()
        self.finished_est, self.finished_real = 0.0, 0.0

    def update(self, key, frac, message):
        import time
        now = time.time()
        keys = [k for k, _, _ in self.steps]
        if key not in keys:
            return
        n = keys.index(key)
        if key != self.current:                         # a new step has started
            if self.current is not None:
                prev = keys.index(self.current)
                self.finished_est += self.steps[prev][2]
                self.finished_real += now - self.step_start
            self.current, self.step_start = key, now
        sec = self.steps[n][2]
        sf = min(max(frac, 0.0), 1.0)
        done_before = sum(x[2] for x in self.steps[:n])
        f = max((done_before + sec * sf) / self.total, self.last)    # never go backwards
        self.last = f
        step_el = now - self.step_start
        speed = (self.finished_real / self.finished_est) if self.finished_est > 0 else 1.0
        speed = min(max(speed, 0.4), 3.0)
        cur_left = step_el * (1 - sf) / sf if (sf > 0.08 and step_el > 2) else sec * speed * (1 - sf)
        future = sum(x[2] for x in self.steps[n + 1:]) * speed
        left_s = cur_left + future
        left = "almost done" if f >= 0.995 or left_s < 3 else f"{nice_time(left_s)} left"
        self.bar.progress(f, text=f"Step {n + 1} of {len(self.steps)} · {self.steps[n][1]}: {message} · "
                                  f"**{f * 100:.0f}%** · {left}")

    def step(self, key):
        """A progress(fraction, message) function for one step."""
        return lambda f, m: self.update(key, f, m)

    def empty(self):
        self.bar.empty()


def nice_time(seconds):
    """Plain words for a duration: 'a few seconds', 'about 40 seconds', 'about 3 minutes'."""
    if seconds < 15:
        return "a few seconds"
    if seconds < 50:
        return f"about {int(round(seconds / 10.0) * 10)} seconds"
    if seconds < 90:
        return "about a minute"
    return f"about {round(seconds / 60)} minutes"


def cells_geojson(geom, var):
    h = T.VARIABLES[var]["grid"].step / 2
    feats = []
    cells = preview_cells(geom.wkb, var)
    for la, lo, fr in zip(cells["Latitude"], cells["Longitude"], cells["Fraction of cell inside"]):
        frac = float(fr) if np.isfinite(fr) else 1.0
        feats.append({"type": "Feature",
                      "properties": {"tip": f"IMD cell {la:.2f} N, {lo:.2f} E | {frac * 100:.0f}% inside",
                                     "op": round(0.08 + 0.35 * min(frac, 1.0), 3)},
                      "geometry": {"type": "Polygon", "coordinates": [[
                          [lo - h, la - h], [lo + h, la - h], [lo + h, la + h], [lo - h, la + h],
                          [lo - h, la - h]]]}})
    return {"type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------------------
# "Try an example"
# ---------------------------------------------------------------------------
if ss.pop("load_example", False):
    ss["area_mode"] = "🗺️ Your shape"
    ss.places_by_mode["shape"] = []
    add_place("Upper Netravati", box(75.30, 12.65, 75.65, 12.95))
    ss["f_years_choice"] = "Last 10 years"
    ss["f_mode"] = "Every IMD grid cell is a station"
    ss["sa_shape_km"], ss["sa_use_margin"] = 5.0, True


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
head_l, head_r = st.columns([3, 1])
head_l.markdown("## 🏞️ SWAT Saathi")
head_l.markdown("<div class='tagline'><b>Watershed delineation and SWAT input builder for India</b></div>"
                "<div class='hint'>Delineate your catchment, then get every SWAT input in one zip: elevation, land "
                "use and soil maps with lookup tables, a soil database, IMD + NASA POWER weather with the weather "
                "generator, and a PDF report. For <b>SWAT 2012</b> (ArcSWAT, QSWAT) and <b>SWAT+</b> "
                "(QSWAT+, SWAT+ Editor).</div>", unsafe_allow_html=True)
if head_r.button("✨ Try an example", width="stretch",
                 help="Loads a sample area in the upper Netravati basin, with the last 10 years of weather."):
    ss.load_example = True
    st.rerun()
THEMES = {"☀️ Light": "Light", "🌙 Dark": "Dark", "💻 System": "System"}
if "theme_choice" not in ss:
    ss.theme_choice = "🌙 Dark" if dark_mode() else "☀️ Light"
    ss.theme_applied = ss.theme_choice
choice = head_r.segmented_control("Theme", list(THEMES), key="theme_choice", width="stretch",
                                  label_visibility="collapsed",
                                  help="Light, dark, or follow your computer's setting (System).")
if choice and choice != ss.get("theme_applied"):
    ss.theme_applied = choice
    # Streamlit has no command to change the theme, so this presses its own theme button
    # (in the menu) for the visitor, then closes the menu again. Same as the IMD Data Puller.
    st.html(f"""<script>
    (async () => {{
      const wait = ms => new Promise(r => setTimeout(r, ms));
      const doc = document;
      const menuBtn = doc.querySelector('[data-testid="stMainMenu"] button') || doc.querySelector('[data-testid="stMainMenu"]');
      if (!menuBtn) return;
      const item = () => doc.querySelector('[data-testid="stMainMenuItem-theme-{THEMES[choice]}"]');
      const isOpen = () => !!doc.querySelector('[data-testid^="stMainMenuItem"]');
      if (!isOpen()) menuBtn.click();
      for (let i = 0; i < 40 && !item(); i++) await wait(25);
      if (item()) item().click();
      await wait(60);
      for (let i = 0; i < 5 && isOpen(); i++) {{
        const el = doc.querySelector('[data-testid^="stMainMenuItem"]') || doc.body;
        el.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Escape', code: 'Escape', keyCode: 27, bubbles: true}}));
        await wait(60);
        if (isOpen()) {{ menuBtn.click(); await wait(80); }}
      }}
    }})();
    </script>""", unsafe_allow_javascript=True)

data_dir_default = os.environ.get("IMD_DATA_DIR", "imd-data")
backup_dir = os.environ.get("IMD_BACKUP_DIR") or None


def step_title(n, text, sub=""):
    st.markdown(f"<div class='step'><span class='step-n'>{n}</span>{text}"
                + (f"<span class='step-sub'>{sub}</span>" if sub else "") + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------
SUB_COLOURS = ["#e57373", "#64b5f6", "#81c784", "#ffb74d", "#ba68c8", "#4db6ac", "#f06292", "#aed581",
               "#ffd54f", "#7986cb", "#a1887f", "#4dd0e1"]


def build_map(sites, area, grid_var, base, mode):
    india, grids = india_layers()
    m = folium.Map(location=INDIA_CENTRE, zoom_start=INDIA_ZOOM, tiles=None, control_scale=True,
                   min_zoom=4, prefer_canvas=True)
    if base == "Light map":
        folium.TileLayer("Esri.WorldGrayCanvas", name="Light map").add_to(m)
    elif base == "Dark map":
        dark_tiles().add_to(m)
    elif base == "Satellite":
        folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    on_dark = base in ("Satellite", "Dark map")
    line = "#ffffff" if base == "Satellite" else ("#aab2bd" if base == "Dark map" else "#8a8f98")
    if grid_var:
        folium.GeoJson(grids[grid_var], name="IMD grid", interactive=False,
                       style_function=lambda _: {"color": line, "weight": 0.5, "opacity": 0.45}).add_to(m)
    folium.GeoJson(india, name="India boundary", interactive=False,
                   style_function=lambda _: {"color": "#e6e9ee" if on_dark else "#3d4652",
                                             "weight": 1.3, "fill": False}).add_to(m)
    if area is not None:
        if grid_var:
            gj = cells_geojson(area, grid_var)
            if 0 < len(gj["features"]) <= 20000:
                folium.GeoJson(gj, name="IMD cells used",
                               style_function=lambda f: {"color": "#2a78d6", "weight": 0.6, "fillColor": "#2a78d6",
                                                         "fillOpacity": f["properties"]["op"]},
                               tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False)).add_to(m)
        folium.GeoJson(mapping(area), name="Download area (your area + safety margin)",
                       tooltip="Download area: your area + safety margin",
                       style_function=lambda _: {"color": "#1f9e89", "weight": 2.4, "dashArray": "7 5",
                                                 "fillOpacity": 0.0}).add_to(m)
    dname, tr = current_delineation()
    if tr is not None:
        net = tr.net
        fg = folium.FeatureGroup(name="Subbasins")
        for row in net.subbasins:
            col = SUB_COLOURS[row["Subbasin"] % len(SUB_COLOURS)]
            folium.GeoJson(mapping(row["geometry"]),
                           style_function=lambda _, c=col: {"color": "#ffffff" if on_dark else "#5f6b7a",
                                                            "weight": 0.8, "fillColor": c, "fillOpacity": 0.28},
                           tooltip=(f"Subbasin {row['Subbasin']} · {row['Area (km²)']:,.2f} km² · flows into "
                                    + ("the outlet" if row["Flows into"] == 0 else f"subbasin {row['Flows into']}"))
                           ).add_to(fg)
        fg.add_to(m)
        fs = folium.FeatureGroup(name="Streams")
        for g, order, ln, sub in net.streams:
            folium.PolyLine([(y, x) for x, y in g.coords], color="#1565c0", weight=0.8 + 0.9 * order, opacity=0.95,
                            tooltip=f"Stream, order {order}, {ln:,.2f} km, in subbasin {sub}").add_to(fs)
        fs.add_to(m)
    for s in sites:
        if s.is_point:
            folium.CircleMarker([s.geometry.y, s.geometry.x], radius=7, color="#ffffff", weight=2,
                                fill=True, fill_color="#eb6834", fill_opacity=1, tooltip=s.name).add_to(m)
        else:
            folium.GeoJson(mapping(s.geometry), tooltip=f"Your area: {s.name}",
                           style_function=lambda _: {"color": "#eb6834", "weight": 2.6,
                                                     "fillOpacity": 0.0 if tr is not None else 0.06}).add_to(m)
    draw = {"marker": True, "rectangle": False, "polygon": False, "polyline": False, "circle": False,
            "circlemarker": False}
    if mode == "shape":
        draw.update({"marker": False, "rectangle": True, "polygon": {"allowIntersection": False}})
    Draw(position="topleft", draw_options=draw, edit_options={"edit": False, "remove": False}).add_to(m)
    Geocoder(position="topright", collapsed=True, add_marker=False).add_to(m)
    MousePosition(position="bottomleft", separator=" N,  ", num_digits=4, prefix="",
                  lat_formatter="function(n){return L.Util.formatNum(n,4)}",
                  lng_formatter="function(n){return L.Util.formatNum(n,4) + ' E'}").add_to(m)
    Fullscreen(position="topleft").add_to(m)
    if tr is not None:
        folium.LayerControl(collapsed=True).add_to(m)
    shapes = [s.geometry for s in sites] + ([area] if area is not None else [])
    if shapes:
        minx, miny, maxx, maxy = np.array([g.bounds for g in shapes]).T
        pad = 0.08 if (maxx.max() - minx.min()) < 1 else 0.2
        m.fit_bounds([[miny.min() - pad, minx.min() - pad], [maxy.max() + pad, maxx.max() + pad]])
    return m


# ---------------------------------------------------------------------------
# Step ①: your area (three ways), one map
# ---------------------------------------------------------------------------
def margin_controls():
    with st.container(border=True):
        b0, b1, b2 = st.columns([1.3, 0.8, 1.3])
        b0.toggle("Add a safety margin (recommended)", key="sa_use_margin",
                  help="Switch off only if your boundary was delineated from this same DEM and you want the "
                       "files cut exactly to it.")
        b1.number_input("Margin (km)", 0.5, 100.0, step=1.0, key="sa_shape_km", disabled=not ss.sa_use_margin)
        b2.radio("Margin shape", ["Follow the outline", "Rectangle"], key="sa_shape_style", horizontal=True,
                 disabled=not ss.sa_use_margin)
        with st.expander("Why add a safety margin?"):
            st.markdown(
                "**Your area** (orange) is what SWAT will model. The **download area** (green dashed line) is your "
                "area plus a margin, because:\n"
                "1. **Watershed delineation needs ground beyond the boundary.** QSWAT and ArcSWAT trace the edge "
                "and the streams from the DEM; if the DEM stops exactly at your boundary, the traced watershed can "
                "come out smaller or cut off.\n"
                "2. **Boundaries rarely match the DEM exactly.** A boundary from another source can be a few hundred "
                "metres off the divide this DEM finds; the margin keeps land that really drains to your outlet.\n"
                "3. **Weather stations near the edge.** IMD cells just outside the boundary are included, so edge "
                "subbasins get their true nearest station.\n\n"
                "**Is it required?** No, but it is safer. 5 km is usually enough (2–3 km for small catchments, "
                "5–10 km for large or flat ones).")


def coordinate_inputs(label, on_add):
    with st.expander(label):
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 0.7])
        pname = c1.text_input("Name", placeholder="e.g. Bantwal gauge", key=f"pn_{label}")
        plat = c2.number_input("Latitude", -90.0, 90.0, value=None, format="%.4f", placeholder="12.8900",
                               key=f"pla_{label}")
        plon = c3.number_input("Longitude", -180.0, 180.0, value=None, format="%.4f", placeholder="75.0350",
                               key=f"plo_{label}")
        c4.write("")
        if c4.button("Add", width="stretch", key=f"padd_{label}") and plat is not None and plon is not None:
            on_add(pname.strip(), Point(plon, plat))
            refresh_map_area()


def places_table(sites):
    table = pd.DataFrame({
        "Name": [s.name for s in sites],
        "Type": ["Point" if s.is_point else "Area" for s in sites],
        "Latitude": [round(s.centre[0], 4) for s in sites],
        "Longitude": [round(s.centre[1], 4) for s in sites],
        "Area (km²)": ["" if s.is_point else f"{area_of(s.geometry.wkb):,.1f}" for s in sites],
        "Remove": [False] * len(sites)})
    edited = st.data_editor(table, key=f"places_{mode_key()}_{ss.map_version}", hide_index=True, width="stretch",
                            disabled=["Type", "Latitude", "Longitude", "Area (km²)"],
                            height=min(250, 38 + 35 * len(sites)))
    a, b = st.columns(2)
    if a.button("Apply changes to the list", width="stretch"):
        pl = places_list()
        keep = [{**p, "name": str(row["Name"]).strip() or p["name"]}
                for p, (_, row) in zip(pl, edited.iterrows()) if not row["Remove"]]
        pl[:] = keep
        ss.map_version += 1
        refresh_map_area()
    if b.button("Clear all", width="stretch"):
        places_list().clear()
        ss.map_version += 1
        refresh_map_area()


def shape_upload():
    files = st.file_uploader(
        "Upload a watershed or subbasins: a zipped shapefile, the .shp .shx .dbf .prj parts together, or "
        "GeoJSON / GeoPackage / KML. Any coordinate system (QSWAT's UTM shapefiles too).",
        accept_multiple_files=True, key="boundary_upload",
        type=["zip", "shp", "shx", "dbf", "prj", "cpg", "geojson", "json", "gpkg", "kml"])
    if not files:
        return
    key = tuple((f.name, f.size) for f in files)
    if ss.get("upload_key") != key:
        try:
            gdf, warns = T.read_boundary_files([(f.name, f.getvalue()) for f in files])
            ss.upload = (gdf, warns, None)
        except Exception as err:
            ss.upload = (None, [], str(err))
        ss.upload_key = key
    gdf, warns, error = ss.upload
    for w in warns:
        st.warning(w)
    if error:
        st.error(error)
    if gdf is None:
        return
    cols = [c for c in gdf.columns if c != "geometry"]
    guess = T.guess_name_column(gdf)
    new_sites = T.sites_from_frame(gdf, None)
    c0, c1, c2 = st.columns([1.2, 1, 1])
    name_col = c0.selectbox("Which column holds the names?", cols or ["(none)"],
                            index=cols.index(guess) if guess in cols else 0)
    new_sites = T.sites_from_frame(gdf, name_col if cols else None)
    if len(new_sites) == 1:
        if c1.button("Add this shape", type="primary", width="stretch"):
            add_place(new_sites[0].name, new_sites[0].geometry)
            refresh_map_area()
    else:
        st.markdown(f"<div class='hint'>The file has <b>{len(new_sites)} shapes</b>. Keep them separate if they "
                    f"are subbasins (each can get its own weather station), or join them into one watershed.</div>",
                    unsafe_allow_html=True)
        if c1.button(f"Keep {len(new_sites)} separate", type="primary", width="stretch"):
            for s_ in new_sites:
                add_place(s_.name, s_.geometry)
            refresh_map_area()
        if c2.button("Join into one watershed", width="stretch"):
            add_place("Watershed", shapely.union_all([s_.geometry for s_ in new_sites]))
            refresh_map_area()


def delineate_controls(sites):
    outlet = next((s for s in sites if s.is_point), None)
    dname, tr = current_delineation()
    if outlet is None:
        st.info("Place your outlet: click 📍 on the map (left edge), then click on the river, or type its "
                "coordinates below.")
    elif tr is not None:
        st.markdown(f"<div class='hint'>Outlet: <b>{outlet.name}</b> · {outlet.geometry.y:.4f} N, "
                    f"{outlet.geometry.x:.4f} E (moved {tr.moved_m:,.0f} m onto the main river) · delineated at "
                    f"{tr.res} m</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        c1.markdown(f"<div class='hint' style='margin-top:1.8rem'>Outlet: <b>{outlet.name}</b> · "
                    f"{outlet.geometry.y:.4f} N, {outlet.geometry.x:.4f} E</div>", unsafe_allow_html=True)
        snap_m = c2.number_input("Look for the river within (m)", 90, 10000, step=100, key="trace_snap",
                                 help="The outlet moves to the cell with the largest drainage area within this "
                                      "distance. Make it bigger if your point is far from the river.")
        max_km = c3.number_input("Largest search square (km)", 50, 400, step=50, key="trace_max",
                                 help="The search starts with a 50 km square and doubles until the whole "
                                      "catchment fits, up to this size.")
        fine = c4.toggle("Sharp 30 m boundary", key="trace_fine",
                         help="Finds the catchment quickly at 90 m, then again at 30 m for a sharper outline. "
                              "Very big catchments stay at 90 m.")
        if tr is None and st.button("🌊 Delineate the catchment", type="primary", width="stretch"):
            trk = Tracker([("trace", "Delineating", 90)])
            try:
                t_ = C.trace_catchment(outlet.geometry.y, outlet.geometry.x, max_km=max_km / 2,
                                       snap_m=float(snap_m), fine=fine, say=trk.step("trace"))
                trk.empty()
                name = f"Catchment of {outlet.name}"
                ss.places_by_mode["delineate"] = [
                    {"name": outlet.name, "geojson": mapping(t_.outlet), "outlet": True},
                    {"name": name, "geojson": mapping(t_.polygon)}]
                ss.traced_obj = {name: t_}                       # keep only the latest (memory)
                ss.traced = {name: f"the outlet was moved {t_.moved_m:,.0f} m onto the main river and the "
                                   f"catchment covers {t_.area_km2:,.0f} km²."}
                ss.trace_msg = ("warning" if t_.cut_off else "success",
                                f"Delineated {t_.area_km2:,.0f} km² at {t_.res} m. The outlet moved "
                                f"{t_.moved_m:,.0f} m onto the main river. " + t_.note
                                + (" The catchment still touches the edge of the largest search square, so it may be "
                                   "cut off: raise the size and delineate again." if t_.cut_off else ""))
                ss.map_version += 1
                refresh_map_area()
            except Exception as err:
                trk.empty()
                st.error(f"Could not delineate the catchment: {err}")
    msg = ss.pop("trace_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    if tr is None:
        return
    # --- the delineation suite ---
    choices = C.threshold_choices(tr.area_km2)
    cur = min(choices, key=lambda v: abs(v - tr.stream_km2))
    new = st.select_slider("Stream threshold (km²): a stream starts where at least this much land drains to it. "
                           "Left = more, smaller subbasins; right = fewer, bigger ones.", options=choices, value=cur,
                           key=f"thr_{dname}", format_func=lambda v: f"{v:g} km²")
    if new != tr.stream_km2:
        tr.stream_km2 = float(new)
        ss.map_version += 1
        refresh_map_area()
    net = tr.net
    if net.note:
        st.warning(net.note)
    k = st.columns(5)
    k[0].metric("Catchment area", f"{tr.area_km2:,.0f} km²")
    k[1].metric("Subbasins", f"{len(net.subbasins)}")
    k[2].metric("Stream segments", f"{len(net.streams)}")
    k[3].metric("Longest flow path", f"{net.stats['Longest flow path']:,.1f} km")
    k[4].metric("Time of concentration", f"{net.stats['Time of concentration']:,.1f} h",
                help="Kirpich (1940): time for water from the far end to reach the outlet.")
    t1, t2 = st.tabs(["Subbasins", "Catchment numbers"])
    with t1:
        st.dataframe(C.subbasin_table(net), hide_index=True, width="stretch", height=260)
        st.caption("Numbered from the outlet (1) upstream. 'Flows into' is the subbasin downstream.")
    with t2:
        st.dataframe(C.numbers_table(net), hide_index=True, width="stretch", height=260)
    st.download_button(f"⬇️ Download the delineation ({len(net.subbasins)} subbasins): shapefiles, GeoTIFF flow "
                       f"grids, tables and GeoJSON", delineation_zip(dname, tr),
                       file_name=f"{dname}.zip".replace(" ", "-"), mime="application/zip", width="stretch",
                       type="primary", key="dl_delin")
    st.caption("Catchment, outlet, streams, subbasins and subbasin outlets as shapefiles (UTM, for QSWAT, ArcSWAT "
               "and QGIS); flow direction, flow accumulation and subbasin grids (GeoTIFF); the two tables; and one "
               "GeoJSON for Google Earth. Next: step ② makes the SWAT inputs for this catchment, and 'one station "
               "per subbasin' weather uses these subbasins.")
    if st.button("Start again with a new outlet"):
        ss.places_by_mode["delineate"] = []
        ss.traced_obj, ss.traced = {}, {}
        ss.map_version += 1
        refresh_map_area()


@st.fragment
def area_section():
    keep_settings()
    step_title("①", "Your area", "choose one way to define it")
    st.segmented_control("How do you want to define your area?", list(MODES), key="area_mode",
                         label_visibility="collapsed", width="stretch")
    if not ss.get("area_mode"):
        ss.area_mode = "📍 Point + area"
    mode = mode_key()
    st.markdown(f"<div class='hint'>{MODE_HELP[mode]}</div>", unsafe_allow_html=True)
    sites = current_sites()
    area, epsg = current_area()

    t1, t2 = st.columns([1, 1.35])
    show_grid = t1.toggle("Show the IMD grid", value=mode != "delineate", key=f"grid_on_{mode}")
    bases = ["Light map", "Dark map", "Street map", "Satellite"]
    if "base_choice" not in ss:
        ss.base_choice = "Dark map" if dark_mode() else "Light map"
    base = t2.radio("Base map", bases, horizontal=True, key="base_choice", label_visibility="collapsed")
    out = st_folium(build_map(sites, area, "rain" if show_grid else None, base, mode),
                    key=f"map_{mode}_{ss.map_version}", height=520, use_container_width=True,
                    returned_objects=["last_active_drawing"])
    drawing = (out or {}).get("last_active_drawing")
    if drawing and drawing.get("geometry"):
        sig = json.dumps(drawing["geometry"], sort_keys=True)
        if sig != ss.get("seen_drawing"):
            ss.seen_drawing = sig
            geom = shape(drawing["geometry"])
            if mode == "delineate" and geom.geom_type == "Point":
                set_outlet(geom)
            elif mode == "point" and geom.geom_type == "Point":
                add_place(next_name("Point"), geom)
            elif mode == "shape" and geom.geom_type != "Point":
                if abs(geom.area - geom.envelope.area) < 1e-9:
                    add_place(next_name("Box"), geom)
                else:
                    add_place(next_name("Area"), shapely.make_valid(geom))
            refresh_map_area()
    st.markdown("<div class='hint' style='font-size:0.85rem'>"
                "<span style='color:#eb6834;font-weight:700'>━ Orange</span>: your area · "
                "<span style='color:#1f9e89;font-weight:700'>╍ Green dashed</span>: download area (your area + "
                "safety margin) · <span style='color:#2a78d6;font-weight:700'>▪ Blue squares</span>: IMD cells"
                + (" · <span style='color:#1565c0;font-weight:700'>━ Blue lines</span>: streams · coloured patches: "
                   "subbasins" if mode == "delineate" else "") + "</div>", unsafe_allow_html=True)

    # --- the options of the open tab ---
    if mode == "point":
        c1, c2, _ = st.columns([0.9, 1.1, 1.5])
        c1.number_input("Size around each point (km)", 0.5, 200.0, step=1.0, key="sa_point_km",
                        help="For a circle, the radius. For a square, half its side.")
        c2.radio("Shape", ["Circle", "Square"], horizontal=True, key="sa_point_shape")
        coordinate_inputs("Add a point by typing its coordinates",
                          lambda n, g: add_place(n or next_name("Point"), g))
        if sites:
            places_table(sites)
    elif mode == "shape":
        shape_upload()
        margin_controls()
        if sites:
            places_table(sites)
    else:
        if current_delineation()[1] is None:
            coordinate_inputs("Type the outlet's coordinates instead", lambda n, g: set_outlet(g, n or "Outlet"))
        delineate_controls(sites)
        if current_delineation()[1] is not None:
            margin_controls()

    if area is not None:
        km2 = A.area_km2(area, epsg)
        own, _e = current_area(with_margin=False)
        own_km2 = A.area_km2(own, epsg) if own is not None else 0.0
        n_rain, n_temp = len(preview_cells(area.wkb, "rain")), len(preview_cells(area.wkb, "tmax"))
        extra = (f" · the margin adds {km2 - own_km2:,.0f} km²" if mode != "point" and ss.sa_use_margin
                 and km2 > own_km2 + 0.5 else "")
        st.success(f"**Your area: {own_km2:,.0f} km²** · **Download area: {km2:,.0f} km²**{extra} · UTM zone "
                   f"{epsg - 32600} N · about {n_rain} IMD rainfall and {n_temp} temperature cells. "
                   f"Next: step ② below.")


area_section()


# ---------------------------------------------------------------------------
# Step ②: what to make
# ---------------------------------------------------------------------------
step_title("②", "What to make", "maps, soil, weather and the report")
with st.form("controls", border=True):
    f1, f2, f3 = st.columns([1, 1.15, 1], gap="large")
    with f1:
        st.markdown("<div class='panel-title'>MAPS</div>", unsafe_allow_html=True)
        want_dem = st.checkbox("Elevation (DEM)", value=True, key="f_dem")
        want_lu = st.checkbox("Land use + lookup table", value=True, key="f_lu")
        want_soil = st.checkbox("Soil + lookup table + usersoil", value=True, key="f_soil")
        want_rivers = st.checkbox("River lines for burn-in", value=True, key="f_rivers",
                                  help="Real river lines from OpenStreetMap that QSWAT / ArcSWAT can burn into "
                                       "the DEM. Most useful in flat areas.")
        res = st.radio("Map cell size", [30, 90], horizontal=True, key="f_res",
                       format_func=lambda r: f"{r} m" + (" (detailed)" if r == 30 else " (big basins)"),
                       help=f"30 m for areas up to {MAX_KM2[30]:,} km², 90 m up to {MAX_KM2[90]:,} km².")
    with f2:
        st.markdown("<div class='panel-title'>WEATHER</div>", unsafe_allow_html=True)
        want_weather = st.checkbox("Weather files + weather generator", value=True, key="f_weather")
        years_choice = st.radio("Period", ["1991 to 2020", "Last 10 years", "Choose years"],
                                horizontal=True, key="f_years_choice")
        y1, y2 = st.columns(2)
        first_year = y1.number_input("First year", 1951, LAST_YEAR, value=2001, key="f_y1")
        last_year = y2.number_input("Last year", 1951, LAST_YEAR, value=LAST_YEAR, key="f_y2")
        mode_label = st.radio(
            "Stations", ["Every IMD grid cell is a station", "One station per shape or subbasin"], key="f_mode",
            help="Grid cells: every 0.25° rainfall and 1° temperature cell over the download area becomes a "
                 "station; SWAT gives each subbasin its nearest one.\n\nOne per shape or subbasin: each of your "
                 "shapes, or each delineated subbasin, gets one station at its centre with the area-weighted "
                 "average over it.")
        use_power = st.checkbox("Humidity, solar radiation and wind (NASA POWER)", value=True, key="f_power")
    with f3:
        st.markdown("<div class='panel-title'>FOR</div>", unsafe_allow_html=True)
        want_2012 = st.checkbox("SWAT 2012 (ArcSWAT, QSWAT)", value=True, key="f_2012")
        want_plus = st.checkbox("SWAT+ (QSWAT+, SWAT+ Editor)", value=True, key="f_plus")
        st.caption("Every download also has a PDF report, a methods paragraph, a health check and suggested "
                   "SWAT settings.")
    with st.expander("Advanced options"):
        a1, a2, a3 = st.columns(3)
        crop_code = a1.selectbox("SWAT code for cropland", list(M.CROP_CHOICES), format_func=M.CROP_CHOICES.get,
                                 key="f_crop", help="WorldCover has one 'Cropland' class. Pick the crop that "
                                                    "covers most of your area, or keep the generic AGRL.")
        tree_code = a2.selectbox("SWAT code for tree cover", list(M.TREE_CHOICES), format_func=M.TREE_CHOICES.get,
                                 key="f_tree")
        urban_code = a3.selectbox("SWAT code for built-up land", list(M.URBAN_CHOICES),
                                  format_func=M.URBAN_CHOICES.get, key="f_urban")
        river_kinds = a1.multiselect("River lines to include", list(RV.KINDS), default=["river"],
                                     format_func=RV.KINDS.get, key="f_river_kinds")
        lu_year = a2.radio("Land use year (ESA WorldCover)", [2021, 2020], horizontal=True, key="f_lu_year")
        blank = a3.checkbox("Blank out map cells outside the download area", value=True, key="f_blank")
        wet_mm = a1.number_input("A day counts as wet from (mm)", min_value=0.0, value=0.1, step=0.1, key="f_wet")
        hh = a2.number_input("Half-hour rain = largest daily rain x", min_value=0.05, max_value=1.0,
                             value=float(S.HALF_HOUR_FACTOR), step=0.005, format="%.3f", key="f_hh",
                             help="0.275 comes from IMD's rule P(t) = P(24 h) x (t/24)^(1/3).")
        data_dir = a3.text_input("IMD data folder", value=data_dir_default, key="f_data_dir")
    submitted = st.form_submit_button("Make my SWAT inputs", type="primary", width="stretch")
progress_slot = st.container()

mode = "grid" if mode_label.startswith("Every") else "places"


# ---------------------------------------------------------------------------
# Make everything
# ---------------------------------------------------------------------------
if years_choice == "1991 to 2020":
    start_year, end_year = 1991, 2020
elif years_choice == "Last 10 years":
    start_year, end_year = LAST_YEAR - 9, LAST_YEAR
else:
    start_year, end_year = int(first_year), int(last_year)

if submitted:
    sites = current_sites()
    area, epsg = current_area()
    problems = []
    dname_, dtr_ = current_delineation()
    if mode_key() == "delineate" and dtr_ is None:
        problems.append("In step ①, place your outlet and press 'Delineate the catchment' first (or choose another "
                        "way to define your area).")
    elif not sites or area is None:
        problems.append("Step ①: add your area on the map first.")
    if not (want_dem or want_lu or want_soil or want_weather or want_rivers):
        problems.append("Tick at least one thing to make.")
    if want_weather and not (want_2012 or want_plus):
        problems.append("Tick SWAT 2012, SWAT+, or both.")
    if want_weather and start_year > end_year:
        problems.append("The first year is after the last year.")
    if not problems:
        outside = T.check_inside_india(sites)
        if outside:
            problems.append("These are outside IMD's India grid: " + ", ".join(outside))
        km2 = A.area_km2(area, epsg)
        if (want_dem or want_lu or want_soil) and km2 > MAX_KM2[int(res)]:
            problems.append(f"The study area is {km2:,.0f} km², too big for {res} m maps on the free online "
                            f"server (up to {MAX_KM2[int(res)]:,} km²). Choose 90 m, a smaller margin, or run "
                            f"the app on your laptop.")
        if want_weather and mode == "grid" and len(preview_cells(area.wkb, "rain")) > 600:
            problems.append("That area covers more than 600 IMD rainfall cells, too many stations for the free "
                            "server. Choose 'One station per shape or subbasin'.")
    if problems:
        for p in problems:
            progress_slot.warning(p)
    else:
        km2_ = A.area_km2(area, epsg)
        years_ = range(start_year, end_year + 1)
        todo_ = [(v, y) for v in ("rain", "tmax", "tmin") for y in years_
                 if not T.is_available(data_dir, v, y, backup_dir)] if want_weather else []
        steps = []
        if want_rivers:
            steps.append(("rivers", "River lines", 8))
        if want_dem:
            steps.append(("dem", "Elevation", 8 + km2_ * (0.006 if int(res) == 30 else 0.002)))
        if want_lu:
            steps.append(("landuse", "Land use", 10 + km2_ * 0.01))
        if want_soil:
            steps.append(("soil", "Soil", 45 + km2_ * 0.01))
        if want_weather:
            steps.append(("weather", "Weather", 15 + (sum(1 for v, _ in todo_ if v == "rain") * 20
                                                      + sum(1 for v, _ in todo_ if v != "rain") * 4) / 3
                          + (20 if use_power else 0)))
        steps.append(("report", "Report", 8 + (5 if want_dem or want_lu or want_soil else 0)))
        trk = Tracker(steps, where=progress_slot)   # right under the "Make my SWAT inputs" button
        files, ms, ws, errors = {}, None, None, []
        try:
            river_lines, river_problem = None, None
            if want_rivers:
                trk.update("rivers", 0.1, "asking OpenStreetMap for river lines")
                try:
                    river_lines = RV.fetch_rivers(area, tuple(river_kinds or ["river"]), say=trk.step("rivers"))
                    if river_lines:
                        for f, b in RV.rivers_shapefile(river_lines, epsg).items():
                            files[f"Maps/{f}"] = b
                    else:
                        river_problem = "No river lines are mapped in OpenStreetMap for this area."
                except RV.RiverError as err:
                    river_problem = str(err)
                trk.update("rivers", 1.0, f"{len(river_lines or [])} river segments")
            if want_dem or want_lu or want_soil:
                ms = M.build_maps(area, epsg, int(res), want_dem, want_lu, want_soil, landuse_year=int(lu_year),
                                  crop_code=crop_code, tree_code=tree_code, urban_code=urban_code,
                                  blank_outside=blank, progress=lambda stage, f, m: trk.update(stage, f, m))
                files.update(M.map_files(ms, sites, area))
            if want_weather:
                if mode == "grid":
                    w_sites = [T.Site("Study area", area)]
                elif dtr_ is not None:                      # one station per delineated subbasin
                    w_sites = [T.Site(f"Subbasin {r_['Subbasin']}", r_["geometry"]) for r_ in dtr_.net.subbasins]
                else:
                    w_sites = [s_ for s_ in sites if not any(p.get("outlet") and p["name"] == s_.name
                                                             for p in places_list())]
                ws = S.build_weather(
                    w_sites, datetime.date(start_year, 1, 1), datetime.date(end_year, 12, 31), data_dir,
                    mode=mode, use_power=use_power, wet_mm=float(wet_mm), half_hour_factor=float(hh),
                    backup_dir=backup_dir, progress=trk.step("weather"))
                files.update(S.weather_files(ws, swat2012=want_2012, swatplus=want_plus))
            trk.update("report", 0.1, "writing the README")
            km2 = A.area_km2(area, epsg)
            own, _e = current_area(with_margin=False)
            own_km2 = A.area_km2(own, epsg) if own is not None else km2
            area_text = (f"Your area: {own_km2:,.1f} km2 ({', '.join(s.name for s in sites)}). "
                         f"Download area: {km2:,.1f} km2 = your area + "
                         + (f"a {margin_km():g} km safety margin ({ss.sa_shape_style.lower()})" if ss.sa_use_margin
                            else "no margin")
                         + f". UTM zone {epsg - 32600} N (EPSG:{epsg}). The margin gives QSWAT / ArcSWAT ground "
                           f"beyond the boundary for clean delineation; SWAT models only your area.")
            text = B.readme(area_text, maps=ms, weather=S.weather_summary(ws) if ws else None)
            shapes_ = [s for s in sites if not s.is_point]
            pts_ = [s for s in sites if s.is_point]
            outlets = {p["name"] for p in places_list() if p.get("outlet")}
            pts_ = [s for s in pts_ if s.name not in outlets]
            margin = " and ".join(
                ([f"a {margin_km():g} km safety margin around the boundary"
                  + (" (as a rectangle)" if ss.sa_shape_style == "Rectangle" else "")]
                 if shapes_ and ss.sa_use_margin else [])
                + ([f"a {ss.sa_point_km:g} km {ss.sa_point_shape.lower()} around each point"] if pts_ else []))
            traced = [v for k, v in ss.traced.items() if k in {s.name for s in sites}]
            info = {"area_km2": km2, "own_km2": own_km2, "epsg": epsg, "res": int(res), "lu_year": int(lu_year),
                    "places": f"{len(sites)} place{'s' if len(sites) > 1 else ''} ({', '.join(s.name for s in sites)})",
                    "margin": margin or "no margin", "traced": traced[0] if traced else None,
                    "rivers": len(river_lines) if river_lines else None,
                    "traced_res": next((t.res for k, t in ss.traced_obj.items() if k in {s.name for s in sites}), 90),
                    "subbasins": len(dtr_.net.subbasins) if dtr_ is not None else None,
                    "stream_km2": dtr_.stream_km2 if dtr_ is not None else None}
            trk.update("report", 0.3, "checking the data and working out suggestions")
            tr_obj = next((t for k, t in ss.traced_obj.items() if k in {s.name for s in sites}), None)
            adv = SA.build_advice(ms, ws, sites, river_lines if want_rivers else None, tr_obj)
            files["Report/Setup suggestions.txt"] = SA.advice_text(adv)
            files["Report/Health check.csv"] = adv["health"].to_csv(index=False)
            trk.update("report", 0.5, "writing the PDF data report")
            pdf = R.pdf_report(info, sites, area, ms, ws, river_lines, adv)
            files["Report/SWAT Saathi data report.pdf"] = pdf
            files["Report/Methods paragraph.txt"] = R.methods_file(info, ms, ws)
            for name_, tr_ in ss.traced_obj.items():
                if name_ in {s.name for s in sites}:
                    for f, b in C.delineation_files(tr_, name_).items():
                        files[f"Delineation/{name_.replace(' ', '-')}/{f}"] = b
            trk.update("report", 0.75, "packing the zip and drawing the map previews")
            ss.result = {"ms": ms, "ws": ws, "zip": B.build_zip(files, text), "km2": km2,
                         "previews": {k: M.preview_png(ms, k) for k in ("dem", "landuse", "soil")
                                      if ms is not None and getattr(ms, k) is not None},
                         "overlays": M.overlay_layers(ms) if ms is not None else {},
                         "area": area, "sites": sites, "rivers": river_lines, "river_problem": river_problem,
                         "pdf": pdf, "methods": R.methods_text(info, ms, ws), "advice": adv,
                         "files": files, "readme": text, "info": info, "traced": tr_obj, "epsg": epsg,
                         "own_km2": own_km2, "river_kinds": tuple(river_kinds or ["river"]),
                         "rivers_wanted": want_rivers}
            trk.update("report", 1.0, "done")
            trk.empty()
        except Exception as err:
            trk.empty()
            st.error(f"Something went wrong: {err}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def stations_map(ws):
    m = folium.Map(tiles=None, control_scale=True, prefer_canvas=True)
    if dark_mode():
        dark_tiles().add_to(m)
    else:
        folium.TileLayer("Esri.WorldGrayCanvas", name="Light map").add_to(m)
    for s in ws.imd.sites:
        if not s.is_point:
            folium.GeoJson(mapping(s.geometry), tooltip=s.name, interactive=False,
                           style_function=lambda _: {"color": "#e6e9ee" if dark_mode() else "#3d4652",
                                                     "weight": 1.6,
                                                     "fillOpacity": 0.03}).add_to(m)
    for kind, label, r in [("pcp", "Rainfall stations", 5),
                           ("tmp", "Temperature (and humidity, solar, wind) stations", 9)]:
        fg = folium.FeatureGroup(name=label)
        for st_ in ws.stations.get(kind, []):
            folium.CircleMarker([st_.lat, st_.lon], radius=r, color="#ffffff", weight=1.5, fill=True,
                                fill_color=KIND_COLOURS[kind], fill_opacity=0.95,
                                tooltip=f"{st_.id} · {st_.lat:.2f} N, {st_.lon:.2f} E · "
                                        f"{st_.elev:.0f} m").add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    la, lo = np.array([(s.lat, s.lon) for s in ws.all_stations()]).T
    m.fit_bounds([[la.min() - 0.3, lo.min() - 0.3], [la.max() + 0.3, lo.max() + 0.3]])
    return m


def viewer_map(r):
    """Everything on one web map: the three maps as layers, the study area, places and weather stations."""
    ws, area = r["ws"], r["area"]
    m = folium.Map(tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(m)
    if dark_mode():
        dark_tiles().add_to(m)
    else:
        folium.TileLayer("Esri.WorldGrayCanvas", name="Light map").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    labels = {"dem": "Elevation (DEM)", "landuse": "Land use", "soil": "Soil types"}
    first = next(iter(r["overlays"]), None)
    for k, (url, bounds) in r["overlays"].items():
        folium.raster_layers.ImageOverlay(url, bounds=bounds, name=labels[k], opacity=0.75,
                                          show=(k == first), interactive=False).add_to(m)
    folium.GeoJson(mapping(area), name="Download area (your area + safety margin)",
                   style_function=lambda _: {"color": "#ffffff", "weight": 2.4, "dashArray": "7 5",
                                             "fill": False}).add_to(m)
    tr_ = r.get("traced")
    if tr_ is not None and getattr(tr_, "flw", None) is not None:
        net = tr_.net
        fg = folium.FeatureGroup(name="Subbasins", show=False)
        for row in net.subbasins:
            folium.GeoJson(mapping(row["geometry"]), tooltip=f"Subbasin {row['Subbasin']}",
                           style_function=lambda _: {"color": "#ffffff", "weight": 1, "fill": False}).add_to(fg)
        fg.add_to(m)
        fs = folium.FeatureGroup(name="Delineated streams")
        for g, order, _ln, _sub in net.streams:
            folium.PolyLine([(y, x) for x, y in g.coords], color="#1565c0", weight=0.8 + 0.9 * order).add_to(fs)
        fs.add_to(m)
    if r.get("rivers"):
        fg = folium.FeatureGroup(name="River lines (OpenStreetMap)")
        for g, kind, nm in r["rivers"]:
            folium.PolyLine([(y, x) for x, y in g.coords], color="#4fc3f7", weight=2 if kind == "river" else 1,
                            tooltip=nm or kind.capitalize()).add_to(fg)
        fg.add_to(m)
    places = folium.FeatureGroup(name="Your area")
    for s in r["sites"]:
        if s.is_point:
            folium.CircleMarker([s.geometry.y, s.geometry.x], radius=6, color="#ffffff", weight=2, fill=True,
                                fill_color="#eb6834", fill_opacity=1, tooltip=s.name).add_to(places)
        else:
            folium.GeoJson(mapping(s.geometry), tooltip=s.name,
                           style_function=lambda _: {"color": "#eb6834", "weight": 2.2, "fill": False}).add_to(places)
    places.add_to(m)
    if ws is not None:
        for kind, label, rad in [("pcp", "Rainfall stations", 4), ("tmp", "Temperature stations", 7)]:
            fg = folium.FeatureGroup(name=label, show=False)
            for st_ in ws.stations.get(kind, []):
                folium.CircleMarker([st_.lat, st_.lon], radius=rad, color="#ffffff", weight=1.5, fill=True,
                                    fill_color=KIND_COLOURS[kind], fill_opacity=0.95,
                                    tooltip=f"{st_.id} · {st_.elev:.0f} m").add_to(fg)
            fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen(position="topleft").add_to(m)
    w, s_, e, n = area.bounds
    m.fit_bounds([[s_, w], [n, e]])
    return m


def wgn_chart(w):
    df = w.stats.reset_index()
    df["Month name"] = S.MONTHS
    base = alt.Chart(df).encode(x=alt.X("Month name:N", sort=S.MONTHS, title=None))
    bars = base.mark_bar(color="#2a78d6", opacity=0.75).encode(
        y=alt.Y("pcp_ave:Q", title="Average monthly rainfall (mm)"),
        tooltip=[alt.Tooltip("Month name:N", title="Month"), alt.Tooltip("pcp_ave:Q", title="Rainfall (mm)"),
                 alt.Tooltip("pcp_days:Q", title="Wet days")])
    tdf = df.melt(id_vars=["Month name"], value_vars=["tmp_max_ave", "tmp_min_ave"], var_name="which",
                  value_name="t")
    tdf["which"] = tdf["which"].map({"tmp_max_ave": "Average maximum", "tmp_min_ave": "Average minimum"})
    lines = alt.Chart(tdf).mark_line(point=True, strokeWidth=2.2).encode(
        x=alt.X("Month name:N", sort=S.MONTHS, title=None),
        y=alt.Y("t:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
        color=alt.Color("which:N", title=None, scale=alt.Scale(range=["#eb6834", "#f2a541"]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("Month name:N", title="Month"), alt.Tooltip("which:N", title="Temperature"),
                 alt.Tooltip("t:Q", title="°C", format=".1f")])
    return alt.layer(bars, lines).resolve_scale(y="independent").properties(height=300)


def soil_legend(ms):
    cols = M.soil_colours(ms)
    rows = "".join(
        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
        f"<span style='flex:none;width:14px;height:14px;border-radius:3px;background:{cols[c]};"
        f"border:1px solid #ccc'></span><span><b>{n}</b>: {w} ({p:.1f}%)</span></div>"
        for c, n, w, p in zip(ms.soil_units["Soil code"], ms.soil_units["Soil name"],
                              ms.soil_units["What it is"], ms.soil_units["Area (%)"]))
    return f"<div style='font-size:0.85rem'>{rows}</div>"


def landuse_legend(lk):
    rows = "".join(
        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
        f"<span style='width:14px;height:14px;border-radius:3px;background:{M.WORLDCOVER_COLOURS.get(v, '#999')};"
        f"border:1px solid #ccc'></span><span>{name} → <b>{code}</b> ({pct:.1f}%)</span></div>"
        for v, name, code, pct in zip(lk["Value"], lk["Land cover (ESA WorldCover)"], lk["SWAT code"],
                                      lk["Area (%)"]))
    return f"<div style='font-size:0.88rem'>{rows}</div>"


def retry_rivers(r):
    """Fetch only the river lines again and add them to the zip, report and map viewer."""
    with st.spinner("Asking OpenStreetMap for the river lines again…"):
        try:
            lines = RV.fetch_rivers(r["area"], r["river_kinds"])
        except RV.RiverError as err:
            st.error(str(err))
            return
    if not lines:
        r["river_problem"] = "No river lines are mapped in OpenStreetMap for this area."
        st.rerun()
    files = r["files"]
    for f, b in RV.rivers_shapefile(lines, r["epsg"]).items():
        files[f"Maps/{f}"] = b
    r["info"]["rivers"] = len(lines)
    r["advice"] = SA.build_advice(r["ms"], r["ws"], r["sites"], lines, r["traced"])
    files["Report/Setup suggestions.txt"] = SA.advice_text(r["advice"])
    files["Report/Health check.csv"] = r["advice"]["health"].to_csv(index=False)
    r["pdf"] = R.pdf_report(r["info"], r["sites"], r["area"], r["ms"], r["ws"], lines, r["advice"])
    files["Report/SWAT Saathi data report.pdf"] = r["pdf"]
    files["Report/Methods paragraph.txt"] = R.methods_file(r["info"], r["ms"], r["ws"])
    r["methods"] = R.methods_text(r["info"], r["ms"], r["ws"])
    r["zip"] = B.build_zip(files, r["readme"])
    r["rivers"], r["river_problem"] = lines, None
    st.toast(f"Added {len(lines)} river segments to the zip.")
    st.rerun()


def advice_section(adv):
    h = adv["health"]
    n_fix, n_check = int((h["Status"] == "Fix").sum()), int((h["Status"] == "Check").sum())
    if n_fix:
        st.error(f"{n_fix} thing(s) to fix and {n_check} to check before SWAT.")
    elif n_check:
        st.warning(f"Nothing to fix; {n_check} thing(s) worth a look.")
    else:
        st.success("Everything looks good.")
    icon = {"OK": "✅", "Check": "⚠️", "Fix": "❌"}
    show = h.copy()
    show["Status"] = show["Status"].map(lambda x: f"{icon.get(x, '')} {x}")
    st.dataframe(show, hide_index=True, width="stretch")

    st.markdown("#### Suggested settings")
    if "stream_km2" in adv:
        st.markdown(f"**Stream threshold (watershed delineation):** about **{adv['stream_km2']:g} km²**, which "
                    f"gives roughly 20 to 40 subbasins for this {adv['catchment_km2']:,.0f} km² catchment. "
                    f"Smaller gives more subbasins and more detail; bigger gives fewer.")
    hru = adv.get("hru", {})
    if hru:
        st.markdown("**HRU thresholds** (QSWAT / ArcSWAT 'Multiple HRUs'). SWAT drops land uses, soils and slope "
                    "classes smaller than these percentages and gives their area to the others. Each suggestion "
                    "moves at most 5% of the catchment. SWAT applies them inside each subbasin, so treat them "
                    "as a starting point.")
        cols = st.columns(len(hru))
        labels = {"landuse": "Land use", "soil": "Soil", "slope": "Slope"}
        for col, (k, v) in zip(cols, hru.items()):
            col.metric(f"{labels[k]} threshold", f"{v['pick']}%")
            if v.get("exempt"):
                col.caption("Keep even if small (exempt): " + ", ".join(v["exempt"]))
        for k, v in hru.items():
            with st.expander(f"What each {labels[k].lower()} threshold would drop"):
                st.dataframe(v["table"], hide_index=True, width="stretch")
        if "slope" in hru:
            sc = hru["slope"]["classes"]
            st.markdown(f"**Slope classes:** breaks at **{', '.join(f'{b:g}%' for b in sc['breaks'])}** "
                        f"(type `{sc['qswat']}` in QSWAT). Median slope {sc['median']:.1f}%; 90% of the land is "
                        f"below {sc['p90']:.1f}%. Each class holds a fair share of the catchment:")
            st.dataframe(sc["table"], hide_index=True)
    e = adv.get("elevation")
    if e:
        if e["use_bands"]:
            st.markdown(f"**Elevation bands: yes, {e['bands']} bands.** The catchment rises from "
                        f"{e['lowest']:,.0f} m to {e['highest']:,.0f} m ({e['relief']:,.0f} m of relief), so "
                        f"temperature and rain change with height. Set **TLAPS = {e['tlaps']} °C/km** "
                        f"({e['tlaps_note']}) and **PLAPS = {e['plaps']:g} mm/km** ({e['plaps_note']}).")
        else:
            st.markdown(f"**Elevation bands: not needed.** The relief is only {e['relief']:,.0f} m "
                        f"({e['lowest']:,.0f} to {e['highest']:,.0f} m).")
    st.caption("All of this is also in the zip (Report/Setup suggestions.txt, Report/Health check.csv) and in "
               "the PDF report.")


def tidy_problems(problems):
    """Many 'could not download year X' lines become one short line."""
    failed = [p for p in problems if p.startswith("Could not download")]
    rest = [p for p in problems if not p.startswith("Could not download")]
    if len(failed) > 2:
        import re
        years = sorted({int(y) for p in failed for y in re.findall(r"for (\d{4})", p)})
        span = f"{years[0]} to {years[-1]}" if years else ""
        rest.insert(0, f"{len(failed)} yearly IMD files could not be downloaded ({span}). IMD's server is "
                       f"sometimes busy: press the button again later and only the missing years will be "
                       f"fetched. Those days are -99 in the files for now.")
    else:
        rest = failed + rest
    return rest


@st.fragment
def results_section():
    r = ss.result
    if not r:
        return
    ms, ws = r["ms"], r["ws"]
    step_title("③", "Your results", "download, check and look around")
    d1, d2 = st.columns([1.3, 2])
    d1.download_button(f"⬇️ Download the zip ({len(r['zip']) / 1e6:.1f} MB)", r["zip"],
                       file_name="SWAT Saathi inputs.zip", mime="application/zip", type="primary",
                       width="stretch")
    d1.download_button("📄 Download the data report (PDF)", r["pdf"], file_name="SWAT Saathi data report.pdf",
                       mime="application/pdf", width="stretch")
    d2.markdown("<div class='hint'>The zip has a README with step-by-step loading instructions for QSWAT, "
                "QSWAT+, ArcSWAT and SWAT+ Editor. Unzip it to a short folder path with no spaces.</div>",
                unsafe_allow_html=True)

    c = st.columns(5)
    c[0].metric("Your area / download area", f"{r.get('own_km2', r['km2']):,.0f} / {r['km2']:,.0f} km²",
                help="Your area is what SWAT models. The download area adds the safety margin around it.")
    if ms is not None:
        c[1].metric("Map size", f"{ms.grid.width:,} × {ms.grid.height:,}")
        if ms.landuse_lookup is not None:
            c[2].metric("Land use classes", len(ms.landuse_lookup))
        if ms.soil_units is not None:
            c[3].metric("Soil types", len(ms.soil_units))
    if ws is not None:
        c[4].metric("Rain / temperature stations", f"{len(ws.stations.get('pcp', []))} / "
                                                   f"{len(ws.stations.get('tmp', []))}")

    if r.get("river_problem"):
        w1, w2 = st.columns([3, 1])
        w1.warning("River lines: " + r["river_problem"])
        if r.get("rivers_wanted") and w2.button("🔁 Try river lines again", width="stretch"):
            retry_rivers(r)
    for p in tidy_problems((ms.problems if ms else []) + (ws.problems if ws else [])):
        st.warning(p)
    for n in (ws.notes if ws else []):
        st.info(n)

    names = ["Map viewer", "Health check and setup tips"]
    if ms is not None:
        names += ["Maps", "Soil types"] if ms.soil_units is not None else ["Maps"]
    if ws is not None:
        names += ["Weather stations", "Weather generator", "Daily weather"]
    names.append("Methods paragraph")
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Methods paragraph"]:
        text, refs = r["methods"]
        st.markdown("<div class='hint'>Ready to paste into your thesis or paper; edit freely. It is also in the "
                    "zip (Report folder) and in the PDF report.</div>", unsafe_allow_html=True)
        st.code(text, language=None, wrap_lines=True)
        st.markdown("**References**")
        st.code("\n\n".join(refs), language=None, wrap_lines=True)

    with tabs["Health check and setup tips"]:
        advice_section(r["advice"])
    with tabs["Map viewer"]:
        st_folium(viewer_map(r), key="viewer_map", height=600, use_container_width=True, returned_objects=[])
        st.caption("Switch layers on and off with the box at the top right. Click the square at the top left "
                   "for full screen. Maps are shown slightly simplified here; the files in the zip have full detail.")
        if ms is not None and ms.landuse_lookup is not None:
            l1, l2 = st.columns(2)
            l1.markdown("**Land use**")
            l1.markdown(landuse_legend(ms.landuse_lookup), unsafe_allow_html=True)
            if ms.soil_units is not None:
                cols = M.soil_colours(ms)
                l2.markdown("**Soil types**")
                l2.markdown("<div style='font-size:0.88rem'>" + "".join(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
                    f"<span style='width:14px;height:14px;border-radius:3px;background:{cols[c]};"
                    f"border:1px solid #ccc'></span><span><b>{n}</b>: {w} ({p:.1f}%)</span></div>"
                    for c, n, w, p in zip(ms.soil_units["Soil code"], ms.soil_units["Soil name"],
                                          ms.soil_units["What it is"], ms.soil_units["Area (%)"]))
                    + "</div>", unsafe_allow_html=True)
    if "Maps" in tabs:
        with tabs["Maps"]:
            cols = st.columns(3)
            labels = {"dem": "Elevation", "landuse": "Land use", "soil": "Soil types"}
            for col, (k, png) in zip(cols, r["previews"].items()):
                col.markdown(f"**{labels[k]}**")
                col.image(png, width="stretch")
                if k == "landuse":
                    col.markdown(landuse_legend(ms.landuse_lookup), unsafe_allow_html=True)
                if k == "soil" and ms.soil_units is not None:
                    col.markdown(soil_legend(ms), unsafe_allow_html=True)
                if k == "dem":
                    v = ms.dem[ms.dem != M.DEM_NODATA]
                    if v.size:
                        col.caption(f"From {v.min():,.0f} m to {v.max():,.0f} m")
            st.caption(" · ".join(f"{k}: {v}" for k, v in ms.settings.items()))
    if "Soil types" in tabs:
        with tabs["Soil types"]:
            st.dataframe(ms.soil_units, hide_index=True, width="stretch")
            st.caption("Each soil type is the topsoil (0-30 cm) texture over the subsoil (30-100 cm) texture "
                       "from SoilGrids. Water holding (AWC) and Ksat: Saxton and Rawls (2006). USLE K: Williams "
                       "(1995). Hydrologic group: NRCS (2009), from the slowest layer. All rows are in "
                       "SoilDatabase/usersoil.csv.")
    if "Weather stations" in tabs:
        with tabs["Weather stations"]:
            st_folium(stations_map(ws), key="stations_map", height=440, use_container_width=True,
                      returned_objects=[])
            st.dataframe(S.station_table(ws), hide_index=True, width="stretch")
    if "Weather generator" in tabs:
        with tabs["Weather generator"]:
            pick = st.selectbox("Weather generator station", [w.id for w in ws.wgn],
                                format_func=lambda n: f"{n}  ({next(w.based_on for w in ws.wgn if w.id == n)})")
            w = next(x for x in ws.wgn if x.id == pick)
            st.altair_chart(wgn_chart(w), width="stretch")
            t = w.stats.copy()
            t.index = S.MONTHS
            t.columns = [f"{c[2]}  [{c[0]} / {c[1]}]" for c in S.WGN_COLUMNS]
            st.dataframe(t.T, width="stretch")
            if w.estimated:
                st.caption("Estimated, not measured: " + "; ".join(w.estimated) + ".")
    if "Daily weather" in tabs:
        with tabs["Daily weather"]:
            kind = st.radio("Show", list(S.KINDS), format_func=lambda k: S.KINDS[k]["name"], horizontal=True,
                            key="daily_kind")
            if ws.stations.get(kind):
                st.caption(f"{S.KINDS[kind]['name']} ({S.KINDS[kind]['unit']}), one column per station. "
                           f"Empty cells are written as -99 in the SWAT files.")
                st.dataframe(S.daily_table(ws, kind).round(S.DECIMALS[kind]), width="stretch", height=360)
            else:
                st.info("Not included in this download.")


results_section()

st.caption(f"Version: maps engine {getattr(M, 'VERSION', 'old (before the soil fix)')} · SWAT Saathi is an "
           f"independent tool, not the official SWAT software (SWAT is developed by Texas A&M AgriLife Research "
           f"and USDA-ARS).")
st.markdown("<br><div class='hint' style='font-size:0.8rem'>Data: Copernicus DEM (ESA); ESA WorldCover; "
            "SoilGrids 2.0 (ISRIC); IMD Pune gridded rainfall (Pai et al. 2014) and temperature (Srivastava "
            "et al. 2009); NASA POWER. IMD download engine from the "
            "<a href='https://doi.org/10.5281/zenodo.24352511'>IMD Data Puller</a>.</div>",
            unsafe_allow_html=True)
