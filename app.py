"""
SWAT Bharat: every SWAT input for any catchment in India, in one zip.

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

st.set_page_config(page_title="SWAT Bharat", page_icon="🏞️", layout="wide",
                   initial_sidebar_state="collapsed")

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = datetime.date.today()
LAST_YEAR = TODAY.year - 1 if TODAY.month >= 6 else TODAY.year - 2   # latest year IMD has usually published
INDIA_CENTRE, INDIA_ZOOM = [22.5, 80.5], 5
MAX_KM2 = {30: 15000, 90: 150000}          # largest study area per map cell size (free server memory)
KIND_COLOURS = {"pcp": "#2a78d6", "tmp": "#eb6834"}
ss = st.session_state
ss.setdefault("places", [])
ss.setdefault("map_version", 0)
ss.setdefault("result", None)
ss.setdefault("f_years_choice", "1991 to 2020")
ss.setdefault("f_mode", "Every IMD grid cell is a station")
ss.setdefault("sa_shape_km", 5.0)
ss.setdefault("sa_point_km", 10.0)
ss.setdefault("sa_point_shape", "Circle")
ss.setdefault("sa_shape_style", "Follow the outline")
ss.setdefault("traced_obj", {})      # catchment name -> Traced result (for the health check)
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
.tagline {font-size: 1.02rem; opacity: 0.9; margin-top: -0.6rem;}
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


def current_sites():
    used = set()
    return [T.Site(T.clean_name(p["name"], used), shape(p["geojson"])) for p in ss.places]


def current_area():
    """The study area (WGS 84 geometry) and its UTM EPSG code, from the places and buffer choices."""
    if not ss.places:
        return None, None
    from shapely import wkb as W
    wkb, epsg = study_area_cached(json.dumps(ss.places, sort_keys=True), float(ss.sa_shape_km),
                                  float(ss.sa_point_km), ss.sa_point_shape, ss.sa_shape_style)
    return (W.loads(wkb) if wkb else None), epsg


def add_place(name, geom):
    used = {p["name"] for p in ss.places}
    ss.places.append({"name": T.clean_name(name, used), "geojson": mapping(geom)})
    ss.map_version += 1


def refresh_map_area():
    from streamlit.errors import StreamlitInvalidLayoutContextError
    try:
        st.rerun(scope="fragment")
    except StreamlitInvalidLayoutContextError:
        st.rerun()


def next_name(kind):
    n = sum(1 for p in ss.places if p["name"].startswith(kind)) + 1
    return f"{kind} {n}"


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
    ss.places = []
    add_place("Upper Netravati", box(75.30, 12.65, 75.65, 12.95))
    add_place("Bantwal gauge", Point(75.035, 12.89))
    ss["f_years_choice"] = "Last 10 years"
    ss["f_mode"] = "Every IMD grid cell is a station"
    ss["sa_shape_km"], ss["sa_point_km"] = 5.0, 8.0


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
head_l, head_r = st.columns([3, 1])
head_l.markdown("## 🏞️ SWAT Bharat")
head_l.markdown("<div class='tagline'>Every SWAT input for any catchment in India, in one zip.</div>"
                "<div class='hint'>Trace your catchment from an outlet, then get elevation, land use and soil maps with "
                "lookup tables, a ready soil database, river lines for burn-in, IMD + NASA POWER weather with the "
                "weather generator, and a PDF report with a ready methods paragraph, for <b>SWAT 2012</b> "
                "(ArcSWAT, QSWAT) and <b>SWAT+</b> (QSWAT+, SWAT+ Editor).</div>", unsafe_allow_html=True)
if head_r.button("✨ Try an example", width="stretch",
                 help="Loads a sample area in the upper Netravati basin and a gauge point, with the last 10 years."):
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

panel, mapcol = st.columns([1, 2.6], gap="medium")
data_dir_default = os.environ.get("IMD_DATA_DIR", "imd-data")
backup_dir = os.environ.get("IMD_BACKUP_DIR") or None


# ---------------------------------------------------------------------------
# Left: the control panel
# ---------------------------------------------------------------------------
with panel:
    with st.form("controls", border=True):
        st.markdown("<div class='panel-title'>1 · WHAT TO MAKE</div>", unsafe_allow_html=True)
        want_dem = st.checkbox("Elevation map (DEM)", value=True, key="f_dem")
        want_lu = st.checkbox("Land use map + lookup table", value=True, key="f_lu")
        want_soil = st.checkbox("Soil map + lookup table + soil database (usersoil)", value=True, key="f_soil")
        want_rivers = st.checkbox("River lines for DEM burn-in (OpenStreetMap)", value=True, key="f_rivers",
                                  help="Real river lines that QSWAT / ArcSWAT can burn into the DEM, so the "
                                       "streams follow the real rivers. Most useful in flat areas.")
        want_weather = st.checkbox("Weather files + weather generator", value=True, key="f_weather")
        res = st.radio("Map cell size", [30, 90], horizontal=True, key="f_res",
                       format_func=lambda r: f"{r} m" + (" (detailed)" if r == 30 else " (big basins)"),
                       help=f"30 m for areas up to {MAX_KM2[30]:,} km², 90 m up to {MAX_KM2[90]:,} km².")
        o1, o2 = st.columns(2)
        want_2012 = o1.checkbox("SWAT 2012", value=True, key="f_2012")
        want_plus = o2.checkbox("SWAT+", value=True, key="f_plus")

        st.markdown("<div class='panel-title'>2 · WEATHER PERIOD</div>", unsafe_allow_html=True)
        years_choice = st.radio("Period", ["1991 to 2020", "Last 10 years", "Choose years"],
                                horizontal=True, key="f_years_choice", label_visibility="collapsed")
        y1, y2 = st.columns(2)
        first_year = y1.number_input("First year", 1951, LAST_YEAR, value=2001, key="f_y1")
        last_year = y2.number_input("Last year", 1951, LAST_YEAR, value=LAST_YEAR, key="f_y2")
        st.caption("Temperature starts in 1951; humidity, solar radiation and wind (NASA POWER) in 1981. "
                   "Add a few warm-up years before your calibration period.")

        st.markdown("<div class='panel-title'>3 · WEATHER STATIONS</div>", unsafe_allow_html=True)
        mode_label = st.radio(
            "Station layout", ["Every IMD grid cell is a station", "One station per shape"],
            key="f_mode", label_visibility="collapsed",
            help="Grid cells: every 0.25° rainfall cell and 1° temperature cell over the study area "
                 "(your places + buffer) becomes a station, and SWAT gives each subbasin its nearest one.\n\n"
                 "One per shape: upload your subbasins, and each gets one station at its centre holding "
                 "the area-weighted average over it.")
        use_power = st.checkbox("Humidity, solar radiation and wind from NASA POWER", value=True, key="f_power")

        with st.expander("Advanced options"):
            crop_code = st.selectbox("SWAT code for cropland", list(M.CROP_CHOICES),
                                     format_func=M.CROP_CHOICES.get, key="f_crop",
                                     help="WorldCover has one 'Cropland' class. Pick the crop that covers "
                                          "most of your area, or keep the generic AGRL.")
            tree_code = st.selectbox("SWAT code for tree cover", list(M.TREE_CHOICES),
                                     format_func=M.TREE_CHOICES.get, key="f_tree")
            urban_code = st.selectbox("SWAT code for built-up land", list(M.URBAN_CHOICES),
                                      format_func=M.URBAN_CHOICES.get, key="f_urban")
            river_kinds = st.multiselect("River lines to include", list(RV.KINDS), default=["river"],
                                         format_func=RV.KINDS.get, key="f_river_kinds",
                                         help="Streams and canals add detail but many are small or man-made.")
            lu_year = st.radio("Land use year (ESA WorldCover)", [2021, 2020], horizontal=True, key="f_lu_year")
            blank = st.checkbox("Blank out map cells outside the study area", value=True, key="f_blank",
                                help="Untick to keep the full rectangle around the study area.")
            wet_mm = st.number_input("A day counts as wet from (mm)", min_value=0.0, value=0.1, step=0.1,
                                     key="f_wet")
            hh = st.number_input("Half-hour rain = largest daily rain x", min_value=0.05, max_value=1.0,
                                 value=float(S.HALF_HOUR_FACTOR), step=0.005, format="%.3f", key="f_hh",
                                 help="0.275 comes from IMD's rule P(t) = P(24 h) x (t/24)^(1/3).")
            data_dir = st.text_input("IMD data folder", value=data_dir_default, key="f_data_dir")
        submitted = st.form_submit_button("Make my SWAT inputs", type="primary", width="stretch")

mode = "grid" if mode_label.startswith("Every") else "places"


# ---------------------------------------------------------------------------
# Right: the map, the places and the study area
# ---------------------------------------------------------------------------
def build_map(sites, area, grid_var, base):
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
    folium.GeoJson(grids[grid_var], name="IMD grid", interactive=False,
                   style_function=lambda _: {"color": line, "weight": 0.5, "opacity": 0.45}).add_to(m)
    folium.GeoJson(india, name="India boundary", interactive=False,
                   style_function=lambda _: {"color": "#e6e9ee" if on_dark else "#3d4652",
                                             "weight": 1.3, "fill": False}).add_to(m)
    if area is not None:
        gj = cells_geojson(area, grid_var)
        if 0 < len(gj["features"]) <= 20000:
            folium.GeoJson(gj, name="IMD cells used",
                           style_function=lambda f: {"color": "#2a78d6", "weight": 0.6, "fillColor": "#2a78d6",
                                                     "fillOpacity": f["properties"]["op"]},
                           tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False)).add_to(m)
        folium.GeoJson(mapping(area), name="Study area (with buffer)", tooltip="Study area (your places + buffer)",
                       style_function=lambda _: {"color": "#1f9e89", "weight": 2.4, "dashArray": "7 5",
                                                 "fillColor": "#1f9e89", "fillOpacity": 0.06}).add_to(m)
    for s in sites:
        if s.is_point:
            folium.CircleMarker([s.geometry.y, s.geometry.x], radius=6, color="#ffffff", weight=2,
                                fill=True, fill_color="#eb6834", fill_opacity=1, tooltip=s.name).add_to(m)
        else:
            folium.GeoJson(mapping(s.geometry), tooltip=s.name,
                           style_function=lambda _: {"color": "#eb6834", "weight": 2.2,
                                                     "fillOpacity": 0.04}).add_to(m)
    Draw(position="topleft",
         draw_options={"marker": True, "rectangle": True, "polygon": {"allowIntersection": False},
                       "polyline": False, "circle": False, "circlemarker": False},
         edit_options={"edit": False, "remove": False}).add_to(m)
    Geocoder(position="topright", collapsed=True, add_marker=False).add_to(m)
    MousePosition(position="bottomleft", separator=" N,  ", num_digits=4, prefix="",
                  lat_formatter="function(n){return L.Util.formatNum(n,4)}",
                  lng_formatter="function(n){return L.Util.formatNum(n,4) + ' E'}").add_to(m)
    Fullscreen(position="topleft").add_to(m)
    shapes = [s.geometry for s in sites] + ([area] if area is not None else [])
    if shapes:
        minx, miny, maxx, maxy = np.array([g.bounds for g in shapes]).T
        pad = 0.15
        m.fit_bounds([[miny.min() - pad, minx.min() - pad], [maxy.max() + pad, maxx.max() + pad]])
    return m


@st.fragment
def map_section():
    sites = current_sites()

    # --- the safety margin around the places ---
    with st.container(border=True):
        st.markdown("<div class='panel-title' style='margin-top:0'>STUDY AREA · YOUR PLACES + A SAFETY MARGIN"
                    "</div>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns([1, 1.25, 1, 1.25])
        b1.number_input("Margin around shapes (km)", 0.0, 100.0, step=1.0, key="sa_shape_km",
                        help="Maps and weather are fetched a little beyond your catchment, so QSWAT / ArcSWAT "
                             "can delineate streams cleanly at the edges. 5 km is a safe default.")
        b2.radio("Margin shape", ["Follow the outline", "Rectangle"], key="sa_shape_style")
        b3.number_input("Size around points (km)", 0.5, 200.0, step=1.0, key="sa_point_km",
                        help="For a circle, the radius. For a square, half its side (the distance from the "
                             "point to each edge).")
        b4.radio("Point area shape", ["Circle", "Square"], horizontal=True, key="sa_point_shape")

    area, epsg = current_area()
    t1, t2 = st.columns([1, 1.35])
    grid_label = t1.radio("Grid", ["Rainfall grid (0.25°)", "Temperature grid (1°)"], horizontal=True,
                          key="grid_choice", label_visibility="collapsed")
    bases = ["Light map", "Dark map", "Street map", "Satellite"]
    if "base_choice" not in ss:
        ss.base_choice = "Dark map" if dark_mode() else "Light map"
    base = t2.radio("Base map", bases, horizontal=True, key="base_choice", label_visibility="collapsed")
    grid_var = "rain" if grid_label.startswith("Rain") else "tmax"
    if not sites:
        st.markdown("<div class='hint'>Upload your watershed or subbasins below (the best way), or use the map "
                    "tools on the left edge: 📍 a point (e.g. your outlet), ⬠ an outline or ▭ a box. "
                    "Search a place with 🔍.</div>", unsafe_allow_html=True)
    out = st_folium(build_map(sites, area, grid_var, base), key=f"map_{ss.map_version}", height=520,
                    use_container_width=True, returned_objects=["last_active_drawing"])
    drawing = (out or {}).get("last_active_drawing")
    if drawing and drawing.get("geometry"):
        sig = json.dumps(drawing["geometry"], sort_keys=True)
        if sig != ss.get("seen_drawing"):
            ss.seen_drawing = sig
            geom = shape(drawing["geometry"])
            if geom.geom_type == "Point":
                add_place(next_name("Point"), geom)
            elif abs(geom.area - geom.envelope.area) < 1e-9:
                add_place(next_name("Box"), geom)
            else:
                add_place(next_name("Area"), shapely.make_valid(geom))
            refresh_map_area()

    if sites:
        km2 = A.area_km2(area, epsg)
        n_rain = len(preview_cells(area.wkb, "rain"))
        n_temp = len(preview_cells(area.wkb, "tmax"))
        st.markdown(f"**Study area: {km2:,.0f} km²** (green dashed line) · UTM zone {epsg - 32600} N · "
                    f"about **{n_rain}** IMD rainfall and **{n_temp}** temperature cells")
        table = pd.DataFrame({
            "Name": [s.name for s in sites],
            "Type": ["Point" if s.is_point else "Area" for s in sites],
            "Latitude": [round(s.centre[0], 4) for s in sites],
            "Longitude": [round(s.centre[1], 4) for s in sites],
            "Area (km²)": ["" if s.is_point else f"{area_of(s.geometry.wkb):,.1f}" for s in sites],
            "Remove": [False] * len(sites)})
        edited = st.data_editor(table, key=f"places_{ss.map_version}", hide_index=True, width="stretch",
                                disabled=["Type", "Latitude", "Longitude", "Area (km²)"],
                                height=min(280, 38 + 35 * len(sites)))
        a, b = st.columns([1, 1])
        if a.button("Apply changes to the list", width="stretch"):
            keep = []
            for p, (_, row) in zip(ss.places, edited.iterrows()):
                if not row["Remove"]:
                    keep.append({**p, "name": str(row["Name"]).strip() or p["name"]})
            ss.places = keep
            ss.map_version += 1
            refresh_map_area()
        if b.button("Clear all places", width="stretch"):
            ss.places = []
            ss.map_version += 1
            refresh_map_area()

    with st.expander("Upload a watershed or subbasin boundary (shapefile, GeoJSON, GeoPackage or KML)",
                     expanded=not sites):
        files = st.file_uploader(
            "Zipped shapefile, or the .shp .shx .dbf .prj parts together, or GeoJSON / GeoPackage / KML. "
            "Any coordinate system is fine (QSWAT's UTM shapefiles too).",
            accept_multiple_files=True, key="boundary_upload",
            type=["zip", "shp", "shx", "dbf", "prj", "cpg", "geojson", "json", "gpkg", "kml"])
        if files:
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
            if gdf is not None:
                cols = [c for c in gdf.columns if c != "geometry"]
                guess = T.guess_name_column(gdf)
                name_col = st.selectbox("Which column holds the names?", cols or ["(none)"],
                                        index=cols.index(guess) if guess in cols else 0)
                new_sites = T.sites_from_frame(gdf, name_col if cols else None)
                c1, c2 = st.columns(2)
                if c1.button(f"Add these {len(new_sites)} shape(s)", type="primary", width="stretch",
                             help="Keeps each shape separate (use this for subbasins)."):
                    for s in new_sites:
                        add_place(s.name, s.geometry)
                    refresh_map_area()
                if len(new_sites) > 1 and c2.button("Add them as one watershed", width="stretch",
                                                    help="Joins all shapes into one outline."):
                    add_place("Watershed", shapely.union_all([s.geometry for s in new_sites]))
                    refresh_map_area()

    points = [s for s in sites if s.is_point]
    with st.expander("🌊 Trace a catchment from an outlet point", expanded=bool(points) and not
                     any(not s.is_point for s in sites)):
        st.markdown("<div class='hint'>Add your outlet or gauge as a point (map tool 📍 or typed below), "
                    "then trace everything that drains to it from the Copernicus 90 m DEM. The point is moved "
                    "onto the biggest river nearby.</div>", unsafe_allow_html=True)
        if not points:
            st.info("Add a point first.")
        else:
            c1, c2, c3, c4 = st.columns([1.3, 1, 1, 0.9])
            pick = c1.selectbox("Outlet point", [s.name for s in points], key="trace_pick")
            snap_m = c2.number_input("Look for the river within (m)", 90, 10000, 1000, step=100, key="trace_snap",
                                     help="The point moves to the cell with the largest drainage area within "
                                          "this distance. Make it bigger if your point is far from the river.")
            max_km = c3.number_input("Largest search square (km)", 50, 400, 300, step=50, key="trace_max",
                                     help="The search starts with a 50 km square and doubles until the whole "
                                          "catchment fits, up to this size.")
            fine = c4.checkbox("Sharp 30 m boundary", value=True, key="trace_fine",
                               help="Finds the catchment quickly at 90 m, then re-traces it at 30 m for a sharper "
                                    "outline. Very big catchments stay at 90 m.")
            if st.button("Trace the catchment", type="primary", width="stretch"):
                pt = next(s for s in points if s.name == pick)
                if True:
                    trk = Tracker([("trace", "Tracing the catchment", 90)])
                    try:
                        tr = C.trace_catchment(pt.geometry.y, pt.geometry.x, max_km=max_km / 2,
                                               snap_m=float(snap_m), fine=fine, say=trk.step("trace"))
                        trk.empty()
                        name = f"Catchment of {pick}"
                        for p in ss.places:                      # move the point onto the river
                            if p["name"] == pick:
                                p["geojson"] = mapping(tr.outlet)
                                p["outlet"] = True
                        add_place(name, tr.polygon)
                        ss.traced_obj[name] = tr
                        ss.traced[name] = (f"the outlet was moved {tr.moved_m:,.0f} m onto the main river and the "
                                           f"traced catchment covers {tr.area_km2:,.0f} km².")
                        ss.trace_msg = ("success" if not tr.cut_off else "warning",
                                        f"Traced {tr.area_km2:,.0f} km² at {tr.res} m. The outlet moved "
                                        f"{tr.moved_m:,.0f} m onto the river. " + tr.note + (" The catchment still touches the edge of the "
                                        "largest search square, so it may be cut off: raise the size or upload "
                                        "the boundary instead." if tr.cut_off else ""))
                        refresh_map_area()
                    except Exception as err:
                        trk.empty()
                        st.error(f"Could not trace the catchment: {err}")
        msg = ss.pop("trace_msg", None)
        if msg:
            getattr(st, msg[0])(msg[1])

    with st.expander("Add a point by typing its coordinates (e.g. an outlet or gauge)"):
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 0.7])
        pname = c1.text_input("Name", placeholder="e.g. Bantwal gauge")
        plat = c2.number_input("Latitude", -90.0, 90.0, value=None, format="%.4f", placeholder="12.8900")
        plon = c3.number_input("Longitude", -180.0, 180.0, value=None, format="%.4f", placeholder="75.0350")
        c4.write("")
        if c4.button("Add", width="stretch") and plat is not None and plon is not None:
            add_place(pname.strip() or next_name("Point"), Point(plon, plat))
            refresh_map_area()


with mapcol:
    map_section()


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
    if not sites:
        problems.append("Add your watershed, subbasins or a point on the map first.")
    if not (want_dem or want_lu or want_soil or want_weather or want_rivers):
        problems.append("Tick at least one thing to make.")
    if want_weather and not (want_2012 or want_plus):
        problems.append("Tick SWAT 2012, SWAT+, or both.")
    if want_weather and start_year > end_year:
        problems.append("The first year is after the last year.")
    if sites:
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
                            "server. Upload subbasins and choose 'One station per shape'.")
    if problems:
        for p in problems:
            panel.warning(p)
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
        trk = Tracker(steps, where=panel)          # right under the "Make my SWAT inputs" button
        files, ms, ws, errors = {}, None, None, []
        try:
            river_lines, river_problem = None, None
            if want_rivers:
                trk.update("rivers", 0.1, "asking OpenStreetMap for river lines")
                try:
                    river_lines = RV.fetch_rivers(area, tuple(river_kinds or ["river"]))
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
                w_sites = [T.Site("Study area", area)] if mode == "grid" else sites
                ws = S.build_weather(
                    w_sites, datetime.date(start_year, 1, 1), datetime.date(end_year, 12, 31), data_dir,
                    mode=mode, use_power=use_power, wet_mm=float(wet_mm), half_hour_factor=float(hh),
                    backup_dir=backup_dir, progress=trk.step("weather"))
                files.update(S.weather_files(ws, swat2012=want_2012, swatplus=want_plus))
            trk.update("report", 0.1, "writing the README")
            km2 = A.area_km2(area, epsg)
            area_text = (f"{km2:,.1f} km2, UTM zone {epsg - 32600} N (EPSG:{epsg}). Places: "
                         + ", ".join(s.name for s in sites)
                         + f". Margin: {ss.sa_shape_km:g} km around shapes ({ss.sa_shape_style.lower()}), "
                         f"{ss.sa_point_km:g} km {ss.sa_point_shape.lower()} around points.")
            text = B.readme(area_text, maps=ms, weather=S.weather_summary(ws) if ws else None)
            shapes_ = [s for s in sites if not s.is_point]
            pts_ = [s for s in sites if s.is_point]
            outlets = {p["name"] for p in ss.places if p.get("outlet")}
            pts_ = [s for s in pts_ if s.name not in outlets]
            margin = " and ".join(
                ([f"a {ss.sa_shape_km:g} km margin around the boundary"
                  + (" (as a rectangle)" if ss.sa_shape_style == "Rectangle" else "")] if shapes_ else [])
                + ([f"a {ss.sa_point_km:g} km {ss.sa_point_shape.lower()} around each point"] if pts_ else []))
            traced = [v for k, v in ss.traced.items() if k in {s.name for s in sites}]
            info = {"area_km2": km2, "epsg": epsg, "res": int(res), "lu_year": int(lu_year),
                    "places": f"{len(sites)} place{'s' if len(sites) > 1 else ''} ({', '.join(s.name for s in sites)})",
                    "margin": margin or "no margin", "traced": traced[0] if traced else None,
                    "rivers": len(river_lines) if river_lines else None,
                    "traced_res": next((t.res for k, t in ss.traced_obj.items() if k in {s.name for s in sites}), 90)}
            trk.update("report", 0.3, "checking the data and working out suggestions")
            tr_obj = next((t for k, t in ss.traced_obj.items() if k in {s.name for s in sites}), None)
            adv = SA.build_advice(ms, ws, sites, river_lines if want_rivers else None, tr_obj)
            files["Report/Setup suggestions.txt"] = SA.advice_text(adv)
            files["Report/Health check.csv"] = adv["health"].to_csv(index=False)
            trk.update("report", 0.5, "writing the PDF data report")
            pdf = R.pdf_report(info, sites, area, ms, ws, river_lines, adv)
            files["Report/SWAT Bharat data report.pdf"] = pdf
            files["Report/Methods paragraph.txt"] = R.methods_file(info, ms, ws)
            trk.update("report", 0.75, "packing the zip and drawing the map previews")
            ss.result = {"ms": ms, "ws": ws, "zip": B.build_zip(files, text), "km2": km2,
                         "previews": {k: M.preview_png(ms, k) for k in ("dem", "landuse", "soil")
                                      if ms is not None and getattr(ms, k) is not None},
                         "overlays": M.overlay_layers(ms) if ms is not None else {},
                         "area": area, "sites": sites, "rivers": river_lines, "river_problem": river_problem,
                         "pdf": pdf, "methods": R.methods_text(info, ms, ws), "advice": adv}
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
    folium.GeoJson(mapping(area), name="Study area (with buffer)",
                   style_function=lambda _: {"color": "#ffffff", "weight": 2.4, "dashArray": "7 5",
                                             "fill": False}).add_to(m)
    if r.get("rivers"):
        fg = folium.FeatureGroup(name="River lines (OpenStreetMap)")
        for g, kind, nm in r["rivers"]:
            folium.PolyLine([(y, x) for x, y in g.coords], color="#4fc3f7", weight=2 if kind == "river" else 1,
                            tooltip=nm or kind.capitalize()).add_to(fg)
        fg.add_to(m)
    places = folium.FeatureGroup(name="Your places")
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


def landuse_legend(lk):
    rows = "".join(
        f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
        f"<span style='width:14px;height:14px;border-radius:3px;background:{M.WORLDCOVER_COLOURS.get(v, '#999')};"
        f"border:1px solid #ccc'></span><span>{name} → <b>{code}</b> ({pct:.1f}%)</span></div>"
        for v, name, code, pct in zip(lk["Value"], lk["Land cover (ESA WorldCover)"], lk["SWAT code"],
                                      lk["Area (%)"]))
    return f"<div style='font-size:0.88rem'>{rows}</div>"


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
    st.divider()
    st.markdown("### Your SWAT inputs")
    d1, d2 = st.columns([1.3, 2])
    d1.download_button(f"⬇️ Download the zip ({len(r['zip']) / 1e6:.1f} MB)", r["zip"],
                       file_name="SWAT Bharat inputs.zip", mime="application/zip", type="primary",
                       width="stretch")
    d1.download_button("📄 Download the data report (PDF)", r["pdf"], file_name="SWAT Bharat data report.pdf",
                       mime="application/pdf", width="stretch")
    d2.markdown("<div class='hint'>The zip has a README with step-by-step loading instructions for QSWAT, "
                "QSWAT+, ArcSWAT and SWAT+ Editor. Unzip it to a short folder path with no spaces.</div>",
                unsafe_allow_html=True)

    c = st.columns(5)
    c[0].metric("Study area", f"{r['km2']:,.0f} km²")
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
        st.warning("River lines: " + r["river_problem"])
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

st.caption(f"Version: maps engine {getattr(M, 'VERSION', 'old (before the soil fix)')}")
st.markdown("<br><div class='hint' style='font-size:0.8rem'>Data: Copernicus DEM (ESA); ESA WorldCover; "
            "SoilGrids 2.0 (ISRIC); IMD Pune gridded rainfall (Pai et al. 2014) and temperature (Srivastava "
            "et al. 2009); NASA POWER. IMD download engine from the "
            "<a href='https://doi.org/10.5281/zenodo.24352511'>IMD Data Puller</a>.</div>",
            unsafe_allow_html=True)
