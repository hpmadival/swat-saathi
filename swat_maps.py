"""
The map side of SWAT Bharat: elevation, land use and soil maps, their lookup
tables, and a SWAT soil database (usersoil) for any study area in India.

What this file does, in plain words:
  1. Makes one grid in the study area's UTM zone (30 m or 90 m), so all three
     maps line up cell for cell, as QSWAT and ArcSWAT like.
  2. Elevation (DEM): Copernicus GLO-30 or GLO-90, read straight from its free
     cloud copy (only the part needed is downloaded).
  3. Land use: ESA WorldCover 10 m (2020 or 2021), resampled to the grid by
     taking the most common class in each cell, plus a lookup table that turns
     each WorldCover class into a SWAT land use code (FRST, AGRL, WATR, ...).
  4. Soil: SoilGrids 250 m (sand, silt, clay, bulk density, organic carbon,
     gravel and pH at 5 depths). Every 250 m cell gets a soil type from its
     topsoil and subsoil texture (for example "clay loam over clay"); each soil
     type becomes one row of the usersoil table, with its SWAT soil properties
     worked out from the averages:
       - water holding (AWC) and saturated conductivity (Ksat): Saxton and Rawls (2006)
       - erodibility (USLE K): Williams (1995)
       - moist soil albedo: Baumer (1990)
       - hydrologic soil group: NRCS (2009), from the slowest layer's Ksat
  5. Writes GeoTIFFs, lookup tables (QSWAT / QSWAT+ CSV and ArcSWAT text) and
     usersoil.csv, ready to go in the zip.
"""

import io
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# GDAL settings for reading cloud files quickly (set before rasterio opens anything)
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "45",
    "VSI_CACHE": "TRUE",
}

VERSION = "2026-11-19 progress"   # shown at the bottom of the app, so you can see which version is running

DEM_NODATA = -9999.0
SOILGRIDS_URL = ("/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url="
                 "https://files.isric.org/soilgrids/latest/data/{var}/{var}_{depth}_mean.vrt")
HOMOLOSINE = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"


class MapError(Exception):
    """A map could not be made."""


# ---------------------------------------------------------------------------
# Land use: ESA WorldCover classes -> SWAT codes
# ---------------------------------------------------------------------------
WORLDCOVER = {
    10: ("Tree cover", "FRST"),
    20: ("Shrubland", "RNGB"),
    30: ("Grassland", "RNGE"),
    40: ("Cropland", "AGRL"),
    50: ("Built-up", "URMD"),
    60: ("Bare or sparse vegetation", "BARR"),
    70: ("Snow and ice", "BARR"),
    80: ("Permanent water bodies", "WATR"),
    90: ("Herbaceous wetland", "WETN"),
    95: ("Mangroves", "WETF"),
    100: ("Moss and lichen", "RNGE"),
}
WORLDCOVER_COLOURS = {10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000",
                      60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8", 90: "#0096a0", 95: "#00cf75",
                      100: "#fae6a0"}
CROP_CHOICES = {"AGRL": "Agricultural land, generic (AGRL)", "AGRR": "Row crops (AGRR)",
                "RICE": "Rice (RICE)", "WWHT": "Wheat (WWHT)", "CORN": "Maize (CORN)",
                "SUGC": "Sugarcane (SUGC)", "COTS": "Cotton (COTS)"}
TREE_CHOICES = {"FRST": "Mixed forest (FRST)", "FRSE": "Evergreen forest (FRSE)",
                "FRSD": "Deciduous forest (FRSD)"}
URBAN_CHOICES = {"URMD": "Residential, medium density (URMD)", "URHD": "Residential, high density (URHD)",
                 "URLD": "Residential, low density (URLD)", "UCOM": "Commercial (UCOM)"}


def worldcover_url(lat0, lon0, year=2021):
    """3 x 3 degree tile named by its south-west corner, e.g. N12E075."""
    tile = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
    version = "v200" if year == 2021 else "v100"
    return (f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map/"
            f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")


def dem_url(lat0, lon0, res=30):
    """1 x 1 degree Copernicus DEM tile named by its south-west corner."""
    ns = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}_00"
    ew = f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}_00"
    code, bucket = ("10", "copernicus-dem-30m") if res == 30 else ("30", "copernicus-dem-90m")
    name = f"Copernicus_DSM_COG_{code}_{ns}_{ew}_DEM"
    return f"https://{bucket}.s3.eu-central-1.amazonaws.com/{name}/{name}.tif"


# ---------------------------------------------------------------------------
# Soil: texture classes and SWAT soil properties
# ---------------------------------------------------------------------------
TEXTURES = ["S", "LS", "SL", "L", "SIL", "SI", "SCL", "CL", "SICL", "SC", "SIC", "C"]
TEXTURE_NAMES = {"S": "sand", "LS": "loamy sand", "SL": "sandy loam", "L": "loam", "SIL": "silt loam",
                 "SI": "silt", "SCL": "sandy clay loam", "CL": "clay loam", "SICL": "silty clay loam",
                 "SC": "sandy clay", "SIC": "silty clay", "C": "clay"}

SOIL_VARS = {   # SoilGrids name -> (our name, divide by, to get)
    "sand": ("sand", 10.0),        # g/kg -> %
    "silt": ("silt", 10.0),
    "clay": ("clay", 10.0),
    "bdod": ("bd", 100.0),         # cg/cm3 -> g/cm3
    "soc": ("oc", 100.0),          # dg/kg -> %
    "cfvo": ("rock", 10.0),        # cm3/dm3 -> % by volume
    "phh2o": ("ph", 10.0),         # pH x 10 -> pH
}
SOIL_NAMES = {"sand": "sand", "silt": "silt", "clay": "clay", "bdod": "bulk density",
              "soc": "organic carbon", "cfvo": "gravel", "phh2o": "pH"}
SOIL_DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"]
DEPTH_CM = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100)]
# SWAT layers: (bottom depth in mm, the SoilGrids depths that make it, with weights = thickness)
SWAT_LAYERS = [(150, {0: 5, 1: 10}), (300, {2: 15}), (600, {3: 30}), (1000, {4: 40})]


def texture_class(sand, silt, clay):
    """USDA texture class (array of strings) from sand, silt and clay in %."""
    s, si, c = np.asarray(sand, float), np.asarray(silt, float), np.asarray(clay, float)
    conds = [
        si + 1.5 * c < 15,
        (si + 1.5 * c >= 15) & (si + 2 * c < 30),
        ((c >= 7) & (c < 20) & (s > 52) & (si + 2 * c >= 30)) | ((c < 7) & (si < 50) & (si + 2 * c >= 30)),
        (c >= 7) & (c < 27) & (si >= 28) & (si < 50) & (s <= 52),
        ((si >= 50) & (c >= 12) & (c < 27)) | ((si >= 50) & (si < 80) & (c < 12)),
        (si >= 80) & (c < 12),
        (c >= 20) & (c < 35) & (si < 28) & (s > 45),
        (c >= 27) & (c < 40) & (s > 20) & (s <= 45),
        (c >= 27) & (c < 40) & (s <= 20),
        (c >= 35) & (s > 45),
        (c >= 40) & (si >= 40),
        (c >= 40) & (s <= 45) & (si < 40),
    ]
    return np.select(conds, TEXTURES, default="L")


def saxton_rawls(sand, clay, om):
    """Saxton and Rawls (2006): returns (wilting point, field capacity, saturation,
    Ksat in mm/h, bulk density in g/cm3). sand and clay in %, organic matter in %."""
    S, C = sand / 100.0, clay / 100.0
    OM = min(max(om, 0.0), 8.0)
    t1500t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    t1500 = t1500t + (0.14 * t1500t - 0.02)
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    t33 = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)
    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    ts = t33 + ts33 - 0.097 * S + 0.043
    t1500, t33 = max(t1500, 0.005), max(t33, t1500 + 0.01)
    ts = max(ts, t33 + 0.01)
    B = (math.log(1500) - math.log(33)) / (math.log(t33) - math.log(t1500))
    lam = 1.0 / B
    ks = 1930.0 * (ts - t33) ** (3 - lam)
    bd = (1 - ts) * 2.65
    return t1500, t33, ts, ks, bd


def usle_k(sand, silt, clay, oc):
    """Williams (1995) soil erodibility factor (t h / (MJ mm))."""
    sn1 = 1 - sand / 100.0
    fcsand = 0.2 + 0.3 * math.exp(-0.256 * sand * (1 - silt / 100.0))
    fclsi = (silt / max(clay + silt, 1e-6)) ** 0.3
    forgc = 1 - 0.25 * oc / (oc + math.exp(3.72 - 2.95 * oc))
    fhisand = 1 - 0.7 * sn1 / (sn1 + math.exp(-5.51 + 22.9 * sn1))
    return fcsand * fclsi * forgc * fhisand


def hydrologic_group(min_ksat_mm_h):
    """NRCS (2009, chapter 7) for soils deeper than 100 cm, from the slowest layer's Ksat."""
    um_s = min_ksat_mm_h / 3.6                       # mm/h -> micrometres per second
    if um_s > 10:
        return "A"
    if um_s > 4:
        return "B"
    if um_s > 0.4:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# The containers
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    epsg: int
    transform: object
    width: int
    height: int
    res: float

    @property
    def crs(self):
        from rasterio.crs import CRS
        return CRS.from_epsg(self.epsg)

    @property
    def bounds(self):
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)


@dataclass
class MapSet:
    grid: Grid
    area_utm: object
    dem: np.ndarray = None
    landuse: np.ndarray = None
    soil: np.ndarray = None
    landuse_lookup: pd.DataFrame = None     # WorldCover value, name, SWAT code, area %
    soil_units: pd.DataFrame = None         # soil code, name, description, area %, key properties
    usersoil: pd.DataFrame = None
    problems: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def make_grid(area_wgs84, epsg, res):
    """A grid in UTM covering the study area, snapped to whole cells."""
    import geopandas as gpd
    from rasterio.transform import from_origin
    area_utm = gpd.GeoSeries([area_wgs84], crs=4326).to_crs(epsg).iloc[0]
    x0, y0, x1, y1 = area_utm.bounds
    x0, y0 = math.floor(x0 / res) * res - res, math.floor(y0 / res) * res - res
    x1, y1 = math.ceil(x1 / res) * res + res, math.ceil(y1 / res) * res + res
    width, height = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    return Grid(epsg, from_origin(x0, y1, res, res), width, height, float(res)), area_utm


def _wgs84_bounds(grid, pad=0.01):
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)
    return w - pad, s - pad, e + pad, n + pad


def area_mask(grid, area_utm):
    from rasterio.features import geometry_mask
    return geometry_mask([area_utm], out_shape=(grid.height, grid.width), transform=grid.transform,
                         invert=True, all_touched=True)


# ---------------------------------------------------------------------------
# Reading cloud rasters (kept separate so tests can swap them for fake data)
# ---------------------------------------------------------------------------
def read_window(url, bounds, bounds_crs="EPSG:4326"):
    """Read only the part of a cloud GeoTIFF / VRT that covers bounds.
    Returns (array, transform, crs, nodata) or None when the file does not exist
    (e.g. a DEM tile over the sea)."""
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    path = url if (url.startswith("/vsicurl") or os.path.exists(url)) else "/vsicurl/" + url
    try:
        with rasterio.Env(**GDAL_ENV), rasterio.open(path) as src:
            b = bounds if str(src.crs) == str(bounds_crs) else transform_bounds(bounds_crs, src.crs, *bounds,
                                                                                 densify_pts=21)
            # whole cells only, written out by hand so it works with every rasterio version
            win = from_bounds(*b, transform=src.transform)
            c0 = max(int(math.floor(win.col_off)), 0)
            r0 = max(int(math.floor(win.row_off)), 0)
            c1 = min(int(math.ceil(win.col_off + win.width)), src.width)
            r1 = min(int(math.ceil(win.row_off + win.height)), src.height)
            if c1 <= c0 or r1 <= r0:
                return None
            win = Window(c0, r0, c1 - c0, r1 - r0)
            arr = src.read(1, window=win)
            return arr, src.window_transform(win), src.crs, src.nodata
    except RasterioIOError as err:
        msg = str(err)
        if "404" in msg or "403" in msg or "does not exist" in msg or "No such file" in msg or "not recognized" in msg:
            return None
        raise MapError(f"Could not read {url.split('/')[-1]}: {msg[:200]}")


def _tiles(bounds, step):
    w, s, e, n = bounds
    lats = range(int(math.floor(s / step) * step), int(math.ceil(n / step) * step), step)
    lons = range(int(math.floor(w / step) * step), int(math.ceil(e / step) * step), step)
    return [(la, lo) for la in lats for lo in lons]


def _paint(dest, pieces, grid, resampling, src_nodata, dst_nodata):
    """Reproject several source pieces onto one destination grid."""
    from rasterio.warp import reproject
    for arr, transform, crs, nod in pieces:
        reproject(arr, dest, src_transform=transform, src_crs=crs,
                  src_nodata=nod if nod is not None else src_nodata,
                  dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=dst_nodata,
                  resampling=resampling, init_dest_nodata=False)
    return dest


def _fetch_many(urls, bounds, reader, parallel=4, say=None, what="tile"):
    """Read several tiles at once; report each finished tile through say(fraction, message)."""
    from concurrent.futures import as_completed
    say = say or (lambda f, m: None)
    out = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        jobs = {pool.submit(reader, u, bounds): i for i, u in enumerate(urls)}
        for n, job in enumerate(as_completed(jobs), 1):
            out[jobs[job]] = job.result()
            say(n / len(urls), f"{what} {n} of {len(urls)} downloaded")
    return out


# ---------------------------------------------------------------------------
# 1. Elevation
# ---------------------------------------------------------------------------
def build_dem(grid, reader=None, say=None):
    from rasterio.enums import Resampling
    reader = reader or read_window
    bounds = _wgs84_bounds(grid)
    src_res = 30 if grid.res < 60 else 90
    urls = [dem_url(la, lo, src_res) for la, lo in _tiles(bounds, 1)]
    say = say or (lambda f, m: None)
    say(0.0, f"reading {len(urls)} Copernicus DEM tile(s)")
    pieces = [p for p in _fetch_many(urls, bounds, reader, say=lambda f, m: say(0.9 * f, m),
                                     what="Copernicus DEM tile") if p is not None]
    say(0.95, "fitting the DEM to the UTM grid")
    if not pieces:
        raise MapError("No Copernicus DEM tiles were found for this area.")
    dem = np.full((grid.height, grid.width), DEM_NODATA, dtype="float32")
    return _paint(dem, [(a.astype("float32"), t, c, n) for a, t, c, n in pieces], grid,
                  Resampling.bilinear, DEM_NODATA, DEM_NODATA)


# ---------------------------------------------------------------------------
# 2. Land use
# ---------------------------------------------------------------------------
def build_landuse(grid, year=2021, reader=None, say=None):
    from rasterio.enums import Resampling
    reader = reader or read_window
    bounds = _wgs84_bounds(grid)
    urls = [worldcover_url(la, lo, year) for la, lo in _tiles(bounds, 3)]
    say = say or (lambda f, m: None)
    say(0.0, f"reading {len(urls)} ESA WorldCover tile(s)")
    pieces = [p for p in _fetch_many(urls, bounds, reader, say=lambda f, m: say(0.85 * f, m),
                                     what="ESA WorldCover tile") if p is not None]
    say(0.9, "taking the most common class in each cell")
    if not pieces:
        raise MapError("No ESA WorldCover tiles were found for this area.")
    lu = np.zeros((grid.height, grid.width), dtype="uint8")
    return _paint(lu, pieces, grid, Resampling.mode, 0, 0)


def landuse_table(lu, mask, codes):
    vals, counts = np.unique(lu[mask & (lu > 0)], return_counts=True)
    total = counts.sum() or 1
    rows = [{"Value": int(v), "Land cover (ESA WorldCover)": WORLDCOVER.get(int(v), ("Unknown", "AGRL"))[0],
             "SWAT code": codes.get(int(v), "AGRL"), "Area (%)": round(100 * c / total, 2)}
            for v, c in zip(vals, counts)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Soil
# ---------------------------------------------------------------------------
SOILGRIDS_WCS = "https://maps.isric.org/mapserv"
WCS_CHUNK_M = 150_000          # ask for at most 150 km x 150 km at a time (600 x 600 cells)


def _wcs_piece(var, depth, x0, y0, x1, y1, tries=2, timeout=60):
    """One GeoTIFF from ISRIC's map service (WCS), in Homolosine metres."""
    import time
    import requests
    from rasterio.io import MemoryFile
    params = [("map", f"/map/{var}.map"), ("SERVICE", "WCS"), ("VERSION", "2.0.1"),
              ("REQUEST", "GetCoverage"), ("COVERAGEID", f"{var}_{depth}_mean"), ("FORMAT", "GEOTIFF_INT16"),
              ("SUBSET", f"X({x0:.0f},{x1:.0f})"), ("SUBSET", f"Y({y0:.0f},{y1:.0f})")]
    last = ""
    for attempt in range(tries):
        try:
            r = requests.get(SOILGRIDS_WCS, params=params, timeout=timeout,
                             headers={"User-Agent": "SWAT-Bharat (SWAT input tool)"})
            if r.status_code == 200 and r.content[:4] in (b"II*\x00", b"MM\x00*"):
                with MemoryFile(r.content) as mf, mf.open() as src:
                    return src.read(1), src.transform, src.crs, src.nodata
            last = f"HTTP {r.status_code}: {r.text[:120].strip()}" if r.status_code != 200 else \
                f"not a map: {r.text[:120].strip()}"
        except requests.RequestException as err:
            last = err.__class__.__name__
        time.sleep(3 * (attempt + 1))
    raise MapError(f"ISRIC map service: {last}")


def soilgrids_wcs_reader(var, depth, bounds):
    """One SoilGrids layer through ISRIC's map service, in pieces, stitched together."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    igh = CRS.from_proj4(HOMOLOSINE)
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", igh, *bounds, densify_pts=21)
    x0, y0 = math.floor(x0 / 250) * 250, math.floor(y0 / 250) * 250
    x1, y1 = math.ceil(x1 / 250) * 250, math.ceil(y1 / 250) * 250
    if (x1 - x0) <= WCS_CHUNK_M and (y1 - y0) <= WCS_CHUNK_M:
        a, t, c, n = _wcs_piece(var, depth, x0, y0, x1, y1)
        return a, t, c or igh, n
    # big area: fetch chunks and stitch them on one Homolosine grid
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
    w, h = int((x1 - x0) / 250), int((y1 - y0) / 250)
    out = np.full((h, w), -32768, dtype="int16")
    t = from_origin(x0, y1, 250, 250)
    for cx in np.arange(x0, x1, WCS_CHUNK_M):
        for cy in np.arange(y0, y1, WCS_CHUNK_M):
            a, ta, ca, na = _wcs_piece(var, depth, cx, cy, min(cx + WCS_CHUNK_M, x1), min(cy + WCS_CHUNK_M, y1))
            with rasterio.Env():
                reproject(a, out, src_transform=ta, src_crs=ca or igh, src_nodata=na if na is not None else -32768,
                          dst_transform=t, dst_crs=igh, dst_nodata=-32768, init_dest_nodata=False)
    return out, t, igh, -32768


def soilgrids_reader(var, depth, bounds):
    """Read one SoilGrids layer covering WGS 84 bounds: first straight from ISRIC's file server,
    and if that fails or comes back empty, from ISRIC's map service. Raises MapError with the reason."""
    errors = []
    for how, fn in (("file server", lambda: read_window(SOILGRIDS_URL.format(var=var, depth=depth), bounds)),
                    ("map service", lambda: soilgrids_wcs_reader(var, depth, bounds))):
        try:
            piece = fn()
            if piece is not None:
                a, _, _, nod = piece
                good = (a != nod) & (a > -32000) if nod is not None else (a > -32000)
                if good.any():
                    return piece
                errors.append(f"{how}: no values here")
            else:
                errors.append(f"{how}: not found")
        except Exception as err:                       # try the other route
            errors.append(f"{how}: {str(err)[:160]}")
    raise MapError(f"{var} {depth}: " + "; ".join(errors))


def build_soil(grid, area_utm, mask, reader=None, say=None, prefix="SG", time_limit=240):
    """Soil map on the main grid + table of soil units + usersoil rows."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
    from scipy import ndimage
    reader = reader or soilgrids_reader
    bounds = _wgs84_bounds(grid, pad=0.03)

    # a 250 m grid in the same UTM zone, covering the same ground
    x0, y0, x1, y1 = grid.bounds
    r = 250.0
    sx0, sy1 = math.floor(x0 / r) * r, math.ceil(y1 / r) * r
    sw, sh = int(math.ceil((x1 - sx0) / r)), int(math.ceil((sy1 - y0) / r))
    sgrid = Grid(grid.epsg, from_origin(sx0, sy1, r, r), sw, sh, r)

    jobs = [(v, d) for v in SOIL_VARS for d in SOIL_DEPTHS]
    say = say or (lambda f, m: None)
    say(0.0, f"reading {len(jobs)} SoilGrids layers")

    def safe(j):
        try:
            return reader(j[0], j[1], bounds), None
        except Exception as err:
            return None, str(err)

    # 3 at a time (gentle on ISRIC's servers), with an overall time limit so a dead server
    # cannot freeze the app: after time_limit seconds, layers still missing count as failed.
    import time
    from concurrent.futures import FIRST_COMPLETED, wait
    pool = ThreadPoolExecutor(max_workers=3)
    futs = {pool.submit(safe, j): k for k, j in enumerate(jobs)}
    results = [(None, "not finished within the time limit")] * len(jobs)
    deadline = time.time() + time_limit
    done_n, pending = 0, set(futs)
    while pending and time.time() < deadline:
        finished, pending = wait(pending, timeout=max(0.5, min(5, deadline - time.time())),
                                 return_when=FIRST_COMPLETED)
        for f in finished:
            k = futs[f]
            results[k] = f.result()
            done_n += 1
            v, d = jobs[k]
            say(0.9 * done_n / len(jobs), f"layer {done_n} of {len(jobs)} downloaded "
                                          f"({SOIL_NAMES.get(v, v)}, {d.replace('cm', ' cm')})")
    pool.shutdown(wait=False, cancel_futures=True)
    got = [p for p, _ in results]
    errs = [e for _, e in results if e]
    say(0.93, "working out soil types and their properties")
    needed = {("sand", d) for d in SOIL_DEPTHS} | {("clay", d) for d in SOIL_DEPTHS} | \
             {("silt", d) for d in SOIL_DEPTHS}
    missing = [f"{v} {d}" for (v, d), p in zip(jobs, got) if p is None and (v, d) in needed]
    if missing:
        raise MapError("SoilGrids could not be reached for " + ", ".join(missing[:4])
                       + (" and more" if len(missing) > 4 else "") + ". Reason: " + (errs[0] if errs else "unknown")
                       + ". ISRIC's servers are sometimes busy: try again in a few minutes.")
    layers = {}
    for (v, d), piece in zip(jobs, got):
        arr = np.full((sh, sw), np.nan, dtype="float32")
        if piece is not None:
            a, t, c, nod = piece
            a = a.astype("float32")
            if nod is not None:
                a[a == nod] = np.nan
            a[a <= -32000] = np.nan
            reproject(a, arr, src_transform=t, src_crs=c, src_nodata=np.nan, dst_transform=sgrid.transform,
                      dst_crs=sgrid.crs, dst_nodata=np.nan, resampling=Resampling.nearest,
                      init_dest_nodata=False)
        layers[(SOIL_VARS[v][0], d)] = arr / SOIL_VARS[v][1]

    # per SoilGrids depth: texture must add up; zero texture = no data (water, towns)
    for d in SOIL_DEPTHS:
        tot = layers[("sand", d)] + layers[("silt", d)] + layers[("clay", d)]
        bad = ~(tot > 50)
        for k in ("sand", "silt", "clay"):
            layers[(k, d)] = np.where(bad, np.nan, layers[(k, d)] * 100.0 / np.where(bad, 1, tot))

    def depth_avg(k, idxs):
        num = sum(layers[(k, SOIL_DEPTHS[i])] * w for i, w in idxs.items())
        return num / sum(idxs.values())

    top = {k: depth_avg(k, {0: 5, 1: 10, 2: 15}) for k in ("sand", "silt", "clay")}
    sub = {k: depth_avg(k, {3: 30, 4: 40}) for k in ("sand", "silt", "clay")}
    valid = np.isfinite(top["clay"]) & np.isfinite(sub["clay"])
    if not valid.any():
        raise MapError("SoilGrids returned no soil data for this area.")

    # fill gaps (rivers, reservoirs, towns) from the nearest soil cell
    if (~valid).any():
        _, (ii, jj) = ndimage.distance_transform_edt(~valid, return_indices=True)
        for dct in (top, sub):
            for k in dct:
                dct[k] = dct[k][ii, jj]
        for key in layers:
            layers[key] = layers[key][ii, jj]

    ttop = texture_class(top["sand"], top["silt"], top["clay"])
    tsub = texture_class(sub["sand"], sub["silt"], sub["clay"])
    combo = np.char.add(np.char.add(ttop.astype("U4"), "|"), tsub.astype("U4"))

    # only count cells inside the study area when naming and ranking soil types
    from rasterio.features import geometry_mask
    smask = geometry_mask([area_utm], out_shape=(sh, sw), transform=sgrid.transform, invert=True,
                          all_touched=True)
    inside = combo[smask] if smask.any() else combo.ravel()
    names, counts = np.unique(inside, return_counts=True)
    order = np.argsort(-counts)
    names, counts = names[order], counts[order]
    code_of = {n: i + 1 for i, n in enumerate(names)}
    codes250 = np.zeros((sh, sw), dtype="uint16")
    for n, code in code_of.items():
        codes250[combo == n] = code
    # cells outside the area that have a soil type not seen inside: give them the nearest inside type
    if (codes250 == 0).any():
        _, (ii, jj) = ndimage.distance_transform_edt(codes250 == 0, return_indices=True)
        codes250 = codes250[ii, jj]

    units, usersoil = [], []
    total = counts.sum()
    for n, cnt in zip(names, counts):
        code = code_of[n]
        t, s = n.split("|")
        sel = (codes250 == code) & (smask if smask.any() else True)
        if not sel.any():
            sel = codes250 == code
        snam = f"{prefix}_{t}_{s}"
        props = []
        for bottom, idxs in SWAT_LAYERS:
            p = {k: float(np.nanmean(depth_avg(k, idxs)[sel])) for k in ("sand", "silt", "clay", "bd", "oc",
                                                                         "rock", "ph")}
            props.append((bottom, p))
        row, unit = _usersoil_row(len(usersoil) + 1, snam, props)
        usersoil.append(row)
        unit.update({"Soil code": code, "Soil name": snam,
                     "What it is": f"{TEXTURE_NAMES[t].capitalize()} over {TEXTURE_NAMES[s]}",
                     "Area (%)": round(100 * cnt / total, 2)})
        units.append(unit)

    # put the 250 m soil codes on the main grid
    soil = np.zeros((grid.height, grid.width), dtype="uint16")
    with rasterio.Env():
        reproject(codes250, soil, src_transform=sgrid.transform, src_crs=sgrid.crs, src_nodata=0,
                  dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=0,
                  resampling=Resampling.nearest)
    cols = ["Soil code", "Soil name", "What it is", "Area (%)", "Hydrologic group", "Depth (mm)",
            "Topsoil sand (%)", "Topsoil silt (%)", "Topsoil clay (%)", "Topsoil organic carbon (%)",
            "Topsoil AWC (mm/mm)", "Topsoil Ksat (mm/h)", "USLE K"]
    return soil, pd.DataFrame(units)[cols], pd.DataFrame(usersoil)


USERSOIL_LAYER_FIELDS = ["SOL_Z", "SOL_BD", "SOL_AWC", "SOL_K", "SOL_CBN", "CLAY", "SILT", "SAND", "ROCK",
                         "SOL_ALB", "USLE_K", "SOL_EC"]


def _usersoil_row(objectid, snam, props):
    """One usersoil row (SWAT 2012 / QSWAT / QSWAT+ layout) from layer averages."""
    row = {"OBJECTID": objectid, "MUID": snam, "SEQN": 1, "SNAM": snam, "S5ID": "", "CMPPCT": 100,
           "NLAYERS": len(props), "HYDGRP": "", "SOL_ZMX": props[-1][0], "ANION_EXCL": 0.5,
           "SOL_CRK": 0.5, "TEXTURE": ""}
    ks_all, textures, layer_vals = [], [], []
    for bottom, p in props:
        sand, silt, clay = p["sand"], p["silt"], p["clay"]
        oc = max(p["oc"], 0.01) if np.isfinite(p["oc"]) else 0.5
        om = oc * 1.724
        wp, fc, sat, ks, bd_sr = saxton_rawls(sand, clay, om)
        bd = p["bd"] if np.isfinite(p["bd"]) and 0.8 < p["bd"] < 2.2 else bd_sr
        rock = p["rock"] if np.isfinite(p["rock"]) else 0.0
        awc = max((fc - wp) * (1 - rock / 100.0), 0.01)
        ks_all.append(ks)
        textures.append(str(texture_class(sand, silt, clay)))
        layer_vals.append({
            "SOL_Z": bottom, "SOL_BD": round(bd, 3), "SOL_AWC": round(awc, 3), "SOL_K": round(ks, 2),
            "SOL_CBN": round(oc, 3), "CLAY": round(clay, 1), "SILT": round(silt, 1), "SAND": round(sand, 1),
            "ROCK": round(rock, 1), "SOL_ALB": round(0.6 / math.exp(0.4 * om), 3),
            "USLE_K": round(usle_k(sand, silt, clay, oc), 3), "SOL_EC": 0,
            "_ph": round(p["ph"], 1) if np.isfinite(p["ph"]) else 0})
    row["HYDGRP"] = hydrologic_group(min(ks_all))
    row["TEXTURE"] = "-".join(textures)
    for i in range(10):
        v = layer_vals[i] if i < len(layer_vals) else None
        for f in USERSOIL_LAYER_FIELDS:
            row[f"{f}{i + 1}"] = v[f] if v else 0
    for i in range(10):
        row[f"SOL_CAL{i + 1}"] = 0
    for i in range(10):
        row[f"SOL_PH{i + 1}"] = layer_vals[i]["_ph"] if i < len(layer_vals) else 0
    first = layer_vals[0]
    unit = {"Hydrologic group": row["HYDGRP"], "Depth (mm)": row["SOL_ZMX"],
            "Topsoil sand (%)": first["SAND"], "Topsoil silt (%)": first["SILT"],
            "Topsoil clay (%)": first["CLAY"], "Topsoil organic carbon (%)": first["SOL_CBN"],
            "Topsoil AWC (mm/mm)": first["SOL_AWC"], "Topsoil Ksat (mm/h)": first["SOL_K"],
            "USLE K": first["USLE_K"]}
    return row, unit


# ---------------------------------------------------------------------------
# 4. The main job
# ---------------------------------------------------------------------------
def build_maps(area_wgs84, epsg, res=30, want_dem=True, want_landuse=True, want_soil=True,
               landuse_year=2021, crop_code="AGRL", tree_code="FRST", urban_code="URMD",
               blank_outside=True, progress=None, dem_reader=None, lu_reader=None, soil_reader=None):
    report = progress or (lambda stage, f, m: None)

    def stage(name):
        return lambda f, m: report(name, f, m)
    grid, area_utm = make_grid(area_wgs84, epsg, res)
    ms = MapSet(grid=grid, area_utm=area_utm)
    mask = area_mask(grid, area_utm)
    if want_dem:
        ms.dem = build_dem(grid, dem_reader, stage("dem"))
        if (ms.dem[mask] == DEM_NODATA).any():
            ms.problems.append("Part of the study area has no elevation (over the sea, or a DEM tile is "
                               "missing).")
        if blank_outside:
            ms.dem[~mask] = DEM_NODATA
    if want_landuse:
        try:
            ms.landuse = build_landuse(grid, landuse_year, lu_reader, stage("landuse"))
        except MapError as err:
            ms.problems.append(f"Land use could not be made: {err}")
            want_landuse = False
    if want_landuse:
        if blank_outside:
            ms.landuse[~mask] = 0
        codes = {v: c for v, (_, c) in WORLDCOVER.items()}
        codes.update({10: tree_code, 40: crop_code, 50: urban_code})
        ms.landuse_lookup = landuse_table(ms.landuse, mask if blank_outside else np.ones_like(mask), codes)
    if want_soil:
        try:
            ms.soil, ms.soil_units, ms.usersoil = build_soil(grid, area_utm, mask, soil_reader, stage("soil"))
            if blank_outside:
                ms.soil[~mask] = 0
            ms.notes.append(f"Soil: {len(ms.soil_units)} soil types made from SoilGrids topsoil (0-30 cm) and "
                            f"subsoil (30-100 cm) texture, 4 layers down to 1 m.")
        except MapError as err:                       # keep everything else; say what went wrong
            ms.problems.append(f"Soil could not be made: {err}")
    ms.settings = {
        "Map grid": f"{res} m cells, WGS 84 / UTM (EPSG:{epsg})",
        "Map size": f"{grid.width} x {grid.height} cells",
        "Cells outside the study area": "blanked (no data)" if blank_outside else "kept",
        "Land use": f"ESA WorldCover {landuse_year}" if want_landuse else "not made",
    }
    return ms


# ---------------------------------------------------------------------------
# 5. Writing the files
# ---------------------------------------------------------------------------
def _tif_bytes(arr, grid, nodata, dtype):
    import rasterio
    from rasterio.io import MemoryFile
    profile = {"driver": "GTiff", "height": grid.height, "width": grid.width, "count": 1, "dtype": dtype,
               "crs": grid.crs, "transform": grid.transform, "nodata": nodata, "compress": "lzw",
               "tiled": True, "blockxsize": 256, "blockysize": 256}
    if dtype == "float32":
        profile["predictor"] = 3
    with rasterio.Env(), MemoryFile() as mf:
        with mf.open(**profile) as dst:
            dst.write(arr.astype(dtype), 1)
        return mf.read()


def _shapefile_files(geoms, names, epsg, stem):
    """{file name: bytes} for a small shapefile (all its parts)."""
    import geopandas as gpd
    folder = tempfile.mkdtemp(prefix="shp-")
    gdf = gpd.GeoDataFrame({"Name": names}, geometry=geoms, crs=4326).to_crs(epsg)
    gdf.to_file(os.path.join(folder, f"{stem}.shp"))
    out = {}
    for f in sorted(os.listdir(folder)):
        with open(os.path.join(folder, f), "rb") as fh:
            out[f] = fh.read()
    return out


def map_files(ms, sites=None, area_wgs84=None):
    """{path inside the zip: bytes or text}."""
    files = {}
    g = ms.grid
    if ms.dem is not None:
        files["Maps/dem.tif"] = _tif_bytes(ms.dem, g, DEM_NODATA, "float32")
    if ms.landuse is not None:
        files["Maps/landuse.tif"] = _tif_bytes(ms.landuse, g, 0, "uint8")
        lk = ms.landuse_lookup
        files["LookupTables/landuse lookup QSWAT.csv"] = "LANDUSE_ID,SWAT_CODE\n" + "".join(
            f"{v},{c}\n" for v, c in zip(lk["Value"], lk["SWAT code"]))
        files["LookupTables/landuse lookup ArcSWAT.txt"] = '"Value","Landuse"\r\n' + "".join(
            f"{v},{c}\r\n" for v, c in zip(lk["Value"], lk["SWAT code"]))
        files["LookupTables/landuse classes explained.csv"] = lk.to_csv(index=False)
    if ms.soil is not None:
        files["Maps/soil.tif"] = _tif_bytes(ms.soil, g, 0, "uint16")
        su = ms.soil_units
        files["LookupTables/soil lookup QSWAT.csv"] = "SOIL_ID,SNAM\n" + "".join(
            f"{v},{n}\n" for v, n in zip(su["Soil code"], su["Soil name"]))
        files["LookupTables/soil lookup ArcSWAT.txt"] = '"Value","Name"\r\n' + "".join(
            f"{v},{n}\r\n" for v, n in zip(su["Soil code"], su["Soil name"]))
        files["SoilDatabase/usersoil.csv"] = ms.usersoil.to_csv(index=False)
        files["SoilDatabase/soil types explained.csv"] = su.to_csv(index=False)
    if area_wgs84 is not None:
        for f, b in _shapefile_files([area_wgs84], ["Study area"], g.epsg, "studyarea").items():
            files[f"Maps/{f}"] = b
    if sites:
        shapes = [s for s in sites if not s.is_point]
        points = [s for s in sites if s.is_point]
        for group, stem in ((shapes, "places"), (points, "points")):
            if group:
                for f, b in _shapefile_files([s.geometry for s in group], [s.name for s in group], g.epsg,
                                             stem).items():
                    files[f"Maps/{f}"] = b
    return files


# ---------------------------------------------------------------------------
# 6. Quick pictures for the app
# ---------------------------------------------------------------------------
def preview_png(ms, which, max_px=900):
    """A small PNG of one map (bytes), for showing on screen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    arr = {"dem": ms.dem, "landuse": ms.landuse, "soil": ms.soil}[which]
    step = max(1, int(math.ceil(max(arr.shape) / max_px)))
    a = arr[::step, ::step]
    fig, ax = plt.subplots(figsize=(6, 6 * a.shape[0] / max(a.shape[1], 1)), dpi=110)
    ax.set_axis_off()
    if which == "dem":
        m = np.ma.masked_equal(a, DEM_NODATA)
        im = ax.imshow(m, cmap="terrain", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Elevation (m)")
    elif which == "landuse":
        rgba = np.zeros(a.shape + (4,))
        for v, col in WORLDCOVER_COLOURS.items():
            rgba[a == v] = matplotlib.colors.to_rgba(col)
        ax.imshow(rgba, interpolation="nearest")
    else:
        n = int(a.max())
        cmap = ListedColormap(plt.get_cmap("tab20")(np.arange(max(n, 1)) % 20))
        ax.imshow(np.ma.masked_equal(a, 0), cmap=cmap, vmin=0.5, vmax=max(n, 1) + 0.5,
                  interpolation="nearest")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def _colour(which, a, n_soil=1):
    """RGBA image (0-1) of one map, with see-through no-data cells."""
    import matplotlib
    import matplotlib.pyplot as plt
    rgba = np.zeros(a.shape + (4,))
    if which == "dem":
        ok = np.isfinite(a) & (a != DEM_NODATA)
        if ok.any():
            lo, hi = np.percentile(a[ok], [1, 99])
            rgba[ok] = plt.get_cmap("terrain")(np.clip((a[ok] - lo) / max(hi - lo, 1e-6), 0, 1) * 0.85 + 0.15)
    elif which == "landuse":
        for v, col in WORLDCOVER_COLOURS.items():
            rgba[a == v] = matplotlib.colors.to_rgba(col)
    else:
        cmap = plt.get_cmap("tab20")
        for code in range(1, n_soil + 1):
            rgba[a == code] = cmap((code - 1) % 20)
    return rgba


def overlay_layers(ms, max_px=1400):
    """Each map as a see-through PNG in latitude/longitude, for laying over a web map.
    Returns {"dem" / "landuse" / "soil": (PNG as a data URL, [[south, west], [north, east]])}."""
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import calculate_default_transform, reproject
    g = ms.grid
    out = {}
    for which, arr in (("dem", ms.dem), ("landuse", ms.landuse), ("soil", ms.soil)):
        if arr is None:
            continue
        step = max(1, int(math.ceil(max(arr.shape) / max_px)))
        small = arr[::step, ::step]
        x0, y0, x1, y1 = g.bounds
        src_t = from_bounds(x0, y0, x1, y1, small.shape[1], small.shape[0])
        dst_t, w, h = calculate_default_transform(g.crs, "EPSG:4326", small.shape[1], small.shape[0],
                                                  x0, y0, x1, y1)
        nod = DEM_NODATA if which == "dem" else 0
        dst = np.full((h, w), nod, dtype=small.dtype)
        reproject(small, dst, src_transform=src_t, src_crs=g.crs, src_nodata=nod, dst_transform=dst_t,
                  dst_crs="EPSG:4326", dst_nodata=nod,
                  resampling=Resampling.bilinear if which == "dem" else Resampling.nearest)
        n_soil = int(ms.soil.max()) if which == "soil" else 1
        rgba = _colour(which, dst, n_soil)
        buf = io.BytesIO()
        plt.imsave(buf, rgba, format="png")
        west, north = dst_t.c, dst_t.f
        east, south = west + dst_t.a * w, north + dst_t.e * h
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        out[which] = (url, [[south, west], [north, east]])
    return out


def soil_colours(ms):
    """{soil code: hex colour}, matching the soil overlay."""
    import matplotlib
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    return {int(c): matplotlib.colors.to_hex(cmap((int(c) - 1) % 20)) for c in ms.soil_units["Soil code"]}
