"""
The engine behind the IMD Data Puller app.

What this file does, in plain words:
  1. Downloads IMD's yearly gridded files (rainfall, max and min temperature)
     and keeps them in a folder, so each year is only downloaded once.
  2. Reads a year's file and cuts out the values we need.
  3. Works out which grid cells sit under a point or a catchment, and how much
     of each cell is inside (area weighting).
  4. Turns the daily grid values into daily series, monthly and yearly
     summaries, and a Hargreaves potential evaporation estimate.
  5. Packs everything into a zip file to download.

The app screen (app.py) only calls the functions in this file, so the same
functions can also be used directly from a notebook.
"""

import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import Point, box

try:  # use imdlib's download addresses so an imdlib update also updates these
    from imdlib.util import ARCHIVE_URLS
except Exception:  # pragma: no cover - fallback if imdlib changes its layout
    ARCHIVE_URLS = {
        "rain": ("https://imdpune.gov.in/cmpg/Griddata/rainfall.php", "rain"),
        "tmax": ("https://imdpune.gov.in/cmpg/Griddata/maxtemp.php", "maxtemp"),
        "tmin": ("https://imdpune.gov.in/cmpg/Griddata/mintemp.php", "mintemp"),
    }


# ---------------------------------------------------------------------------
# The IMD grids and variables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """A regular IMD grid. lon0/lat0 are the centre of the south-west cell."""
    lon0: float
    lat0: float
    step: float
    nlon: int
    nlat: int

    @property
    def lons(self):
        return self.lon0 + self.step * np.arange(self.nlon)

    @property
    def lats(self):
        return self.lat0 + self.step * np.arange(self.nlat)

    @property
    def bounds(self):
        h = self.step / 2
        return (self.lon0 - h, self.lat0 - h,
                self.lons[-1] + h, self.lats[-1] + h)

    def file_size(self, days):
        return days * self.nlat * self.nlon * 4


RAIN_GRID = Grid(66.5, 6.5, 0.25, 135, 129)   # 0.25 degree, from 1901
TEMP_GRID = Grid(67.5, 7.5, 1.0, 31, 31)      # 1 degree, from 1951

VARIABLES = {
    "rain": {"name": "Rainfall", "column": "Rainfall (mm)", "unit": "mm",
             "grid": RAIN_GRID, "first_year": 1901, "how": "total"},
    "tmax": {"name": "Maximum temperature", "column": "Maximum temperature (°C)",
             "unit": "°C", "grid": TEMP_GRID, "first_year": 1951, "how": "mean"},
    "tmin": {"name": "Minimum temperature", "column": "Minimum temperature (°C)",
             "unit": "°C", "grid": TEMP_GRID, "first_year": 1951, "how": "mean"},
}
PET_COLUMN = "Potential evaporation (mm)"
PET_INFO = {"name": "Potential evaporation", "column": PET_COLUMN, "how": "total"}

KM_PER_DEGREE = 111.32


class NotAvailableError(Exception):
    """IMD has no file for this year yet (the server sent back an empty file)."""


class DownloadFailedError(Exception):
    """The download did not finish properly."""


def days_in_year(year):
    year = int(year)
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


# ---------------------------------------------------------------------------
# 1. Downloading and reading the yearly files
# ---------------------------------------------------------------------------

def grd_path(data_dir, var, year):
    """Where a year's file is kept. Same naming as imdlib's 'yearwise' format,
    so files downloaded with imdlib can be reused here and the other way round."""
    ext = ".grd" if var == "rain" else ".GRD"
    return os.path.join(data_dir, var, f"{int(year)}{ext}")


def is_saved(data_dir, var, year):
    path = grd_path(data_dir, var, year)
    expected = VARIABLES[var]["grid"].file_size(days_in_year(year))
    return os.path.isfile(path) and os.path.getsize(path) == expected


# A second, slower folder (for example Google Drive) that keeps a backup copy of
# every downloaded year. The app works from the fast folder (data_dir) and only
# touches the backup folder to copy a missing year in, or a new year out.
BACKUP_DIR = os.environ.get("IMD_BACKUP_DIR") or None
_backup_pool = None
_backup_jobs = []


def _backup_later(src, dst):
    """Copy a new file to the backup folder in the background, so nobody waits for Drive."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    global _backup_pool
    if _backup_pool is None:
        _backup_pool = ThreadPoolExecutor(max_workers=1)

    def job():
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
    _backup_jobs.append(_backup_pool.submit(job))


def backups_pending():
    """Number of files still being copied to the backup folder."""
    return sum(1 for j in _backup_jobs if not j.done())


def saved_count(data_dir, var, backup_dir=None):
    """How many years of a variable are saved (in the fast folder or the backup)."""
    years = set()
    for folder in [data_dir, backup_dir]:
        if folder and os.path.isdir(os.path.join(folder, var)):
            years |= {f.split(".")[0] for f in os.listdir(os.path.join(folder, var))
                      if f.lower().endswith(".grd")}
    return len(years)


def is_available(data_dir, var, year, backup_dir=None):
    return is_saved(data_dir, var, year) or bool(backup_dir and is_saved(backup_dir, var, year))


def download_year(var, year, data_dir, timeout=300, tries=3, backup_dir=None):
    """Make sure one year's file is in the data folder.

    Order: the fast folder, then the backup folder (copied in), then IMD.
    Returns "saved earlier", "copied from backup" or "downloaded". Raises
    NotAvailableError when IMD has not published that year, and
    DownloadFailedError for other problems.
    """
    if is_saved(data_dir, var, year):
        return "saved earlier"
    os.makedirs(os.path.join(data_dir, var), exist_ok=True)
    if backup_dir and is_saved(backup_dir, var, year):
        import shutil
        path = grd_path(data_dir, var, year)
        shutil.copyfile(grd_path(backup_dir, var, year), path + ".part")
        os.replace(path + ".part", path)
        return "copied from backup"
    url, form_key = ARCHIVE_URLS[var]
    expected = VARIABLES[var]["grid"].file_size(days_in_year(year))
    last_error = None
    for _ in range(tries):
        try:
            reply = requests.post(url, data={form_key: int(year)}, timeout=timeout)
            reply.raise_for_status()
            content = reply.content
        except requests.Timeout:
            last_error = "IMD's server took too long to answer"
            continue
        except requests.RequestException as err:
            last_error = ("could not connect to IMD's server" if isinstance(err, requests.ConnectionError)
                          else f"IMD's server replied with an error ({err.__class__.__name__})")
            continue
        if len(content) == 0:
            raise NotAvailableError(
                f"IMD has not published {VARIABLES[var]['name'].lower()} for {year} yet.")
        if len(content) != expected:
            last_error = f"the file arrived incomplete ({len(content):,} of {expected:,} bytes)"
            continue
        path = grd_path(data_dir, var, year)
        part = path + ".part"
        with open(part, "wb") as f:
            f.write(content)
        os.replace(part, path)
        if backup_dir and os.path.abspath(backup_dir) != os.path.abspath(data_dir):
            _backup_later(path, grd_path(backup_dir, var, year))
        return "downloaded"
    raise DownloadFailedError(
        f"Could not download {VARIABLES[var]['name'].lower()} for {year} "
        f"after {tries} tries: {last_error}. IMD's server is sometimes busy; press the button "
        f"again later and only the missing years will be fetched.")


def read_year(var, year, data_dir):
    """Read one year's file. Returns a float32 array shaped (day, latitude, longitude),
    south to north and west to east, with IMD's 'no data' values turned into NaN."""
    grid = VARIABLES[var]["grid"]
    days = days_in_year(year)
    values = np.fromfile(grd_path(data_dir, var, year), dtype="<f4")
    if values.size != days * grid.nlat * grid.nlon:
        raise DownloadFailedError(f"The saved file for {var} {year} is damaged; delete it and try again.")
    values = values.reshape(days, grid.nlat, grid.nlon)
    if var == "rain":
        values[values < -900] = np.nan        # IMD uses -999 for no data
    else:
        values[values > 99] = np.nan          # IMD uses 99.9 for no data
    return values


# ---------------------------------------------------------------------------
# 2. Places: points, boxes and uploaded shapefiles
# ---------------------------------------------------------------------------

@dataclass
class Site:
    name: str
    geometry: object          # shapely geometry in longitude/latitude (EPSG:4326)

    @property
    def is_point(self):
        return self.geometry.geom_type == "Point"

    @property
    def centre(self):
        c = self.geometry if self.is_point else self.geometry.representative_point()
        return c.y, c.x


def area_km2(geometry):
    """True area of a polygon in square kilometres (equal-area projection)."""
    import geopandas as gpd
    if geometry.geom_type == "Point":
        return 0.0
    return float(gpd.GeoSeries([geometry], crs=4326).to_crs(6933).area.iloc[0] / 1e6)


def clean_name(name, used):
    name = str(name).strip() or "Site"
    base, n = name, 2
    while name in used:
        name = f"{base} ({n})"
        n += 1
    used.add(name)
    return name


def read_boundary_files(files):
    """Read uploaded boundary files.

    files: list of (file name, bytes). Accepts a zipped shapefile, the loose
    shapefile parts (.shp .shx .dbf .prj), GeoJSON, GeoPackage or KML.
    Returns (GeoDataFrame in EPSG:4326, list of warnings).
    """
    import geopandas as gpd
    warnings = []
    folder = tempfile.mkdtemp(prefix="boundary-")
    for fname, data in files:
        fname = os.path.basename(fname)
        if fname.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for member in z.namelist():
                    if member.endswith("/") or "__MACOSX" in member:
                        continue
                    target = os.path.join(folder, os.path.basename(member))
                    with open(target, "wb") as out:
                        out.write(z.read(member))
        else:
            with open(os.path.join(folder, fname), "wb") as out:
                out.write(data)

    readable = [f for f in sorted(os.listdir(folder))
                if f.lower().endswith((".shp", ".geojson", ".json", ".gpkg", ".kml"))]
    if not readable:
        raise ValueError("No boundary file found. Upload a zipped shapefile, or the .shp, "
                         ".shx, .dbf and .prj files together, or a GeoJSON / GeoPackage / KML file.")
    if len(readable) > 1:
        warnings.append(f"Several boundary files found; using '{readable[0]}'.")
    path = os.path.join(folder, readable[0])
    if path.lower().endswith(".shp"):
        stem = path[:-4]
        missing = [e for e in (".shx", ".dbf") if not any(os.path.exists(stem + x) for x in (e, e.upper()))]
        if missing:
            raise ValueError(f"The shapefile is missing its {' and '.join(missing)} part. "
                             "Upload all the parts together, or zip them first.")
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("The file has no shapes in it.")
    if gdf.crs is None:
        minx, miny, maxx, maxy = gdf.total_bounds
        if -180 <= minx <= 180 and -90 <= miny <= 90 and maxx <= 180 and maxy <= 90:
            warnings.append("The file has no coordinate system (.prj). It looks like plain "
                            "latitude/longitude, so that is what I assumed.")
            gdf = gdf.set_crs(4326)
        else:
            raise ValueError("The file has no coordinate system (.prj file), and its numbers "
                             "are not latitude/longitude. Please include the .prj file.")
    gdf = gdf.to_crs(4326)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf, warnings


def guess_name_column(gdf):
    cols = [c for c in gdf.columns if c != "geometry"]
    for c in cols:
        if any(k in c.lower() for k in ("name", "gauge", "station", "basin", "catch", "id")):
            return c
    return cols[0] if cols else None


def sites_from_frame(gdf, name_column=None):
    used, sites = set(), []
    for i, row in enumerate(gdf.itertuples(index=False)):
        geom = row.geometry
        if name_column:
            name = getattr(row, name_column) if name_column.isidentifier() else gdf.iloc[i][name_column]
        else:
            name = f"Catchment {i + 1}"
        if geom.geom_type == "MultiPoint":
            geom = geom.centroid
        sites.append(Site(clean_name(name, used), geom))
    return sites


def points_from_table(table):
    """Turn a table of points into Name / Latitude / Longitude, finding the columns by name."""
    cols = {c.lower().strip(): c for c in table.columns}
    def find(*words):
        for low, orig in cols.items():
            if any(w in low for w in words):
                return orig
        return None
    lat, lon = find("lat"), find("lon", "lng", "long")
    if lat is None or lon is None:
        raise ValueError("No latitude and longitude columns found. Name them 'Latitude' and 'Longitude'.")
    name = find("name", "station", "gauge", "site", "id")
    out = pd.DataFrame({
        "Name": table[name].astype(str) if name else [f"Point {i + 1}" for i in range(len(table))],
        "Latitude": pd.to_numeric(table[lat], errors="coerce"),
        "Longitude": pd.to_numeric(table[lon], errors="coerce"),
    }).dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("The latitude and longitude columns have no numbers in them.")
    return out


def check_inside_india(sites):
    """Return a list of names of sites that fall completely outside the IMD grid."""
    grid_box = box(*RAIN_GRID.bounds)
    return [s.name for s in sites if not s.geometry.intersects(grid_box)]


# ---------------------------------------------------------------------------
# 3. Which grid cells, and how much weight each one gets
# ---------------------------------------------------------------------------

_BOX_CACHE = {}


def _cell_boxes(grid):
    """Shapely boxes for every cell of a grid (cached)."""
    if grid not in _BOX_CACHE:
        h = grid.step / 2
        lon, lat = np.meshgrid(grid.lons, grid.lats)
        _BOX_CACHE[grid] = (shapely.box(lon - h, lat - h, lon + h, lat + h), lat, lon)
    return _BOX_CACHE[grid]


def cells_for_site(site, var, has_data=None):
    """Grid cells used for a site.

    For a catchment or box: every cell that overlaps it, with the area of the
    overlap. Weight = overlap area / catchment area, so the weights add up to 1
    when the whole catchment sits on the grid.
    For a point: the nearest cell that has data (weight 1).

    has_data: optional True/False array (lat, lon) saying which cells carry data
    (land cells). Without it, all cells count as having data (used for previews).
    Returns a DataFrame, one row per cell.
    """
    grid = VARIABLES[var]["grid"]
    boxes, lat2d, lon2d = _cell_boxes(grid)
    if has_data is None:
        has_data = np.ones(lat2d.shape, dtype=bool)

    if site.is_point:
        py, px = site.geometry.y, site.geometry.x
        dist = np.hypot((lat2d - py), (lon2d - px) * np.cos(np.radians(py))) * KM_PER_DEGREE
        dist = np.where(has_data, dist, np.inf)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        return pd.DataFrame({
            "ilat": [i], "ilon": [j],
            "Latitude": [lat2d[i, j]], "Longitude": [lon2d[i, j]],
            "Has data": [bool(has_data[i, j])],
            "Fraction of cell inside": [np.nan],
            "Area inside (km²)": [np.nan],
            "Distance from point (km)": [round(float(dist[i, j]), 2)],
            "Weight": [1.0],
        })

    geom = site.geometry
    minx, miny, maxx, maxy = geom.bounds
    h = grid.step / 2
    near = ((lon2d + h > minx) & (lon2d - h < maxx) & (lat2d + h > miny) & (lat2d - h < maxy))
    ii, jj = np.nonzero(near)
    if ii.size == 0:
        return pd.DataFrame(columns=["ilat", "ilon", "Latitude", "Longitude", "Has data",
                                     "Fraction of cell inside", "Area inside (km²)",
                                     "Distance from point (km)", "Weight"])
    overlap = shapely.area(shapely.intersection(boxes[ii, jj], geom))   # square degrees
    keep = overlap > 0
    ii, jj, overlap = ii[keep], jj[keep], overlap[keep]
    cell_area = grid.step ** 2
    lats = lat2d[ii, jj]
    km2 = overlap * KM_PER_DEGREE ** 2 * np.cos(np.radians(lats))
    total_km2 = float((shapely.area(geom) * KM_PER_DEGREE ** 2 *
                       np.cos(np.radians(geom.representative_point().y))))
    # use the summed overlaps when the catchment lies fully on the grid (more exact)
    on_grid = shapely.area(shapely.intersection(geom, box(*grid.bounds))) / max(shapely.area(geom), 1e-12)
    if on_grid > 0.999:
        total_km2 = float(km2.sum())
    return pd.DataFrame({
        "ilat": ii, "ilon": jj,
        "Latitude": lats, "Longitude": lon2d[ii, jj],
        "Has data": has_data[ii, jj],
        "Fraction of cell inside": np.round(overlap / cell_area, 4),
        "Area inside (km²)": np.round(km2, 3),
        "Distance from point (km)": np.nan,
        "Weight": km2 / total_km2,
    })


def weighted_average(values, weights):
    """Area-weighted daily average that skips missing cells day by day.
    values: (days, cells); weights: (cells,)."""
    ok = ~np.isnan(values)
    w = np.where(ok, weights[None, :], 0.0)
    total = w.sum(axis=1)
    sums = np.nansum(np.where(ok, values, 0.0) * w, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, sums / total, np.nan)


# ---------------------------------------------------------------------------
# 4. Potential evaporation (Hargreaves, FAO-56)
# ---------------------------------------------------------------------------

def extraterrestrial_radiation(latitude, dates):
    """Daily extraterrestrial radiation Ra in MJ per m² per day (FAO-56 eq. 21)."""
    j = pd.DatetimeIndex(dates).dayofyear.values.astype(float)
    phi = np.radians(latitude)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    delta = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def hargreaves(tmax, tmin, latitude, dates):
    """Hargreaves potential evaporation in mm per day.
    PET = 0.0023 x Ra x (Tmean + 17.8) x square root of (Tmax - Tmin), Ra in mm/day."""
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    ra_mm = 0.408 * extraterrestrial_radiation(latitude, dates)
    tmean = (tmax + tmin) / 2
    pet = 0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None))
    return np.round(pet, 3)


# ---------------------------------------------------------------------------
# 5. The main job: pull the data for all sites
# ---------------------------------------------------------------------------

@dataclass
class Result:
    sites: list
    variables: list
    start_year: int
    end_year: int
    start_date: object = None                         # first day kept (a date)
    end_date: object = None                           # last day kept (a date)
    daily: dict = field(default_factory=dict)        # site name -> DataFrame
    cells: pd.DataFrame = None                        # grid cells used
    cell_values: dict = field(default_factory=dict)   # (site, var) -> DataFrame
    cell_means: dict = field(default_factory=dict)    # (site, var) -> DataFrame of long-term mean per cell
    grids: dict = field(default_factory=dict)         # var -> xarray.DataArray
    problems: list = field(default_factory=list)      # data that could not be got
    notes: list = field(default_factory=list)         # cautions about using the data
    years_missing: dict = field(default_factory=dict) # var -> list of years
    settings: dict = field(default_factory=dict)


DATE_SHIFT_TEXT = {
    0: "No shift (dates as IMD publishes them)",
    -1: "Rainfall moved one day earlier",
    1: "Rainfall moved one day later",
}


def pull_data(sites, variables, start_year, end_year, data_dir, add_pet=False,
              keep_cell_values=False, keep_grid=False, rain_shift=0,
              days_since_rain_mm=None, progress=None, backup_dir=BACKUP_DIR, parallel=3,
              start_date=None, end_date=None):
    """Download (if needed) and extract daily data for every site.

    rain_shift: 0, -1 or +1 days, to line rainfall dates up with other records.
    days_since_rain_mm: if a number, add a 'Days since rain' column using that
                        threshold (mm) for a day to count as rainy.
    progress: optional function progress(fraction 0-1, message).
    """
    say = progress or (lambda f, m: None)
    variables = list(variables)
    if add_pet:
        for v in ("tmax", "tmin"):
            if v not in variables:
                variables.append(v)
    years = list(range(int(start_year), int(end_year) + 1))
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    res = Result(sites=sites, variables=variables, start_year=int(start_year), end_year=int(end_year))
    res.settings = {"Rainfall date shift": DATE_SHIFT_TEXT.get(int(rain_shift), str(rain_shift))}
    series = {s.name: {} for s in sites}
    cell_tables = []
    total_steps = len(variables) * len(years) * 2
    step = 0

    if "rain" in variables and int(start_year) < 1951:
        res.notes.append("Rainfall before about 1950 is built from fewer rain gauges than today, "
                         "so be careful reading long-term trends that start in the early 1900s.")

    for var in variables:
        info = VARIABLES[var]
        grid = info["grid"]
        usable = [y for y in years if y >= info["first_year"]]
        if len(usable) < len(years):
            res.problems.append(f"{info['name']} starts in {info['first_year']}; "
                                f"years before that are left empty.")
        # --- download ---
        # a few years at a time: much faster than one by one, gentle on IMD's server
        from concurrent.futures import ThreadPoolExecutor, as_completed
        ready = []
        with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as pool:
            jobs = {pool.submit(download_year, var, y, data_dir, backup_dir=backup_dir): y for y in usable}
            for done, job in enumerate(as_completed(jobs), 1):
                y = jobs[job]
                try:
                    job.result()
                    ready.append(y)
                except (NotAvailableError, DownloadFailedError) as err:
                    res.problems.append(str(err))
                except OSError as err:
                    res.problems.append(f"Could not save {info['name'].lower()} for {y}: {err}")
                step += 1
                say(step / total_steps, f"Got {info['name'].lower()} for {done} of {len(usable)} years")
        ready.sort()
        step += len(years) - len(usable)
        res.years_missing[var] = [y for y in years if y not in ready]
        if not ready:
            step += len(years)
            continue

        # --- which cells (land mask taken from the first year read) ---
        first = read_year(var, ready[0], data_dir)
        has_data = ~np.all(np.isnan(first), axis=0)
        site_cells = {}
        for s in sites:
            c = cells_for_site(s, var, has_data)
            site_cells[s.name] = c
            t = c.drop(columns=["ilat", "ilon"]).copy()
            t.insert(0, "Variable", info["name"])
            t.insert(0, "Site", s.name)
            cell_tables.append(t)
            n_ok = int(c["Has data"].sum()) if len(c) else 0
            if n_ok == 0:
                res.problems.append(f"'{s.name}' has no {info['name'].lower()} grid cells with data.")
            elif not s.is_point and n_ok == 1 and var != "tmin":
                what = "temperature" if var == "tmax" else "rainfall"
                size = "about 110 km" if var == "tmax" else "about 25 km"
                res.notes.append(f"'{s.name}' gets its {what} from a single grid cell ({size} across), "
                                 f"so treat it as a value for the wider region, not the catchment itself.")
            if not s.is_point and n_ok and var != "tmin":
                covered = float(c.loc[c["Has data"], "Weight"].sum())
                if covered < 0.9:
                    res.notes.append(f"Only {covered * 100:.0f}% of '{s.name}' is covered by "
                                     f"{info['name'].lower()} cells with data (the rest is over the sea "
                                     f"or off the grid). The average uses the covered part.")

        if keep_grid:
            allc = pd.concat([c for c in site_cells.values() if len(c)])
            i0, i1 = int(allc.ilat.min()), int(allc.ilat.max()) + 1
            j0, j1 = int(allc.ilon.min()), int(allc.ilon.max()) + 1
            grid_parts = []

        # --- read year by year, keep only what is needed ---
        out = {s.name: [] for s in sites}
        cell_out = {s.name: [] for s in sites}
        cell_sum = {s.name: np.zeros(len(site_cells[s.name])) for s in sites}
        cell_n = {s.name: np.zeros(len(site_cells[s.name])) for s in sites}
        for y in years:
            say(step / total_steps, f"Reading {info['name'].lower()} for {y}")
            yd = pd.date_range(f"{y}-01-01", periods=days_in_year(y), freq="D")
            arr = first if y == ready[0] else (read_year(var, y, data_dir) if y in ready else None)
            for s in sites:
                c = site_cells[s.name]
                if len(c) == 0:
                    out[s.name].append(pd.Series(np.nan, index=yd))
                    continue
                if arr is None:
                    vals = np.full((len(yd), len(c)), np.nan, dtype="float32")
                else:
                    vals = arr[:, c.ilat.values, c.ilon.values]
                    cell_sum[s.name] += np.nansum(vals, axis=0)
                    cell_n[s.name] += np.sum(~np.isnan(vals), axis=0)
                avg = weighted_average(vals.astype("float64"), c.Weight.values.astype("float64"))
                out[s.name].append(pd.Series(avg, index=yd))
                if keep_cell_values:
                    cell_out[s.name].append(pd.DataFrame(vals, index=yd, columns=_cell_labels(c)))
            if keep_grid:
                part = (arr[:, i0:i1, j0:j1] if arr is not None
                        else np.full((len(yd), i1 - i0, j1 - j0), np.nan, dtype="float32"))
                grid_parts.append(part)
            step += 1

        shift = int(rain_shift) if var == "rain" else 0
        for s in sites:
            ser = pd.concat(out[s.name]).reindex(dates)
            if shift:
                ser = ser.shift(shift)
            series[s.name][info["column"]] = ser.round(3 if var == "rain" else 2)
            c = site_cells[s.name]
            if len(c):
                with np.errstate(invalid="ignore", divide="ignore"):
                    daily_mean = cell_sum[s.name] / cell_n[s.name]
                value = daily_mean * 365.25 if info["how"] == "total" else daily_mean
                label = (f"Mean annual {info['name'].lower()} ({info['unit']})" if info["how"] == "total"
                         else f"Mean {info['name'].lower()} ({info['unit']})")
                cm = c[["Latitude", "Longitude", "Has data", "Fraction of cell inside", "Weight"]].copy()
                cm[label] = np.round(value, 2)
                res.cell_means[(s.name, var)] = cm.reset_index(drop=True)
            if keep_cell_values and cell_out[s.name]:
                cv = pd.concat(cell_out[s.name]).reindex(dates)
                if shift:
                    cv = cv.shift(shift)
                cv = cv.round(2)
                cv.index.name = "Date"
                res.cell_values[(s.name, var)] = cv
        if keep_grid:
            import xarray as xr
            data = np.concatenate(grid_parts, axis=0)
            res.grids[var] = xr.DataArray(
                data, dims=("time", "lat", "lon"),
                coords={"time": dates, "lat": grid.lats[i0:i1], "lon": grid.lons[j0:j1]},
                name=var, attrs={"long_name": info["name"], "units": info["unit"],
                                 "source": "India Meteorological Department (IMD) Pune gridded data"})
            if shift:
                res.grids[var] = res.grids[var].shift(time=shift)
        del first

    if int(rain_shift) and "rain" in variables:
        res.notes.append(f"{DATE_SHIFT_TEXT[int(rain_shift)]}, as you asked. The first or last day "
                         f"of the period is therefore empty, so that year's yearly totals are left empty too.")

    # --- put daily tables together, add PET and days since rain ---
    for s in sites:
        df = pd.DataFrame(series[s.name], index=dates)
        if add_pet and VARIABLES["tmax"]["column"] in df and VARIABLES["tmin"]["column"] in df:
            df[PET_COLUMN] = hargreaves(df[VARIABLES["tmax"]["column"]],
                                        df[VARIABLES["tmin"]["column"]], s.centre[0], df.index)
        rain = VARIABLES["rain"]["column"]
        if days_since_rain_mm is not None and rain in df:
            df[DAYS_SINCE_RAIN] = days_since_rain(df[rain], float(days_since_rain_mm))
        df.index.name = "Date"
        res.daily[s.name] = df
    if days_since_rain_mm is not None:
        res.settings["Rainy day threshold for 'Days since rain'"] = f"{days_since_rain_mm} mm"
    res.cells = pd.concat(cell_tables, ignore_index=True) if cell_tables else pd.DataFrame()

    # --- keep only the chosen dates (whole IMD years are read, then trimmed) ---
    first = pd.Timestamp(start_date) if start_date is not None else dates[0]
    last = pd.Timestamp(end_date) if end_date is not None else dates[-1]
    res.start_date, res.end_date = first.date(), last.date()
    if first != dates[0] or last != dates[-1]:
        for name in list(res.daily):
            res.daily[name] = res.daily[name].loc[first:last]
        for key in list(res.cell_values):
            res.cell_values[key] = res.cell_values[key].loc[first:last]
        for var in list(res.grids):
            res.grids[var] = res.grids[var].sel(time=slice(first, last))
        if first.month != 1 or first.day != 1 or last.month != 12 or last.day != 31:
            res.notes.append("Your dates don't cover whole years, so months and years that are only "
                             "partly inside them are left empty in the monthly and yearly summaries.")
    say(1.0, "Done")
    return res


DAYS_SINCE_RAIN = "Days since rain"


def days_since_rain(rain, threshold_mm=1.0):
    """Count of days since the last day with at least threshold_mm of rain.
    0 on a rainy day, 1 the day after, and so on. Empty until the first rainy day
    and on days with no data."""
    out = np.full(len(rain), np.nan)
    count = np.nan
    for i, v in enumerate(rain.values):
        if np.isnan(v):
            count = np.nan
        elif v >= threshold_mm:
            count = 0
        elif not np.isnan(count):
            count += 1
        out[i] = count
    return pd.Series(out, index=rain.index)


def _cell_labels(cells):
    return [f"{la:.2f} N, {lo:.2f} E" for la, lo in zip(cells.Latitude, cells.Longitude)]


# ---------------------------------------------------------------------------
# 6. Summaries
# ---------------------------------------------------------------------------

def _how(column):
    if column == PET_COLUMN:
        return "total"
    for v in VARIABLES.values():
        if v["column"] == column:
            return v["how"]
    return "mean"


def _period_days(index, freq):
    """Number of calendar days in each month ('MS') or year ('YS') starting at index."""
    if freq == "MS":
        return pd.Series(index.days_in_month, index=index)
    return pd.Series([days_in_year(y) for y in index.year], index=index)


def _complete(series, freq):
    """True for each month or year where every calendar day has a value."""
    g = series.resample(freq)
    count = g.count()
    return count == _period_days(count.index, freq)


def _aggregate(df, freq):
    """Monthly ('MS') or yearly ('YS') totals for rain/evaporation and means for
    temperature. A period with any missing day, or only partly inside the chosen
    dates, is left empty, so totals are never quietly too small."""
    out = {}
    for col in df.columns:
        if col == DAYS_SINCE_RAIN:
            continue
        s = df[col]
        g = s.resample(freq)
        val = g.sum() if _how(col) == "total" else g.mean()
        out[col] = val.where(_complete(s, freq)).round(2)
    return pd.DataFrame(out)


def monthly_table(df):
    m = _aggregate(df, "MS")
    m.insert(0, "Month", m.index.month)
    m.insert(0, "Year", m.index.year)
    return m.reset_index(drop=True)


def yearly_table(df):
    y = _aggregate(df, "YS")
    rain = VARIABLES["rain"]["column"]
    if rain in df:
        s = df[rain]
        jjas = s[s.index.month.isin([6, 7, 8, 9])]
        g = jjas.resample("YS")
        mons = g.sum().where(_complete(s, "YS").reindex(g.sum().index, fill_value=False))
        y.insert(list(y.columns).index(rain) + 1, "Monsoon rainfall, June to September (mm)",
                 mons.reindex(y.index).round(2))
    y.insert(0, "Year", y.index.year)
    return y.reset_index(drop=True)


def site_summary(res):
    """One row per site with long-term numbers and data-coverage checks."""
    rows = []
    for s in res.sites:
        df = res.daily[s.name]
        yr = yearly_table(df)
        lat, lon = s.centre
        row = {"Site": s.name,
               "Type": "Point" if s.is_point else "Area",
               "Latitude": round(lat, 4), "Longitude": round(lon, 4),
               "Area (km²)": round(area_km2(s.geometry), 2) if not s.is_point else None}
        rain = VARIABLES["rain"]["column"]
        if rain in df:
            row["Mean annual rainfall (mm)"] = round(yr[rain].mean(), 1)
            mons = "Monsoon rainfall, June to September (mm)"
            row["Monsoon share of rainfall (%)"] = round(100 * yr[mons].sum() / yr[rain].sum(), 1) \
                if yr[rain].sum() > 0 else None
        for v in ("tmax", "tmin"):
            col = VARIABLES[v]["column"]
            if col in df:
                row[f"Mean {VARIABLES[v]['name'].lower()} (°C)"] = round(df[col].mean(), 2)
        if PET_COLUMN in df:
            row["Mean annual potential evaporation (mm)"] = round(yr[PET_COLUMN].mean(), 1)
            if rain in df and yr[rain].mean() > 0:
                row["Dryness index (evaporation ÷ rainfall)"] = round(yr[PET_COLUMN].mean() / yr[rain].mean(), 2)
        for v in res.variables:
            c = res.cells[(res.cells.Site == s.name) & (res.cells.Variable == VARIABLES[v]["name"])] \
                if len(res.cells) else pd.DataFrame()
            label = VARIABLES[v]["name"].lower()
            if v == "tmin" and "tmax" in res.variables:
                continue  # same grid as max temperature
            label = "temperature" if v in ("tmax", "tmin") else label
            row[f"Grid cells used for {label}"] = int(c["Has data"].sum()) if len(c) else 0
            if not s.is_point and len(c):
                row[f"Area with {label} data (%)"] = round(100 * c.loc[c["Has data"], "Weight"].sum(), 1)
        first = VARIABLES[res.variables[0]]["column"] if res.variables else None
        if first and first in df:
            row["Missing days (%)"] = round(100 * df[first].isna().mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# 6b. Rainfall indices (IMD and ETCCDI definitions)
# ---------------------------------------------------------------------------

RAINY_DAY_MM = 2.5        # IMD: a rainy day has 2.5 mm or more
HEAVY_MM = 64.5           # IMD: heavy rainfall
VERY_HEAVY_MM = 115.6     # IMD: very heavy rainfall
DRY_DAY_MM = 1.0          # ETCCDI: a dry day has less than 1 mm

INDEX_INFO = {
    "Rainy days (2.5 mm or more)": "Days with at least 2.5 mm of rain (IMD's rainy day).",
    "Heavy rain days (64.5 mm or more)": "Days with at least 64.5 mm (IMD's 'heavy rainfall').",
    "Very heavy rain days (115.6 mm or more)": "Days with at least 115.6 mm (IMD's 'very heavy rainfall').",
    "Wettest day (mm)": "The largest one-day rainfall in the year.",
    "Wettest 5 days (mm)": "The largest rainfall total over any 5 days in a row.",
    "Longest dry spell (days)": "The longest run of days with less than 1 mm (within the year).",
    "Longest wet spell (days)": "The longest run of days with 1 mm or more (within the year).",
    "Rain from heavy days (%)": "Share of the year's rain that fell on heavy rain days.",
}


def _longest_run(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def rain_indices(df):
    """Yearly rainfall indices for one site. A year with any missing day is left empty."""
    col = VARIABLES["rain"]["column"]
    if col not in df:
        return pd.DataFrame()
    rows = []
    for year, s in df[col].groupby(df.index.year):
        row = {"Year": year}
        if s.isna().any() or len(s) < days_in_year(year):
            row.update({k: np.nan for k in INDEX_INFO})
        else:
            v = s.values
            total = v.sum()
            heavy = v >= HEAVY_MM
            row.update({
                "Rainy days (2.5 mm or more)": int((v >= RAINY_DAY_MM).sum()),
                "Heavy rain days (64.5 mm or more)": int(heavy.sum()),
                "Very heavy rain days (115.6 mm or more)": int((v >= VERY_HEAVY_MM).sum()),
                "Wettest day (mm)": round(float(v.max()), 1),
                "Wettest 5 days (mm)": round(float(s.rolling(5).sum().max()), 1),
                "Longest dry spell (days)": _longest_run(v < DRY_DAY_MM),
                "Longest wet spell (days)": _longest_run(v >= DRY_DAY_MM),
                "Rain from heavy days (%)": round(100 * v[heavy].sum() / total, 1) if total > 0 else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6c. Trends (Mann-Kendall test and Sen's slope)
# ---------------------------------------------------------------------------

def mann_kendall(y):
    """Mann-Kendall trend test. Returns (Sen's slope per step, p-value).
    Sen's slope is the median of the slopes between every pair of points,
    so one odd year cannot pull it much."""
    from math import erfc, sqrt
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    n = len(y)
    if n < 8:
        return np.nan, np.nan
    i, j = np.triu_indices(n, 1)
    s = np.sign(y[j] - y[i]).sum()
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / sqrt(var)
    p = erfc(abs(z) / sqrt(2))
    slope = float(np.median((y[j] - y[i]) / (x[j] - x[i])))
    return slope, p


def trend_table(res):
    """Trend of every yearly value for every site."""
    rows = []
    for s in res.sites:
        yt = yearly_table(res.daily[s.name]).set_index("Year")
        ri = rain_indices(res.daily[s.name])
        if len(ri):
            yt = yt.join(ri.set_index("Year"))
        for col in yt.columns:
            slope, p = mann_kendall(yt[col].values)
            if np.isnan(slope):
                continue
            years_used = int(yt[col].notna().sum())
            mean = yt[col].mean()
            rows.append({
                "Site": s.name, "Measure": col, "Years used": years_used,
                "Long-term average": round(mean, 2),
                "Change per decade": round(slope * 10, 3),
                "Change per decade (% of average)": round(100 * slope * 10 / mean, 2) if mean else np.nan,
                "p-value": round(p, 4),
                "Clear trend? (p below 0.05)": "Yes, rising" if p < 0.05 and slope > 0
                    else ("Yes, falling" if p < 0.05 and slope < 0 else "No"),
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 7. Packing the results into a zip
# ---------------------------------------------------------------------------

def safe_file_name(name):
    return re.sub(r'[\\/:*?"<>|]+', "-", str(name)).strip() or "site"


def _csv(df, index=True):
    return df.to_csv(index=index, date_format="%Y-%m-%d").encode("utf-8-sig")


README = """IMD data pulled with the IMD Data Puller app
=============================================

Source: India Meteorological Department (IMD), Pune, gridded daily data.
  Rainfall: 0.25 x 0.25 degree grid (Pai et al., 2014).
  Maximum and minimum temperature: 1 x 1 degree grid (Srivastava et al., 2009).
Please cite IMD and these dataset papers when you use the data.

What is in this folder
----------------------
Daily data/            One file per site: date plus every variable.
Daily data by variable/ One file per variable with a column for each site (only
                        when there is more than one site).
Monthly summary.csv    Monthly totals (rainfall, evaporation) and means (temperature).
Yearly summary.csv     The same per year, plus June to September monsoon rainfall.
Summary by site.csv    Long-term averages and data-coverage checks for each site.
Rain indices.csv       Yearly rainy days, heavy rain days, wettest day, wettest
                       5 days, longest dry and wet spells (definitions below).
Trends.csv             Trend of every yearly value: change per decade and whether
                       it is a clear trend (Mann-Kendall test, Sen's slope).
Grid cells used.csv    Every IMD grid cell used, its position, how much of it lies
                       inside the site, and its weight in the average.
Long-term value per grid cell.csv
                       The long-term average of each cell, to see how rainfall or
                       temperature varies across a catchment.
Settings used.txt      Exactly what was asked for, so the pull can be repeated,
                       plus any cautions about the data.
Grid cell daily values/ (optional) Daily values of each grid cell, before averaging.
Grids/                 (optional) The cut-out grid as NetCDF files.

How the averages are made
-------------------------
Point: the value of the nearest grid cell that has data.
Area:  an area-weighted average. Each cell counts in proportion to how much of
       the site's area it covers, so a cell half inside the boundary counts half.
       On a day when a cell has no data, the other cells share its weight.
Monthly and yearly values are left empty if any day in that period is missing.

Potential evaporation (if included) uses the Hargreaves method (FAO-56):
  PET = 0.0023 x Ra x (Tmean + 17.8) x square root of (Tmax - Tmin)
  where Ra is the radiation at the top of the atmosphere for the site's latitude.
It is an estimate from temperature only; treat it as a guide, not a measurement.

Days since rain (if included): days since the last day with at least the chosen
amount of rain. 0 on a rainy day. Useful for picking dry-weather (recession) periods.

Rain indices
  Rainy day: 2.5 mm or more (IMD). Heavy: 64.5 mm or more. Very heavy: 115.6 mm or more (IMD).
  Dry spell: days in a row with less than 1 mm. Wet spell: 1 mm or more (ETCCDI).
  A year with any missing day is left empty.

Trends
  Sen's slope: the median of the slopes between every pair of years, so one odd
  year cannot pull it much. Mann-Kendall p-value: below 0.05 is usually read as a
  clear trend. Trends need at least 8 years.

Dates
  IMD's rain day runs from 8:30 am to 8:30 am Indian time. If your streamflow data
  uses a different day, the app can move rainfall one day earlier or later
  (see "Rainfall date shift" in Settings used.txt).
"""


def all_tables(res):
    """Work out every summary table once, for the screen and for the downloads."""
    monthly, yearly, indices, means = [], [], [], []
    for s in res.sites:
        df = res.daily[s.name]
        m = monthly_table(df); m.insert(0, "Site", s.name); monthly.append(m)
        y = yearly_table(df); y.insert(0, "Site", s.name); yearly.append(y)
        ri = rain_indices(df)
        if len(ri):
            ri.insert(0, "Site", s.name); indices.append(ri)
    for (site, var), cm in res.cell_means.items():
        t = cm.copy()
        t.insert(0, "Variable", VARIABLES[var]["name"]); t.insert(0, "Site", site)
        value_col = t.columns[-1]
        t = t.rename(columns={value_col: "Long-term value"})
        t.insert(len(t.columns) - 1, "Unit", VARIABLES[var]["unit"] + (" per year" if VARIABLES[var]["how"] == "total" else ""))
        means.append(t)
    return {
        "Summary by site": site_summary(res),
        "Monthly summary": pd.concat(monthly, ignore_index=True),
        "Yearly summary": pd.concat(yearly, ignore_index=True),
        "Rain indices": pd.concat(indices, ignore_index=True) if indices else pd.DataFrame(),
        "Trends": trend_table(res),
        "Grid cells used": res.cells.round(5) if res.cells is not None else pd.DataFrame(),
        "Long-term value per grid cell": pd.concat(means, ignore_index=True) if means else pd.DataFrame(),
    }


def build_zip(res, tables=None):
    """Put all outputs into a zip file and return its bytes."""
    tables = tables or all_tables(res)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Read me.txt", README)
        z.writestr("Settings used.txt", settings_text(res))
        for s in res.sites:
            z.writestr(f"Daily data/{safe_file_name(s.name)}.csv", _csv(res.daily[s.name]))
        if len(res.sites) > 1:
            for col in res.daily[res.sites[0].name].columns:
                wide = pd.DataFrame({s.name: res.daily[s.name][col] for s in res.sites})
                wide.index.name = "Date"
                z.writestr(f"Daily data by variable/{safe_file_name(col)}.csv", _csv(wide))
        for name, table in tables.items():
            if len(table):
                z.writestr(f"{name}.csv", _csv(table, index=False))
        for (site, var), cv in res.cell_values.items():
            z.writestr(f"Grid cell daily values/{safe_file_name(site)} - "
                       f"{safe_file_name(VARIABLES[var]['column'])}.csv", _csv(cv))
        for var, da in res.grids.items():
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                path = tmp.name
            da.to_dataset().to_netcdf(path, encoding={var: {"zlib": True, "complevel": 4}}
                                      if _has_netcdf4() else None)
            with open(path, "rb") as f:
                z.writestr(f"Grids/{VARIABLES[var]['name']} grid.nc", f.read())
            os.remove(path)
    return buf.getvalue()


def _sheet_name(name, used):
    name = re.sub(r"[\\/*?:\[\]]", "-", str(name))[:31]
    base, n = name, 2
    while name.lower() in used:
        suffix = f" ({n})"
        name = base[:31 - len(suffix)] + suffix
        n += 1
    used.add(name.lower())
    return name


def build_excel(res, tables=None, max_daily_sheets=25):
    """One Excel workbook: summaries first, then one daily sheet per site."""
    tables = tables or all_tables(res)
    buf = io.BytesIO()
    used = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        notes = pd.DataFrame({"Read me": README.splitlines()})
        notes.to_excel(xw, sheet_name=_sheet_name("Read me", used), index=False)
        for name, table in tables.items():
            if len(table) and len(table) < 1_000_000:
                table.to_excel(xw, sheet_name=_sheet_name(name, used), index=False)
        for s in res.sites[:max_daily_sheets]:
            df = res.daily[s.name].copy()
            df.index = df.index.date
            df.index.name = "Date"
            df.to_excel(xw, sheet_name=_sheet_name(f"Daily - {s.name}", used))
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
            ws.freeze_panes = "B2" if ws.title.startswith("Daily") else "A2"
    return buf.getvalue()


def _has_netcdf4():
    try:
        import netCDF4  # noqa: F401
        return True
    except Exception:
        return False


def settings_text(res):
    s = {
        "Pulled on": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Variables": [VARIABLES[v]["name"] for v in res.variables]
                     + (["Potential evaporation (Hargreaves)"] if any(
                         PET_COLUMN in d for d in res.daily.values()) else []),
        "Dates": f"{res.start_date:%d/%m/%Y} to {res.end_date:%d/%m/%Y}" if res.start_date else
                 f"{res.start_year} to {res.end_year}",
        "Sites": [{"Name": s.name,
                   "Type": "Point" if s.is_point else "Area",
                   "Latitude": round(s.centre[0], 5), "Longitude": round(s.centre[1], 5)}
                  for s in res.sites],
        "Years not available": {VARIABLES[v]["name"]: ys for v, ys in res.years_missing.items() if ys},
        "Data that could not be got": res.problems,
        "Cautions": res.notes,
    }
    s.update(res.settings)
    return json.dumps(s, indent=2, ensure_ascii=False)
