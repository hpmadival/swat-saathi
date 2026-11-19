"""
Tracing a catchment from an outlet point, using the Copernicus 90 m DEM.

In plain words:
  1. Download the DEM in a square around your point (25 km each way to start).
  2. Fill the pits, work out which way water flows from every cell, and how much
     land drains through each cell (pyflwdir, from Deltares).
  3. Move your point onto the biggest river within a short distance (1 km by default),
     because a gauge marked on a map is rarely exactly on the DEM's river line.
  4. Collect every cell that drains to that point: that is the catchment.
  5. If the catchment touches the edge of the square, it is cut off, so the square
     is doubled and everything is done again (up to a limit).

The result is a polygon you can use like an uploaded shapefile. It is a good
starting boundary; QSWAT / ArcSWAT will delineate again from the full DEM.
"""

import math
from dataclasses import dataclass

import numpy as np

import study_area as A
import swat_maps as M


class TraceError(Exception):
    """The catchment could not be traced."""


@dataclass
class Traced:
    polygon: object           # WGS 84 catchment boundary
    outlet: object            # WGS 84 point, moved onto the stream
    area_km2: float
    moved_m: float            # how far the point was moved to reach the stream
    search_km: float          # half-size of the square that was used
    cut_off: bool             # True if it still touched the edge at the largest square
    res: int = 90             # cell size of the final boundary (30 or 90 m)
    note: str = ""


def _square(lon, lat, half_km, epsg):
    import geopandas as gpd
    from shapely.geometry import Point, box
    p = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(epsg).iloc[0]
    m = half_km * 1000.0
    return gpd.GeoSeries([box(p.x - m, p.y - m, p.x + m, p.y + m)], crs=epsg).to_crs(4326).iloc[0], (p.x, p.y)


def _delineate(dem, grid, x, y, snap_m):
    """Run the flow routing on one DEM array (pyflwdir, Deltares).
    Returns (catchment mask, snapped x, snapped y)."""
    import pyflwdir
    flw = pyflwdir.from_dem(data=dem, nodata=M.DEM_NODATA, transform=grid.transform, latlon=False,
                            outlets="edge")
    upa = flw.upstream_area(unit="km2")
    # move the point to the cell with the largest drainage area within snap_m (the main river nearby)
    rows, cols = np.indices(dem.shape)
    cx, cy = grid.transform * (cols + 0.5, rows + 0.5)
    near = (cx - x) ** 2 + (cy - y) ** 2 <= max(snap_m, grid.res) ** 2
    if not near.any():
        raise TraceError("The point is outside the elevation data.")
    pick = np.where(near, upa, -1.0)
    idxs = np.array([int(np.argmax(pick))])
    basin = flw.basins(idxs=idxs)
    r, c = np.unravel_index(int(idxs[0]), dem.shape)
    sx, sy = grid.transform * (c + 0.5, r + 0.5)
    return basin > 0, float(sx), float(sy)


MAX_FINE_CELLS = 12_000_000      # 30 m re-trace only if the box has at most this many cells (memory)


def _mask_to_polygon(mask, grid):
    import shapely
    from rasterio.features import shapes
    from shapely.geometry import shape
    polys = [shape(geom) for geom, v in shapes(mask.astype("uint8"), mask=mask, transform=grid.transform)
             if v == 1]
    poly = shapely.union_all(polys).buffer(0)
    if poly.geom_type == "MultiPolygon":                       # keep the main piece
        poly = max(poly.geoms, key=lambda p: p.area)
    return shapely.Polygon(poly.exterior).simplify(grid.res / 2)   # no holes, smoother outline


def _touches_edge(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def trace_catchment(lat, lon, start_km=25.0, max_km=150.0, snap_m=1000.0, fine=True, reader=None, say=None):
    """Trace the catchment draining to (lat, lon). Returns a Traced result.

    Step 1 (90 m): start with a 50 km square (start_km = half its side) and double it until the
    catchment no longer touches the edge, up to max_km.
    Step 2 (30 m, if fine): re-trace inside the 90 m catchment's box plus a margin, with the
    Copernicus 30 m DEM, for a sharper boundary. Very big catchments stay at 90 m.
    """
    import geopandas as gpd
    from shapely.geometry import Point, box
    say = say or (lambda f, m: None)
    epsg = A.utm_epsg(lon, lat)
    half = float(start_km)
    tries = 0
    while True:
        square, (x, y) = _square(lon, lat, half, epsg)
        tries += 1
        say(min(0.12 * tries, 0.5), f"finding the catchment at 90 m in a {2 * half:.0f} km square")
        grid, _ = M.make_grid(square, epsg, 90)
        dem = M.build_dem(grid, reader)
        if (dem == M.DEM_NODATA).all():
            raise TraceError("There is no elevation data around this point.")
        mask, sx, sy = _delineate(dem, grid, x, y, snap_m)
        edge = _touches_edge(mask)
        if not edge or half >= max_km:
            break
        half = min(half * 2, max_km)
    if mask.sum() < 4:
        raise TraceError("The traced catchment is tiny. Move the point onto the river and try again.")
    poly = _mask_to_polygon(mask, grid)
    res_used, note = 90, ""

    if fine and not edge:
        margin = 3000.0
        for _ in range(2):
            x0, y0, x1, y1 = poly.bounds
            bx = box(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
            cells = (bx.bounds[2] - bx.bounds[0]) * (bx.bounds[3] - bx.bounds[1]) / 900.0
            if cells > MAX_FINE_CELLS:
                note = "The catchment is too big to re-trace at 30 m on the free server, so the 90 m outline is used."
                break
            say(0.6, "re-tracing the catchment at 30 m for a sharp boundary")
            bx_w = gpd.GeoSeries([bx], crs=epsg).to_crs(4326).iloc[0]
            g30, _ = M.make_grid(bx_w, epsg, 30)
            d30 = M.build_dem(g30, reader)
            # snap near the 90 m outlet (already on the main river), within two 90 m cells
            m30, sx30, sy30 = _delineate(d30, g30, sx, sy, 200.0)
            if m30.sum() < 4:
                break
            if _touches_edge(m30):
                margin *= 3
                continue
            poly, sx, sy, grid, res_used = _mask_to_polygon(m30, g30), sx30, sy30, g30, 30
            break
    say(0.97, "drawing the boundary")
    area = poly.area / 1e6
    poly_w = gpd.GeoSeries([poly], crs=epsg).to_crs(4326).iloc[0]
    outlet_w = gpd.GeoSeries([Point(sx, sy)], crs=epsg).to_crs(4326).iloc[0]
    moved = math.hypot(sx - x, sy - y)
    return Traced(poly_w, outlet_w, area, moved, half, bool(edge), res_used, note)
