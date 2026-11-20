"""
The delineation suite: from an outlet point to a catchment with streams and subbasins,
using the Copernicus DEM and pyflwdir (Deltares).

In plain words:
  1. Download the DEM in a square around your point (a 50 km square to start).
  2. Fill the pits, work out which way water flows from every cell (D8: to one of its
     8 neighbours), and how much land drains through each cell.
  3. Move your point onto the biggest river within a short distance (1 km by default),
     because a gauge marked on a map is rarely exactly on the DEM's river line.
  4. Collect every cell that drains to that point: that is the catchment. If it touches the
     edge of the square, the square is doubled and everything is done again.
  5. Re-do it at 30 m inside the catchment's box for a sharper boundary.
  6. Network (can be redone quickly for any stream threshold, without steps 1 to 5):
       - streams: cells that drain at least the stream threshold (km²), split at junctions,
         with their Strahler order (1 = small headwater stream)
       - subbasins: the land draining to each stream segment, the way SWAT splits a catchment,
         numbered from the outlet (1) upstream, with which subbasin each one flows into
       - catchment numbers: area, perimeter, longest flow path, drainage density, shape,
         relief, slope and time of concentration

QSWAT / ArcSWAT will delineate again from the full DEM; this is a quick, good first look
that you can download and use straight away.
"""

import math
from dataclasses import dataclass, field

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
    epsg: int = 0
    stream_km2: float = 0.0   # the stream threshold of the current network
    # kept in memory so the network can be redone for another threshold without re-delineating
    flw: object = None
    upa: object = None
    mask: object = None
    dem: object = None
    grid: object = None
    poly_utm: object = None
    outlet_idx: int = 0
    networks: dict = field(default_factory=dict)      # stream threshold -> Network

    @property
    def net(self):
        """The network for the current stream threshold."""
        return network(self, self.stream_km2)

    @property
    def streams(self):
        """[(LineString WGS 84, Strahler order)] for the current threshold (for drawing)."""
        return [(g, o) for g, o, _, _ in self.net.streams]


def _square(lon, lat, half_km, epsg):
    import geopandas as gpd
    from shapely.geometry import Point, box
    p = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(epsg).iloc[0]
    m = half_km * 1000.0
    return gpd.GeoSeries([box(p.x - m, p.y - m, p.x + m, p.y + m)], crs=epsg).to_crs(4326).iloc[0], (p.x, p.y)


def _delineate(dem, grid, x, y, snap_m, target_km2=None):
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
    if target_km2:
        # the cell near the point whose drainage area best matches the expected catchment
        pick = np.where(near, -np.abs(upa - target_km2), -np.inf)
    else:
        # the cell with the largest drainage area near the point (the main river)
        pick = np.where(near, upa, -1.0)
    idxs = np.array([int(np.argmax(pick))])
    basin = flw.basins(idxs=idxs)
    r, c = np.unravel_index(int(idxs[0]), dem.shape)
    sx, sy = grid.transform * (c + 0.5, r + 0.5)
    return basin > 0, float(sx), float(sy), flw, upa


MAX_FINE_CELLS = 12_000_000      # 30 m re-trace only if the box has at most this many cells (memory)


def _join_corners(mask):
    """Water can flow diagonally (D8), so catchment cells may touch only at a corner. Where that
    happens, also count one of the two side cells, so every part of the catchment shares an edge
    with the rest and the outline comes out as one piece. The added area is negligible."""
    m = mask.copy()
    for _ in range(4):
        a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
        diag1 = a & d & ~b & ~c          # top-left and bottom-right only
        diag2 = b & c & ~a & ~d          # top-right and bottom-left only
        if not (diag1.any() or diag2.any()):
            break
        m[:-1, 1:] |= diag1
        m[:-1, :-1] |= diag2
    return m


def _mask_to_polygon(mask, grid):
    """Outline of the catchment cells, as one polygon without holes."""
    import shapely
    from rasterio.features import shapes
    from shapely.geometry import shape
    m = _join_corners(mask)
    polys = [shape(geom) for geom, v in shapes(m.astype("uint8"), mask=m, transform=grid.transform) if v == 1]
    poly = shapely.union_all(polys).buffer(0)
    if poly.geom_type == "MultiPolygon":                       # should not happen now: keep the main piece
        poly = max(poly.geoms, key=lambda p: p.area)
    return shapely.Polygon(poly.exterior).simplify(grid.res / 2)   # no holes, smoother outline


def _touches_edge(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def trace_catchment(lat, lon, start_km=25.0, max_km=150.0, snap_m=1000.0, fine=True, stream_km2=None,
                    reader=None, say=None):
    """Trace the catchment draining to (lat, lon). Returns a Traced result.

    Step 1 (90 m): start with a 50 km square (start_km = half its side) and double it until the
    catchment no longer touches the edge, up to max_km.
    Step 2 (30 m, if fine): re-trace inside the 90 m catchment's box plus a margin, with the
    Copernicus 30 m DEM, for a sharper boundary. Very big catchments stay at 90 m.
    Step 3: streams, subbasins and catchment numbers for stream_km2 (if not given, about 1% of
    the catchment); network() redoes them quickly for another threshold.
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
        mask, sx, sy, flw, upa = _delineate(dem, grid, x, y, snap_m)
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
            # near the 90 m outlet, find the 30 m cell that drains about the same area
            area90 = mask.sum() * 90 * 90 / 1e6
            try:
                m30, sx30, sy30, f30, u30 = _delineate(d30, g30, sx, sy, 400.0, target_km2=area90)
            except TraceError:
                note = "The 30 m re-trace did not work here, so the 90 m outline is used."
                break
            area30 = m30.sum() * 30 * 30 / 1e6
            if not (0.7 * area90 <= area30 <= 1.3 * area90):
                note = (f"At 30 m the catchment came out very different ({area30:,.0f} km² instead of "
                        f"{area90:,.0f} km²), so the 90 m outline is used.")
                break
            if _touches_edge(m30):
                margin *= 3
                continue
            poly, sx, sy, grid, res_used = _mask_to_polygon(m30, g30), sx30, sy30, g30, 30
            mask, flw, upa, dem = m30, f30, u30, d30
            break
    say(0.85, "finding the streams and subbasins")
    area = poly.area / 1e6
    poly_w = gpd.GeoSeries([poly], crs=epsg).to_crs(4326).iloc[0]
    outlet_w = gpd.GeoSeries([Point(sx, sy)], crs=epsg).to_crs(4326).iloc[0]
    moved = math.hypot(sx - x, sy - y)
    r, c = (~grid.transform) * (sx, sy)
    tr = Traced(poly_w, outlet_w, area, moved, half, bool(edge), res_used, note, epsg,
                flw=flw, upa=upa, mask=mask, dem=dem, grid=grid, poly_utm=poly,
                outlet_idx=int(int(c) * grid.width + int(r)))
    tr.stream_km2 = float(stream_km2) if stream_km2 else auto_stream_km2(area)
    network(tr, tr.stream_km2)
    say(0.97, "drawing the boundary")
    return tr


MAX_SUBBASINS = 400


def auto_stream_km2(area_km2):
    """A stream threshold that gives a clear network: about 1% of the catchment, between 0.5
    and 25 km² (usually 20 to 60 subbasins)."""
    return float(nice_threshold(min(max(area_km2 / 100.0, 0.5), 25.0)))


NICE = [0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750,
        1000, 1500, 2000, 3000, 5000]


def nice_threshold(x):
    return min(NICE, key=lambda v: abs(math.log(v) - math.log(max(x, 0.05))))


def threshold_choices(area_km2):
    """Stream thresholds that make sense for this catchment (from very detailed to very coarse)."""
    lo, hi = max(area_km2 / 2000.0, 0.1), max(area_km2 / 3.0, 0.2)
    return [v for v in NICE if lo <= v <= hi] or [nice_threshold(area_km2 / 100.0)]


@dataclass
class Network:
    thr: float
    streams: list        # [(LineString WGS 84, Strahler order, length km, subbasin number)]
    subbasins: list      # [dict: Subbasin, Flows into, Area (km²), ... , geometry (WGS 84)]
    outlets: list        # [(Point WGS 84, subbasin number)]
    stats: dict          # catchment numbers
    sub_grid: object     # subbasin number of every cell (0 outside the catchment)
    note: str = ""


# name, unit, plain explanation
NUMBERS = [
    ("Area", "km²", "Land that drains to the outlet."),
    ("Perimeter", "km", "Length of the boundary."),
    ("Longest flow path", "km", "The longest route water travels to the outlet."),
    ("Basin length", "km", "Straight-line distance from the outlet to the farthest point of the boundary."),
    ("Total stream length", "km", "All streams added together (above the stream threshold)."),
    ("Drainage density", "km per km²", "Stream length per km² of land: higher means more, closer streams."),
    ("Number of subbasins", "", "One per stream segment, as SWAT splits a catchment."),
    ("Highest stream order", "", "Strahler order at the outlet: 1 = small headwater streams."),
    ("Lowest elevation", "m", "Elevation at the outlet."),
    ("Highest elevation", "m", "Elevation at the highest point."),
    ("Relief", "m", "Highest minus lowest elevation."),
    ("Mean slope", "%", "Average steepness of the land."),
    ("Form factor", "", "Area ÷ basin length². Near 0.8 = round; low = long and narrow."),
    ("Circularity ratio", "", "How close the shape is to a circle (1 = a perfect circle)."),
    ("Elongation ratio", "", "1 = round; under 0.7 = elongated."),
    ("Time of concentration", "hours", "Kirpich (1940): time for water from the far end to reach the outlet."),
]


def _rc(grid, x, y):
    c, r = (~grid.transform) * (x, y)
    return int(r), int(c)


def network(tr, thr):
    """Streams, subbasins and catchment numbers for one stream threshold (km²). Cached on tr."""
    thr = float(thr)
    if thr in tr.networks:
        return tr.networks[thr]
    import geopandas as gpd
    import shapely
    from rasterio.features import shapes
    from shapely.geometry import LineString, Point, shape
    flw, upa, mask, grid, dem = tr.flw, tr.upa, tr.mask, tr.grid, tr.dem
    H, W = mask.shape
    note = ""
    smask = (upa >= thr) & mask
    feats = flw.streams(mask=smask) if smask.sum() >= 2 else []
    if len(feats) > MAX_SUBBASINS:
        note = (f"A threshold of {thr:g} km² would give {len(feats)} subbasins; the most this app makes is "
                f"{MAX_SUBBASINS}. Choose a bigger threshold.")
        feats = []
    order = flw.stream_order(type="strahler", mask=smask) if feats else None

    # each segment: its own cells inside the catchment (from its upstream start down to the junction)
    segs = []
    for f in feats:
        coords = f["geometry"]["coordinates"]
        own = coords[:-1] if len(coords) > 1 else coords        # the last point is the next segment's start
        cells, pts = [], []
        for x, y in own:
            r, c = _rc(grid, x, y)
            if not (0 <= r < H and 0 <= c < W) or not mask[r, c]:
                break                                           # left the catchment (past the outlet)
            cells.append(r * W + c)
            pts.append((x, y))
        if not cells:
            continue
        nxt = coords[-1] if len(own) == len(pts) else None    # junction below, to draw a connected line
        if nxt is not None:
            r, c = _rc(grid, *nxt)
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                pts = pts + [tuple(nxt)]
        segs.append({"outlet": cells[-1], "pts": pts, "order": int(order.ravel()[cells[-1]])})

    outlets = list(dict.fromkeys([sg["outlet"] for sg in segs] + [tr.outlet_idx]))
    basin = flw.basins(idxs=np.array(outlets, dtype="int64"))
    basin = np.where(mask, basin, 0).astype("int32")
    # number from the outlet upstream: 1 = the subbasin at the outlet, then by drainage area
    upa_flat = upa.ravel()
    rank = sorted(range(len(outlets)), key=lambda k: -upa_flat[outlets[k]])
    number = {k + 1: n + 1 for n, k in enumerate(rank)}            # basin id -> subbasin number
    lut = np.zeros(len(outlets) + 1, dtype="int32")
    for bid, num in number.items():
        lut[bid] = num
    sub = lut[basin]
    ids_ds = flw.idxs_ds
    flows_into = {}
    for k, idx in enumerate(outlets):
        ds = int(ids_ds[idx])
        flows_into[number[k + 1]] = int(sub.ravel()[ds]) if (ds != idx and 0 <= ds < sub.size) else 0

    # numbers for every subbasin
    d = dem.astype("float64")
    d[dem == M.DEM_NODATA] = np.nan
    gy, gx = np.gradient(d, grid.res)
    slope = np.sqrt(gx ** 2 + gy ** 2) * 100
    n = len(outlets)
    flat = sub.ravel()
    ok = (flat > 0) & np.isfinite(d.ravel())
    cnt = np.bincount(flat, minlength=n + 1)
    elev_sum = np.bincount(flat[ok], weights=d.ravel()[ok], minlength=n + 1)
    oks = (flat > 0) & np.isfinite(slope.ravel())
    slope_sum = np.bincount(flat[oks], weights=slope.ravel()[oks], minlength=n + 1)
    nok = np.bincount(flat[ok], minlength=n + 1)
    noks = np.bincount(flat[oks], minlength=n + 1)

    streams_utm = []
    for sg in segs:
        line = LineString(sg["pts"]) if len(sg["pts"]) >= 2 else None
        num = int(flat[sg["outlet"]])
        if line is not None and line.length > 0:
            streams_utm.append((line, sg["order"], line.length / 1000, num))
    stream_km = {}
    for _, _, ln, num in streams_utm:
        stream_km[num] = stream_km.get(num, 0) + ln

    pieces = {}
    for geom, v in shapes(sub, mask=sub > 0, transform=grid.transform):
        pieces.setdefault(int(v), []).append(shape(geom))
    subs = []
    for num in range(1, n + 1):
        if num not in pieces:
            continue
        g = shapely.union_all(pieces[num]).buffer(0).simplify(grid.res / 2)
        subs.append({"Subbasin": num, "Flows into": flows_into.get(num, 0),
                     "Area (km²)": round(cnt[num] * grid.res ** 2 / 1e6, 3),
                     "Mean elevation (m)": round(elev_sum[num] / max(nok[num], 1), 1),
                     "Mean slope (%)": round(slope_sum[num] / max(noks[num], 1), 2),
                     "Stream length (km)": round(stream_km.get(num, 0.0), 3), "geometry": g})

    # catchment numbers
    area = mask.sum() * grid.res ** 2 / 1e6
    poly = tr.poly_utm
    v = d[mask & np.isfinite(d)]
    dist = flw.stream_distance(unit="m")
    lfp = (float(np.nanmax(np.where(mask, dist, np.nan))) - float(dist.ravel()[tr.outlet_idx])) / 1000
    oy, ox = divmod(tr.outlet_idx, W)
    sx, sy = grid.transform * (ox + 0.5, oy + 0.5)
    bx, by = np.asarray(poly.exterior.xy)
    lb = float(np.max(np.hypot(bx - sx, by - sy))) / 1000
    total_len = sum(s_[2] for s_ in streams_utm)
    perim = poly.length / 1000
    lo = float(d.ravel()[tr.outlet_idx]) if np.isfinite(d.ravel()[tr.outlet_idx]) else float(np.nanmin(v))
    hi = float(np.nanmax(v)) if v.size else lo
    L_m = max(lfp * 1000, grid.res)
    tc_min = 0.0195 * L_m ** 0.77 * (max(hi - lo, 1.0) / L_m) ** -0.385          # Kirpich (1940), minutes
    sl = slope[mask & np.isfinite(slope)]
    stats = {"Area": area, "Perimeter": perim, "Longest flow path": lfp, "Basin length": lb,
             "Total stream length": total_len, "Drainage density": total_len / area if area else 0,
             "Number of subbasins": len(subs),
             "Highest stream order": int(max((s_[1] for s_ in streams_utm), default=0)),
             "Lowest elevation": lo, "Highest elevation": hi, "Relief": hi - lo,
             "Mean slope": float(sl.mean()) if sl.size else 0.0,
             "Form factor": area / lb ** 2 if lb else 0, "Circularity ratio": 4 * math.pi * area / perim ** 2,
             "Elongation ratio": (2 / lb) * math.sqrt(area / math.pi) if lb else 0,
             "Time of concentration": tc_min / 60}

    def to_w(geoms):
        return list(gpd.GeoSeries(geoms, crs=tr.epsg).to_crs(4326)) if geoms else []
    sw = to_w([s_[0] for s_ in streams_utm])
    streams = [(g, o, ln, num) for g, (_, o, ln, num) in zip(sw, streams_utm)]
    for row, g in zip(subs, to_w([r_["geometry"] for r_ in subs])):
        row["geometry"] = g
    outs = []
    for k, idx in enumerate(outlets):
        r_, c_ = divmod(idx, W)
        outs.append((Point(grid.transform * (c_ + 0.5, r_ + 0.5)), number[k + 1]))
    pw = to_w([p_ for p_, _ in outs])
    outlets_w = [(g, num) for g, (_, num) in zip(pw, outs)]
    net = Network(thr, streams, subs, outlets_w, stats, sub, note)
    tr.networks[thr] = net
    return net


def numbers_table(net):
    import pandas as pd
    rows = []
    for name, unit, what in NUMBERS:
        v = net.stats.get(name)
        if v is None:
            continue
        txt = f"{v:,.0f}" if name in ("Number of subbasins", "Highest stream order") else \
            (f"{v:,.0f}" if abs(v) >= 100 else (f"{v:,.2f}" if abs(v) < 10 else f"{v:,.1f}"))
        rows.append({"Number": name, "Value": txt, "Unit": unit, "What it means": what})
    return pd.DataFrame(rows)


def subbasin_table(net):
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "geometry"} for r in net.subbasins])
    if len(df):
        df["Flows into"] = df["Flows into"].map(lambda x: "Outlet" if x == 0 else str(x))
    return df


def delineation_files(tr, name="catchment"):
    """{file name: bytes} for the current network: shapefiles (UTM, ready for QSWAT / ArcSWAT /
    QGIS), one GeoJSON (latitude / longitude, for Google Earth or web maps), the flow direction and
    flow accumulation grids (GeoTIFF), the two tables and a README."""
    import json
    import os
    import tempfile
    import geopandas as gpd
    from shapely.geometry import mapping
    net = tr.net
    folder = tempfile.mkdtemp(prefix="delin-")
    epsg = tr.epsg
    gpd.GeoDataFrame({"Name": [name], "AreaKm2": [round(tr.area_km2, 2)]}, geometry=[tr.polygon],
                     crs=4326).to_crs(epsg).to_file(os.path.join(folder, "catchment.shp"))
    gpd.GeoDataFrame({"Name": [f"Outlet of {name}"], "MovedM": [round(tr.moved_m)]}, geometry=[tr.outlet],
                     crs=4326).to_crs(epsg).to_file(os.path.join(folder, "outlet.shp"))
    if net.streams:
        gpd.GeoDataFrame({"Order": [o for _, o, _, _ in net.streams],
                          "LengthKm": [round(l_, 3) for _, _, l_, _ in net.streams],
                          "Subbasin": [s_ for _, _, _, s_ in net.streams]},
                         geometry=[g for g, _, _, _ in net.streams], crs=4326).to_crs(epsg).to_file(
            os.path.join(folder, "streams.shp"))
    if net.subbasins:
        g = gpd.GeoDataFrame([{"Subbasin": r["Subbasin"], "FlowsInto": r["Flows into"],
                               "AreaKm2": r["Area (km²)"], "ElevM": r["Mean elevation (m)"],
                               "SlopePct": r["Mean slope (%)"], "StreamKm": r["Stream length (km)"]}
                              for r in net.subbasins], geometry=[r["geometry"] for r in net.subbasins], crs=4326)
        g.to_crs(epsg).to_file(os.path.join(folder, "subbasins.shp"))
        gpd.GeoDataFrame({"Subbasin": [n_ for _, n_ in net.outlets]}, geometry=[p_ for p_, _ in net.outlets],
                         crs=4326).to_crs(epsg).to_file(os.path.join(folder, "subbasin-outlets.shp"))
    out = {}
    for f in sorted(os.listdir(folder)):
        with open(os.path.join(folder, f), "rb") as fh:
            out[f] = fh.read()
    # flow grids (the whole box the catchment was delineated in)
    d8 = tr.flw.to_array(ftype="d8").astype("uint8")
    out["flow-direction-d8.tif"] = M._tif_bytes(d8, tr.grid, 247, "uint8")
    acc = np.where(tr.dem == M.DEM_NODATA, -9999.0, tr.upa).astype("float32")
    out["flow-accumulation-km2.tif"] = M._tif_bytes(acc, tr.grid, -9999.0, "float32")
    out["subbasins.tif"] = M._tif_bytes(net.sub_grid.astype("int32"), tr.grid, 0, "int32")
    feats = [{"type": "Feature", "properties": {"Layer": "Catchment", "Name": name,
                                                "AreaKm2": round(tr.area_km2, 2)}, "geometry": mapping(tr.polygon)},
             {"type": "Feature", "properties": {"Layer": "Outlet", "Name": f"Outlet of {name}"},
              "geometry": mapping(tr.outlet)}]
    feats += [{"type": "Feature", "properties": {"Layer": "Subbasin", "Subbasin": r["Subbasin"],
                                                 "FlowsInto": r["Flows into"], "AreaKm2": r["Area (km²)"]},
               "geometry": mapping(r["geometry"])} for r in net.subbasins]
    feats += [{"type": "Feature", "properties": {"Layer": "Stream", "Order": o, "Subbasin": s_},
               "geometry": mapping(g)} for g, o, _, s_ in net.streams]
    out["delineation.geojson"] = json.dumps({"type": "FeatureCollection", "features": feats}).encode()
    out["Subbasins.csv"] = subbasin_table(net).to_csv(index=False).encode("utf-8")
    out["Catchment numbers.csv"] = numbers_table(net).to_csv(index=False).encode("utf-8")
    out["README.txt"] = (
        f"Catchment of {name}, delineated by SWAT Saathi from the Copernicus {tr.res} m DEM\r\n"
        f"(pyflwdir, D8 flow directions). Area {tr.area_km2:,.1f} km2. Stream threshold {net.thr:g} km2:\r\n"
        f"{len(net.subbasins)} subbasins, {len(net.streams)} stream segments.\r\n"
        f"Shapefiles and GeoTIFFs: WGS 84 / UTM zone {epsg - 32600}N (EPSG:{epsg}). GeoJSON: latitude / longitude.\r\n\r\n"
        "catchment.shp            the catchment boundary\r\n"
        "outlet.shp               the outlet, moved onto the main river (use it as the outlet in QSWAT / ArcSWAT)\r\n"
        "streams.shp              stream segments: Strahler order, length, subbasin (can be used for burn-in)\r\n"
        "subbasins.shp            one subbasin per stream segment, numbered from the outlet (1) upstream;\r\n"
        "                         FlowsInto = the subbasin downstream (0 = the catchment outlet)\r\n"
        "subbasin-outlets.shp     the bottom point of every subbasin\r\n"
        "flow-direction-d8.tif    D8 flow direction (1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE; 0 pit)\r\n"
        "flow-accumulation-km2.tif land draining through each cell (km2)\r\n"
        "subbasins.tif            subbasin number of every cell\r\n"
        "Subbasins.csv            area, mean elevation, mean slope, stream length and flows-into of each subbasin\r\n"
        "Catchment numbers.csv    area, perimeter, longest flow path, drainage density, shape, relief,\r\n"
        "                         slope and time of concentration (Kirpich, 1940)\r\n\r\n"
        "QSWAT / ArcSWAT will delineate again from the DEM; use these to choose a threshold, check the result\r\n"
        "and for maps. The subbasins also work as 'one station per subbasin' for SWAT Saathi's weather.\r\n").encode()
    return out
