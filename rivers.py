"""
River lines for "burning in" to the DEM in QSWAT / ArcSWAT.

In flat areas (deltas, coastal plains, irrigated plains) a DEM alone often puts the
streams in the wrong place. QSWAT and ArcSWAT can "burn" a real river map into the
DEM first, so the delineated streams follow the real rivers.

This file gets the rivers (and optionally canals and streams) mapped in
OpenStreetMap, through the free Overpass service, and cuts them to the study area.
"""

import math
import time

import requests
from shapely.prepared import prep
from shapely.geometry import LineString, box

OVERPASS = ["https://overpass-api.de/api/interpreter",          # main server (Germany)
            "https://overpass.private.coffee/api/interpreter",   # volunteer mirrors
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
CHUNK_DEG = 0.5          # ask for at most half a degree (about 55 km) at a time
TIME_LIMIT = 150         # give up on river lines after this many seconds in total

KINDS = {"river": "Rivers", "stream": "Streams", "canal": "Canals"}


class RiverError(Exception):
    """River lines could not be fetched."""


def _query(bounds, kinds):
    w, s, e, n = bounds
    pattern = "|".join(kinds)
    return (f'[out:json][timeout:90];way["waterway"~"^({pattern})$"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
            f"out geom qt;")


def parse_overpass(data):
    """Turn an Overpass JSON reply into a list of (way id, LineString, kind, name)."""
    out = []
    for el in data.get("elements", []):
        pts = el.get("geometry") or []
        if el.get("type") != "way" or len(pts) < 2:
            continue
        tags = el.get("tags", {})
        line = LineString([(p["lon"], p["lat"]) for p in pts])
        out.append((el.get("id"), line, tags.get("waterway", ""), tags.get("name:en") or tags.get("name") or ""))
    return out


def _ask_servers(q, deadline):
    """Try each Overpass server in turn; a busy server is skipped at once rather than waited for."""
    last = "no answer"
    for rnd in range(3):
        for url in OVERPASS:
            if time.time() > deadline:
                raise RiverError(f"The OpenStreetMap river service is too busy right now ({last}). "
                                 f"Try again in a few minutes with the 'Try river lines again' button.")
            try:
                r = requests.post(url, data={"data": q}, timeout=min(100, max(10, deadline - time.time())),
                                  headers={"User-Agent": "SWAT-Saathi (SWAT input tool)"})
                if r.status_code == 200:
                    return r.json()
                last = f"busy ({r.status_code})" if r.status_code in (429, 502, 503, 504) else f"error {r.status_code}"
            except (requests.RequestException, ValueError) as err:
                last = err.__class__.__name__
            time.sleep(1)
        time.sleep(5 * (rnd + 1))
    raise RiverError(f"The OpenStreetMap river service did not answer ({last}). Try again in a few minutes "
                     f"with the 'Try river lines again' button.")


def _chunks(bounds):
    w, s, e, n = bounds
    nx, ny = max(1, math.ceil((e - w) / CHUNK_DEG)), max(1, math.ceil((n - s) / CHUNK_DEG))
    dx, dy = (e - w) / nx, (n - s) / ny
    return [(w + i * dx, s + j * dy, w + (i + 1) * dx, s + (j + 1) * dy) for i in range(nx) for j in range(ny)]


def fetch_rivers(area_wgs84, kinds=("river",), fetch=None, say=None):
    """River lines inside the study area: list of (LineString in WGS 84, kind, name).
    Big areas are asked for in pieces; a line that crosses pieces is kept once."""
    say = say or (lambda f, m: None)
    deadline = time.time() + TIME_LIMIT
    fetch = fetch or (lambda q: _ask_servers(q, deadline))
    prepared = prep(area_wgs84)
    seen, lines = set(), []
    pieces = [c for c in _chunks(area_wgs84.bounds) if prepared.intersects(box(*c))]
    for k, c in enumerate(pieces, 1):
        say((k - 1) / len(pieces), f"asking OpenStreetMap, piece {k} of {len(pieces)}")
        for wid, line, kind, name in parse_overpass(fetch(_query(c, kinds))):
            if wid in seen or not prepared.intersects(line):
                continue
            seen.add(wid)
            cut = line.intersection(area_wgs84)
            for part in getattr(cut, "geoms", [cut]):
                if part.geom_type == "LineString" and part.length > 0:
                    lines.append((part, kind, name))
    return lines


def rivers_shapefile(lines, epsg):
    """{file name: bytes} for rivers.shp in the maps' UTM zone."""
    import os
    import tempfile
    import geopandas as gpd
    folder = tempfile.mkdtemp(prefix="riv-")
    gdf = gpd.GeoDataFrame({"Type": [k for _, k, _ in lines], "Name": [n[:80] for _, _, n in lines]},
                           geometry=[g for g, _, _ in lines], crs=4326).to_crs(epsg)
    gdf["LengthKm"] = (gdf.length / 1000).round(2)
    gdf.to_file(os.path.join(folder, "rivers.shp"), encoding="utf-8")
    out = {}
    for f in sorted(os.listdir(folder)):
        with open(os.path.join(folder, f), "rb") as fh:
            out[f] = fh.read()
    return out
