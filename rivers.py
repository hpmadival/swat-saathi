"""
River lines for "burning in" to the DEM in QSWAT / ArcSWAT.

In flat areas (deltas, coastal plains, irrigated plains) a DEM alone often puts the
streams in the wrong place. QSWAT and ArcSWAT can "burn" a real river map into the
DEM first, so the delineated streams follow the real rivers.

This file gets the rivers (and optionally canals and streams) mapped in
OpenStreetMap, through the free Overpass service, and cuts them to the study area.
"""

import time

import requests
from shapely.prepared import prep
from shapely.geometry import LineString

OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]

KINDS = {"river": "Rivers", "stream": "Streams", "canal": "Canals"}


class RiverError(Exception):
    """River lines could not be fetched."""


def _query(bounds, kinds):
    w, s, e, n = bounds
    pattern = "|".join(kinds)
    return (f'[out:json][timeout:180];(way["waterway"~"^({pattern})$"]({s:.5f},{w:.5f},{n:.5f},{e:.5f}););'
            f"out geom;")


def parse_overpass(data):
    """Turn an Overpass JSON reply into a list of (LineString, kind, name)."""
    out = []
    for el in data.get("elements", []):
        pts = el.get("geometry") or []
        if el.get("type") != "way" or len(pts) < 2:
            continue
        tags = el.get("tags", {})
        line = LineString([(p["lon"], p["lat"]) for p in pts])
        out.append((line, tags.get("waterway", ""), tags.get("name:en") or tags.get("name") or ""))
    return out


def fetch_rivers(area_wgs84, kinds=("river",), timeout=200, fetch=None):
    """River lines inside the study area: list of (LineString in WGS 84, kind, name)."""
    if fetch is None:
        def fetch(q):
            last = None
            for url in OVERPASS:
                for attempt in range(2):
                    try:
                        r = requests.post(url, data={"data": q}, timeout=timeout,
                                          headers={"User-Agent": "SWAT-Bharat (SWAT input tool)"})
                        if r.status_code in (429, 504):
                            time.sleep(5 * (attempt + 1))
                            last = f"busy ({r.status_code})"
                            continue
                        r.raise_for_status()
                        return r.json()
                    except (requests.RequestException, ValueError) as err:
                        last = err.__class__.__name__
            raise RiverError(f"The OpenStreetMap river service did not answer ({last}). Try again later.")
    data = fetch(_query(area_wgs84.bounds, kinds))
    lines = []
    prepared = prep(area_wgs84)
    for line, kind, name in parse_overpass(data):
        if not prepared.intersects(line):
            continue
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
