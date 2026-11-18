"""
The study area: your places plus a safety margin (buffer) around them.

In plain words:
  - a point becomes a circle or a square of the size you choose
  - a shape (catchment, subbasins) gets a margin of a few km all round,
    either following its outline or as a rectangle around it
  - everything is joined into one study area; maps and weather are fetched for it

Buffers are worked out in metres in the right UTM zone, so "5 km" really is 5 km.
"""

import math

import geopandas as gpd
import shapely
from shapely.geometry import box


def utm_epsg(lon, lat):
    """EPSG code of the WGS 84 UTM zone for a place (India is zones 42 to 47 north)."""
    zone = int(math.floor((lon + 180) / 6)) + 1
    return (32600 if lat >= 0 else 32700) + zone


def buffer_place(geom, km, point_shape="Circle", shape_style="Follow the outline", epsg=None):
    """Buffer one place (WGS 84 geometry) by km. Returns a WGS 84 polygon."""
    c = geom.representative_point()
    epsg = epsg or utm_epsg(c.x, c.y)
    g = gpd.GeoSeries([geom], crs=4326).to_crs(epsg).iloc[0]
    m = float(km) * 1000.0
    if g.geom_type == "Point":
        if m <= 0:
            m = 1000.0                                   # a point needs some area
        out = (box(g.x - m, g.y - m, g.x + m, g.y + m) if point_shape == "Square"
               else g.buffer(m, quad_segs=32))
    elif shape_style == "Rectangle":
        x0, y0, x1, y1 = g.bounds
        out = box(x0 - m, y0 - m, x1 + m, y1 + m)
    else:
        out = g.buffer(m, quad_segs=16) if m > 0 else g
    return gpd.GeoSeries([out], crs=epsg).to_crs(4326).iloc[0]


def study_area(sites, shape_km=5.0, point_km=10.0, point_shape="Circle",
               shape_style="Follow the outline"):
    """All places, buffered and joined. Returns (WGS 84 geometry, UTM EPSG code)."""
    if not sites:
        return None, None
    allg = shapely.union_all([s.geometry for s in sites])
    c = allg.representative_point()
    epsg = utm_epsg(c.x, c.y)
    parts = [buffer_place(s.geometry, point_km if s.is_point else shape_km, point_shape, shape_style, epsg)
             for s in sites]
    area = shapely.make_valid(shapely.union_all(parts))
    return area, epsg


def area_km2(geom, epsg):
    return float(gpd.GeoSeries([geom], crs=4326).to_crs(epsg).area.iloc[0] / 1e6)
