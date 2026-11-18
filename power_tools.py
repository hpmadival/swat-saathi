"""
Getting the weather that IMD's gridded data does not have.

IMD gives rainfall and temperature only. SWAT also wants relative humidity,
solar radiation and wind speed, so this file fetches them from NASA POWER
(free, no login, daily from 1 January 1981, worldwide), and finds the ground
elevation of each weather station.

In plain words:
  - fetch_power()      one request per location, returns a daily table
  - fetch_elevations() ground height (m) for many locations at once
"""

import datetime
import time

import numpy as np
import pandas as pd
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_FIRST_DAY = datetime.date(1981, 1, 1)

# NASA POWER name -> (our column name, how to convert)
POWER_PARAMETERS = {
    "RH2M": "Relative humidity (fraction)",           # POWER gives %, SWAT wants 0 to 1
    "ALLSKY_SFC_SW_DWN": "Solar radiation (MJ/m²/day)",  # AG community gives MJ/m²/day
    "WS2M": "Wind speed (m/s)",                        # wind at 2 m height
    "T2MDEW": "Dew point (°C)",                        # used for the weather generator
}
RH, SOLAR, WIND, DEW = POWER_PARAMETERS.values()

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"


class PowerError(Exception):
    """NASA POWER could not give the data."""


def power_period(start_date, end_date):
    """The part of the chosen period that NASA POWER covers (it starts in 1981).
    Returns (first, last) or None when there is no overlap."""
    first = max(pd.Timestamp(start_date).date(), POWER_FIRST_DAY)
    last = min(pd.Timestamp(end_date).date(), datetime.date.today() - datetime.timedelta(days=7))
    return (first, last) if first <= last else None


def fetch_power(lat, lon, start_date, end_date, timeout=120, tries=3):
    """Daily humidity, solar radiation, wind and dew point for one place.

    Returns (DataFrame indexed by date, elevation in m or None).
    Days NASA POWER has no value for are left empty (NaN).
    """
    period = power_period(start_date, end_date)
    if period is None:
        raise PowerError("NASA POWER starts on 1 January 1981, so it has nothing for this period.")
    first, last = period
    params = {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": "AG",
        "longitude": round(float(lon), 4),
        "latitude": round(float(lat), 4),
        "start": first.strftime("%Y%m%d"),
        "end": last.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "LST",
    }
    last_error = None
    for attempt in range(tries):
        try:
            reply = requests.get(POWER_URL, params=params, timeout=timeout)
            if reply.status_code == 429:          # asked to slow down
                time.sleep(5 * (attempt + 1))
                last_error = "NASA POWER asked us to slow down"
                continue
            reply.raise_for_status()
            data = reply.json()
            break
        except (requests.RequestException, ValueError) as err:
            last_error = err.__class__.__name__
            time.sleep(2 * (attempt + 1))
    else:
        raise PowerError(f"Could not get NASA POWER data for {lat:.2f} N, {lon:.2f} E ({last_error}).")
    return parse_power(data, first, last)


def parse_power(data, first, last):
    """Turn NASA POWER's JSON reply into a daily table (separate so it can be tested)."""
    props = data.get("properties", {}).get("parameter", {})
    fill = data.get("header", {}).get("fill_value", -999)
    dates = pd.date_range(first, last, freq="D")
    table = pd.DataFrame(index=dates)
    for key, column in POWER_PARAMETERS.items():
        values = props.get(key, {})
        s = pd.Series({pd.Timestamp(k): v for k, v in values.items()}, dtype="float64")
        s = s.where(s > fill + 1)                     # -999 means no value
        table[column] = s.reindex(dates)
    table[RH] = (table[RH] / 100.0).clip(0.01, 1.0)
    table[WIND] = table[WIND].clip(lower=0)
    table[SOLAR] = table[SOLAR].clip(lower=0)
    table.index.name = "Date"
    coords = data.get("geometry", {}).get("coordinates", [])
    elevation = float(coords[2]) if len(coords) >= 3 and coords[2] is not None else None
    return table, elevation


def fetch_power_many(points, start_date, end_date, parallel=3, progress=None):
    """fetch_power for many (lat, lon) points, a few at a time.
    Returns ({(lat, lon): table}, {(lat, lon): elevation}, [problems])."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    say = progress or (lambda f, m: None)
    tables, elevations, problems = {}, {}, []
    points = list(dict.fromkeys(points))
    if not points:
        return tables, elevations, problems
    with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as pool:
        jobs = {pool.submit(fetch_power, la, lo, start_date, end_date): (la, lo) for la, lo in points}
        for n, job in enumerate(as_completed(jobs), 1):
            key = jobs[job]
            try:
                tables[key], elevations[key] = job.result()
            except PowerError as err:
                problems.append(str(err))
            say(n / len(points), f"NASA POWER: {n} of {len(points)} places")
    return tables, elevations, problems


def fetch_elevations(points, timeout=60):
    """Ground elevation (m) for many (lat, lon) points, from the Copernicus 90 m DEM
    via the free Open-Meteo service (100 points per request).
    Returns {(lat, lon): elevation}; points that fail are simply missing."""
    out = {}
    points = list(dict.fromkeys(points))
    for i in range(0, len(points), 100):
        chunk = points[i:i + 100]
        try:
            reply = requests.get(ELEVATION_URL, timeout=timeout, params={
                "latitude": ",".join(f"{la:.4f}" for la, _ in chunk),
                "longitude": ",".join(f"{lo:.4f}" for _, lo in chunk)})
            reply.raise_for_status()
            heights = reply.json().get("elevation", [])
        except (requests.RequestException, ValueError):
            continue
        for p, h in zip(chunk, heights):
            if h is not None and np.isfinite(h):
                out[p] = float(h)
    return out
