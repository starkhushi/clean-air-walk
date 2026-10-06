"""Data sources: Open-Meteo (weather + CAMS air-quality forecast, no key) and
OpenAQ v3 (real ground-sensor PM2.5, free key in OPENAQ_API_KEY).

All series are returned on the location's local clock (naive timestamps),
using the utc offset Open-Meteo reports for the coordinates.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

PAST_DAYS = 92          # Open-Meteo maximum
FORECAST_DAYS = 5
TIMEOUT = 30
UA = {"User-Agent": "clean-air-walk/1.0 (Hacktoberfest 2026 project)"}

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",   # mixing height: the big driver of winter smog
]


def _get(url: str, params: dict, headers: dict | None = None) -> dict:
    r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _hourly_frame(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def fetch_weather(lat: float, lon: float) -> tuple[pd.DataFrame, int, str]:
    """Hourly weather for the past 92 days plus a 5-day forecast."""
    params = dict(latitude=lat, longitude=lon, past_days=PAST_DAYS,
                  forecast_days=FORECAST_DAYS, timezone="auto")
    try:
        js = _get("https://api.open-meteo.com/v1/forecast",
                  {**params, "hourly": ",".join(WEATHER_VARS)})
    except requests.HTTPError:
        # Some model mixes lack boundary-layer height; fall back without it.
        js = _get("https://api.open-meteo.com/v1/forecast",
                  {**params, "hourly": ",".join(WEATHER_VARS[:-1])})
    return _hourly_frame(js), int(js.get("utc_offset_seconds", 0)), js.get("timezone", "GMT")


def fetch_cams(lat: float, lon: float) -> pd.DataFrame:
    """CAMS global air-quality model (~45 km grid) via Open-Meteo: history + forecast."""
    js = _get("https://air-quality-api.open-meteo.com/v1/air-quality", dict(
        latitude=lat, longitude=lon, hourly="pm2_5,pm10",
        past_days=PAST_DAYS, forecast_days=FORECAST_DAYS, timezone="auto"))
    df = _hourly_frame(js)
    return df.rename(columns={"pm2_5": "cams_pm25", "pm10": "cams_pm10"})


# ---------------------------------------------------------------- OpenAQ
OPENAQ = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2


def _openaq_key() -> str | None:
    return os.environ.get("OPENAQ_API_KEY") or None


def find_sensor(lat: float, lon: float, radius_m: int = 25_000) -> dict | None:
    """Nearest OpenAQ PM2.5 sensor that reported in the last 3 days."""
    key = _openaq_key()
    if not key:
        return None
    js = _get(f"{OPENAQ}/locations", dict(
        coordinates=f"{lat},{lon}", radius=radius_m,
        parameters_id=PM25_PARAMETER_ID, limit=50), headers={"X-API-Key": key})
    fresh_after = datetime.now(timezone.utc) - timedelta(days=3)
    best = None
    for loc in js.get("results", []):
        last = ((loc.get("datetimeLast") or {}).get("utc"))
        if not last or pd.Timestamp(last) < pd.Timestamp(fresh_after):
            continue
        sensor = next((s for s in loc.get("sensors", [])
                       if (s.get("parameter") or {}).get("id") == PM25_PARAMETER_ID), None)
        if sensor is None:
            continue
        dist = loc.get("distance")
        if dist is None:
            c = loc.get("coordinates") or {}
            dist = 111_000 * ((c.get("latitude", lat) - lat) ** 2 + (c.get("longitude", lon) - lon) ** 2) ** 0.5
        cand = dict(sensor_id=sensor["id"], location_id=loc["id"], name=loc.get("name") or "OpenAQ sensor",
                    provider=((loc.get("provider") or {}).get("name")), distance_m=float(dist))
        if best is None or cand["distance_m"] < best["distance_m"]:
            best = cand
    return best


def fetch_sensor_hours(sensor_id: int, utc_offset_s: int, days: int = PAST_DAYS) -> pd.Series:
    """Hourly PM2.5 (µg/m³) for one sensor, re-indexed to local time."""
    key = _openaq_key()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    rows, page = [], 1
    while page <= 10:
        js = _get(f"{OPENAQ}/sensors/{sensor_id}/hours", dict(
            datetime_from=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            datetime_to=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=1000, page=page), headers={"X-API-Key": key})
        res = js.get("results", [])
        for m in res:
            t = ((m.get("period") or {}).get("datetimeFrom") or {}).get("utc")
            v = m.get("value")
            if t is not None and v is not None:
                rows.append((t, float(v)))
        if len(res) < 1000:
            break
        page += 1
    if not rows:
        return pd.Series(dtype=float, name="sensor_pm25")
    s = pd.Series({pd.Timestamp(t): v for t, v in rows}, name="sensor_pm25").sort_index()
    s.index = (s.index.tz_convert("UTC") + pd.Timedelta(seconds=utc_offset_s)).tz_localize(None).floor("h")
    s = s[(s > 0) & (s < 1500)]              # drop sensor glitches
    return s.groupby(level=0).mean()


def geocode(q: str, count: int = 5) -> list[dict]:
    js = _get("https://geocoding-api.open-meteo.com/v1/search", dict(name=q, count=count))
    return [dict(name=r["name"], admin=r.get("admin1"), country=r.get("country"),
                 lat=r["latitude"], lon=r["longitude"]) for r in js.get("results", [])]
