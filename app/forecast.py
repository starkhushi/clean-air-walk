"""PM2.5 forecast, walk windows and an honest accuracy check.

The forecast is the Copernicus CAMS global model (via Open-Meteo). Before
settling on it we tried to beat it with TabPFN v2 using real CPCB sensors in
Ghaziabad, Delhi and Noida (6 rolling 72-hour backtests each, Sep 18 - Oct 2,
2026). Raw CAMS averaged 19.9 µg/m³ error; every TabPFN variant was worse
(22.3 - 23.1). So the app uses CAMS and shows, for your nearest sensor, how
close CAMS has been over the last 72 hours.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import data

WALK_HOURS = range(6, 21)       # windows start 06:00 .. 18:00, end by 20:00
ACCURACY_HOURS = 72

# Indian National AQI breakpoints for PM2.5 (µg/m³), used as an hourly guide.
BANDS = [(30, "Good", "#2e9e5b"), (60, "Satisfactory", "#8cc152"), (90, "Moderate", "#e8b830"),
         (120, "Poor", "#ee8a2a"), (250, "Very poor", "#d9452f"), (1e9, "Severe", "#8e2a4f")]


def band(pm: float) -> dict:
    for hi, label, color in BANDS:
        if pm <= hi:
            return dict(label=label, color=color)
    return dict(label="Severe", color=BANDS[-1][2])


def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run(lat: float, lon: float) -> dict:
    t0 = time.time()
    weather, utc_off, tz = data.fetch_weather(lat, lon)
    cams = data.fetch_cams(lat, lon)
    now = (pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(seconds=utc_off)).floor("h")

    fc = pd.DataFrame(dict(
        pm25=cams["cams_pm25"],
        rain=weather["precipitation"].reindex(cams.index),
        temp=weather["temperature_2m"].reindex(cams.index),
    ))
    fc = fc[fc.index >= now].dropna(subset=["pm25"])

    result = dict(lat=lat, lon=lon, timezone=tz, generated_local=str(now),
                  model="Copernicus CAMS global forecast", sensor=None, accuracy=None, now_reading=None)

    # ---- nearest real sensor: current reading + how good CAMS has been there
    try:
        sensor = data.find_sensor(lat, lon)
    except Exception as e:  # noqa: BLE001 -- the forecast must work without OpenAQ
        sensor, result["sensor_error"] = None, repr(e)
    if sensor:
        result["sensor"] = sensor
        y = data.fetch_sensor_hours(sensor["sensor_id"], utc_off, days=14)
        if len(y):
            result["now_reading"] = dict(pm25=round(float(y.iloc[-1]), 1), time=str(y.index[-1]),
                                         age_hours=round((now - y.index[-1]) / pd.Timedelta(hours=1), 1),
                                         **band(float(y.iloc[-1])))
            both = pd.concat([y.rename("sensor"), cams["cams_pm25"].rename("cams")], axis=1).dropna()
            both = both[both.index < now].tail(ACCURACY_HOURS)
            if len(both) >= 24:
                result["accuracy"] = dict(
                    hours=int(len(both)), mae=round(_mae(both["cams"], both["sensor"]), 1),
                    mean_sensor=round(float(both["sensor"].mean()), 1),
                    bias=round(float((both["cams"] - both["sensor"]).mean()), 1),
                    series=dict(time=[str(t) for t in both.index],
                                sensor=both["sensor"].round(1).tolist(),
                                cams=both["cams"].round(1).tolist()))

    result["forecast"] = dict(time=[str(t) for t in fc.index],
                              **{c: fc[c].round(1).tolist() for c in fc.columns})
    result["windows"] = walk_windows(fc)
    result["slots"] = day_slots(fc)
    result["bands"] = [dict(max=hi if hi < 1e8 else None, label=l, color=c) for hi, l, c in BANDS]
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def walk_windows(fc: pd.DataFrame) -> list[dict]:
    """Best and worst 2-hour daylight window for each forecast day."""
    out = []
    d = fc[fc.index.hour.isin(WALK_HOURS)]
    for day, g in d.groupby(d.index.date):
        g = g.sort_index()
        if len(g) < 2:
            continue
        roll = g["pm25"].rolling(2).mean().shift(-1).iloc[:-1]
        wet = (g["rain"].rolling(2).max().shift(-1).iloc[:-1] > 0.5)
        dry = roll[~wet] if (~wet).any() else roll
        best_t, worst_t = dry.idxmin(), roll.idxmax()
        best_pm, worst_pm = float(dry.min()), float(roll.max())
        out.append(dict(
            date=str(day), weekday=pd.Timestamp(day).strftime("%a %d %b"),
            best=dict(start=best_t.strftime("%H:%M"), end=(best_t + pd.Timedelta(hours=2)).strftime("%H:%M"),
                      pm25=round(best_pm, 1), temp=round(float(g.loc[best_t, "temp"]), 1), **band(best_pm)),
            worst=dict(start=worst_t.strftime("%H:%M"), end=(worst_t + pd.Timedelta(hours=2)).strftime("%H:%M"),
                       pm25=round(worst_pm, 1), **band(worst_pm)),
            day_mean=round(float(g["pm25"].mean()), 1)))
    return out


def day_slots(fc: pd.DataFrame) -> list[dict]:
    """Every remaining 2-hour daylight slot (06-08, 08-10, ... 18-20) per day."""
    out = []
    d = fc[fc.index.hour.isin(WALK_HOURS)]
    for day, g in d.groupby(d.index.date):
        slots = []
        for start in range(6, 20, 2):
            h = g[(g.index.hour >= start) & (g.index.hour < start + 2)]
            if len(h) == 0:
                continue
            pm = float(round(h["pm25"].mean()))     # whole numbers, so label and number agree
            slots.append(dict(start=f"{start:02d}:00", end=f"{start + 2:02d}:00", pm25=pm,
                              temp=round(float(h["temp"].mean()), 1), rain=round(float(h["rain"].sum()), 1),
                              **band(pm)))
        if slots:
            out.append(dict(date=str(day), weekday=pd.Timestamp(day).strftime("%a %d %b"), slots=slots))
    return out
