"""Street-level PM2.5 forecast + walk-window finder.

The global CAMS model forecasts air quality on a ~45 km grid. A real sensor on
your street often disagrees with it by a factor of two. We let TabPFN (an
open-weight tabular foundation model) learn, from the last few weeks, how the
real sensor relates to CAMS + local weather, then apply that to the 5-day
forecast. A backtest on the most recent 72 observed hours keeps us honest.
"""
from __future__ import annotations

import os
import threading
import time
import warnings

import numpy as np
import pandas as pd

from . import data

warnings.filterwarnings("ignore", message="Running on CPU with more than")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", 600))   # ~25 days of hourly context
N_ESTIMATORS = int(os.environ.get("TABPFN_N_ESTIMATORS", 4))
HOLDOUT_HOURS = 72
MIN_SENSOR_HOURS = 240
QUANTILES = [0.1, 0.5, 0.9]
WALK_HOURS = range(6, 21)       # 06:00 .. 20:00 local

# Indian National AQI breakpoints for 24h PM2.5 (µg/m³), used as hourly guide.
BANDS = [(30, "Good", "#2e9e5b"), (60, "Satisfactory", "#8cc152"), (90, "Moderate", "#e8b830"),
         (120, "Poor", "#ee8a2a"), (250, "Very poor", "#d9452f"), (1e9, "Severe", "#8e2a4f")]


def band(pm: float) -> dict:
    for hi, label, color in BANDS:
        if pm <= hi:
            return dict(label=label, color=color)
    return dict(label="Severe", color=BANDS[-1][2])


# ---------------------------------------------------------------- model
_fit_lock = threading.Lock()


def _new_model():
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    # TabPFN v2: Prior Labs License (Apache 2.0 + attribution) -- openly usable weights.
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    m.set_params(n_estimators=N_ESTIMATORS, device="cpu")
    return m


def _fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame) -> np.ndarray:
    """Returns (len(Xte), 3) quantiles in log1p space."""
    with _fit_lock:
        m = _new_model()
        m.fit(Xtr.to_numpy(np.float32), ytr.astype(np.float32))
        q = m.predict(Xte.to_numpy(np.float32), output_type="quantiles", quantiles=QUANTILES)
    return np.column_stack([np.asarray(a) for a in q])


# ---------------------------------------------------------------- features
def build_features(weather: pd.DataFrame, cams: pd.DataFrame) -> pd.DataFrame:
    df = weather.join(cams, how="inner")
    out = pd.DataFrame(index=df.index)
    h = df.index.hour
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["weekend"] = (df.index.dayofweek >= 5).astype(float)
    out["log_cams_pm25"] = np.log1p(df["cams_pm25"])
    out["log_cams_pm10"] = np.log1p(df["cams_pm10"])
    out["temp"] = df["temperature_2m"]
    out["rh"] = df["relative_humidity_2m"]
    out["rain"] = df["precipitation"]
    rad = np.deg2rad(df["wind_direction_10m"])
    out["wind_u"] = df["wind_speed_10m"] * np.sin(rad)
    out["wind_v"] = df["wind_speed_10m"] * np.cos(rad)
    # Old archive hours sometimes lack mixing height; keep it if the recent window is complete.
    if ("boundary_layer_height" in df
            and df["boundary_layer_height"].tail(MAX_TRAIN_ROWS + 200).notna().mean() > 0.95):
        out["log_blh"] = np.log1p(df["boundary_layer_height"].clip(lower=0))
    # 24h rolling CAMS captures multi-day smog build-up (stubble season).
    out["log_cams_pm25_24h"] = np.log1p(df["cams_pm25"].rolling(24, min_periods=6).mean())
    return out.dropna()


def _mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------- main entry
def run(lat: float, lon: float) -> dict:
    t0 = time.time()
    weather, utc_off, tz = data.fetch_weather(lat, lon)
    cams = data.fetch_cams(lat, lon)
    X = build_features(weather, cams)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(seconds=utc_off)
    now = now.floor("h")
    future = X[X.index >= now]
    cams_future = cams["cams_pm25"].reindex(future.index)

    sensor = data.find_sensor(lat, lon)
    y = pd.Series(dtype=float)
    if sensor:
        y = data.fetch_sensor_hours(sensor["sensor_id"], utc_off)
    hist = X[X.index < now].join(y.rename("y"), how="inner").dropna() if len(y) else pd.DataFrame()

    result = dict(lat=lat, lon=lon, timezone=tz, generated_local=str(now), sensor=sensor,
                  mode="cams", backtest=None, model="CAMS global forecast (no local sensor)")

    if sensor and len(hist) >= MIN_SENSOR_HOURS:
        feats = list(X.columns)
        # ---- backtest: train on everything before the last 72 h, score the last 72 h
        bt_tr, bt_te = hist.iloc[:-HOLDOUT_HOURS].tail(MAX_TRAIN_ROWS), hist.iloc[-HOLDOUT_HOURS:]
        q = _fit_predict(bt_tr[feats], np.log1p(bt_tr["y"].to_numpy()), bt_te[feats])
        pred_bt = np.expm1(q[:, 1])
        cams_bt = np.expm1(bt_te["log_cams_pm25"].to_numpy())
        ratio = float(np.median(bt_tr["y"] / np.expm1(bt_tr["log_cams_pm25"]).clip(lower=1)))
        truth = bt_te["y"].to_numpy()
        inside = float(np.mean((truth >= np.expm1(q[:, 0])) & (truth <= np.expm1(q[:, 2]))))
        result["backtest"] = dict(
            hours=int(len(bt_te)), train_rows=int(len(bt_tr)),
            mae_tabpfn=_mae(pred_bt, truth), mae_cams=_mae(cams_bt, truth),
            mae_cams_scaled=_mae(cams_bt * ratio, truth), interval_coverage_80=inside,
            series=dict(time=[str(t) for t in bt_te.index], truth=truth.round(1).tolist(),
                        tabpfn=pred_bt.round(1).tolist(), cams=cams_bt.round(1).tolist()))
        # ---- final fit on the most recent rows, forecast the next 5 days
        tr = hist.tail(MAX_TRAIN_ROWS)
        qf = np.expm1(_fit_predict(tr[feats], np.log1p(tr["y"].to_numpy()), future[feats]))
        lo, mid, hi = qf[:, 0], qf[:, 1], qf[:, 2]
        result.update(mode="tabpfn", model="TabPFN v2 street-level correction of CAMS",
                      train_rows=int(len(tr)), last_observed=float(hist["y"].iloc[-1]),
                      last_observed_time=str(hist.index[-1]))
    else:
        mid = cams_future.to_numpy()
        lo, hi = mid * 0.7, mid * 1.4
        if sensor:
            result["note"] = f"Sensor found but only {len(hist)} usable hours; showing CAMS."

    fc = pd.DataFrame(dict(lo=lo, pm25=mid, hi=hi, cams=cams_future.to_numpy(),
                           rain=weather["precipitation"].reindex(future.index).to_numpy(),
                           temp=weather["temperature_2m"].reindex(future.index).to_numpy()),
                      index=future.index).clip(lower=0)
    result["forecast"] = dict(time=[str(t) for t in fc.index],
                              **{c: fc[c].round(1).tolist() for c in fc.columns})
    result["windows"] = walk_windows(fc)
    result["bands"] = [dict(max=hi_ if hi_ < 1e8 else None, label=l, color=c) for hi_, l, c in BANDS]
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def walk_windows(fc: pd.DataFrame) -> list[dict]:
    """Best and worst 2-hour daylight window for each forecast day."""
    out = []
    d = fc[fc.index.hour.isin(WALK_HOURS)].copy()
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
