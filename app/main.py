from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data, forecast

ROOT = Path(__file__).resolve().parent.parent
CACHE_TTL_S = 60 * 60
DEFAULT = dict(lat=28.6692, lon=77.4538, name="Ghaziabad")   # KIET / Delhi NCR

@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("WARM_DEFAULT", "1") == "1":
        threading.Thread(target=lambda: _safe(get_forecast, DEFAULT["lat"], DEFAULT["lon"]),
                         daemon=True).start()
    yield


app = FastAPI(title="Clean Air Walk", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

_cache: dict[tuple, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def _key(lat: float, lon: float) -> tuple:
    return (round(lat, 2), round(lon, 2))     # ~1 km cells share a forecast


def get_forecast(lat: float, lon: float) -> dict:
    k = _key(lat, lon)
    with _cache_lock:
        hit = _cache.get(k)
        if hit and time.time() - hit[0] < CACHE_TTL_S:
            return hit[1]
    res = forecast.run(lat, lon)
    with _cache_lock:
        _cache[k] = (time.time(), res)
    return res


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001 -- warm-up must never kill the server
        print("warm-up failed:", repr(e), flush=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return dict(ok=True, openaq_key=bool(os.environ.get("OPENAQ_API_KEY")))


@app.get("/api/geocode")
def api_geocode(q: str = Query(..., min_length=2, max_length=80)) -> list[dict]:
    try:
        return data.geocode(q)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"geocoding failed: {e}") from e


@app.get("/api/forecast")
def api_forecast(lat: float = Query(DEFAULT["lat"], ge=-90, le=90),
                 lon: float = Query(DEFAULT["lon"], ge=-180, le=180)) -> dict:
    try:
        return get_forecast(lat, lon)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"forecast failed: {type(e).__name__}: {e}") from e
