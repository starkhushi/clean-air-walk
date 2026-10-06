from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data, forecast, health, llm, sky

ROOT = Path(__file__).resolve().parent.parent
CACHE_TTL_S = 30 * 60
DEFAULT = dict(lat=28.6692, lon=77.4538, name="Ghaziabad")   # Delhi NCR


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001 -- warm-up must never kill the server
        print("warm-up failed:", repr(e), flush=True)


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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return dict(ok=True, openaq_key=bool(os.environ.get("OPENAQ_API_KEY")),
                llm_backend=llm.BACKEND,
                llm_model=llm.GEMINI_MODEL if llm.BACKEND == "gemini" else llm.LLM_MODEL,
                llm_ready=bool(os.environ.get("GEMINI_API_KEY")) if llm.BACKEND == "gemini" else True)


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


class Ask(BaseModel):
    question: str = Field(..., min_length=2, max_length=800)
    lat: float = Field(DEFAULT["lat"], ge=-90, le=90)
    lon: float = Field(DEFAULT["lon"], ge=-180, le=180)
    place: str = Field(DEFAULT["name"], max_length=80)
    history: list[dict] = Field(default_factory=list, max_length=12)
    profile: dict = Field(default_factory=dict)


class Sky(BaseModel):
    image: str = Field(..., max_length=3_500_000)      # data URL, resized in the browser
    lat: float = Field(DEFAULT["lat"], ge=-90, le=90)
    lon: float = Field(DEFAULT["lon"], ge=-180, le=180)
    lang: str = Field("en", max_length=8)


@app.post("/api/sky")
def api_sky(body: Sky) -> dict:
    try:
        fc = get_forecast(body.lat, body.lon)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"forecast failed: {e}") from e
    try:
        return sky.check(body.image, fc, body.lang)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


class Plan(BaseModel):
    lat: float = Field(DEFAULT["lat"], ge=-90, le=90)
    lon: float = Field(DEFAULT["lon"], ge=-180, le=180)
    lang: str = Field("en", max_length=8)
    profile: dict = Field(default_factory=dict)


@app.post("/api/exercise")
def api_exercise(body: Plan) -> dict:
    try:
        return health.plan(get_forecast(body.lat, body.lon), body.profile, body.lang)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"exercise plan failed: {e}") from e


@app.post("/api/indoor-plan")
def api_indoor_plan(body: Plan) -> StreamingResponse:
    try:
        fc = get_forecast(body.lat, body.lon)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"forecast failed: {e}") from e
    lang = body.lang if body.lang in ("en", "hi", "hinglish") else "en"
    return StreamingResponse(health.stream_indoor_plan(fc, body.profile, lang),
                             media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- installable app (PWA): manifest + service worker must be served from the root scope
@app.get("/manifest.webmanifest")
def manifest() -> FileResponse:
    return FileResponse(ROOT / "static" / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker() -> FileResponse:
    return FileResponse(ROOT / "static" / "sw.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


@app.post("/api/ask")
def api_ask(body: Ask) -> StreamingResponse:
    try:
        fc = get_forecast(body.lat, body.lon)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"forecast failed: {e}") from e
    context = llm.build_context(body.place, fc)

    def gen():
        try:
            yield from llm.stream_answer(body.question, context, fc, body.history, body.profile)
        except Exception as e:  # noqa: BLE001
            print("ask failed:", repr(e), flush=True)
            yield "\n\nSorry, something went wrong talking to Gemma. Please try again."

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
