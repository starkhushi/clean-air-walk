"""Gemma assistant: turns the forecast into a plan for the person's actual day.

Backends (pick with LLM_BACKEND):
  gemini  - Gemma 4 served by Google AI Studio (GEMINI_API_KEY, free tier).
  openai  - any OpenAI-compatible server running Gemma yourself, e.g.
            llama.cpp `llama-server` or Ollama (LLM_BASE_URL, LLM_MODEL).
The prompt, data and app are identical either way: Gemma's weights are open,
so you can move it onto your own hardware without changing a line.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime

import requests

BACKEND = os.environ.get("LLM_BACKEND", "gemini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")   # Ollama default
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
MAX_HISTORY = 6

SYSTEM = """You are Clean Air Walk, a friendly assistant that helps people in India choose when to go outside, based on air quality.

Rules:
- Use ONLY the forecast table you are given. Never invent numbers. Only suggest slots that appear in the table (past hours are already removed).
- Respect the person's constraints exactly: days, times they are free, how long the activity lasts. If an activity is longer than 2 hours, combine neighbouring slots and quote the worst of them.
- Recommend ONE specific day and time, with the expected PM2.5 and its band. Then give one backup option.
- Lower PM2.5 is better. Bands: Good ≤30, Satisfactory 31-60, Moderate 61-90, Poor 91-120, Very poor 121-250, Severe >250 (India NAQI).
- If the person mentions asthma, heart or lung problems, pregnancy, small children or elderly people, prefer Good or Satisfactory slots. If none fit their constraints, say so honestly and suggest a lighter activity, a well-fitted N95 mask, or another day.
- Mention heat if a slot is above 33°C and rain if the table shows it.
- Reply in the same language and script the person writes in (English, Hindi in Devanagari, or Hinglish).
- Be warm and brief: at most 90 words. No tables. You can use **bold** for the main pick.
- You are not a doctor. For serious symptoms, tell them to follow medical advice."""


def build_context(place: str, fc: dict) -> str:
    now = datetime.fromisoformat(fc["generated_local"])
    lines = [f"Place: {place}", f"Local time now: {now.strftime('%a %d %b %Y, %H:%M')}"]
    nr = fc.get("now_reading")
    if nr and nr["age_hours"] <= 24:          # an old reading would mislead more than help
        age = f"{nr['age_hours']:.0f} hours ago" if nr["age_hours"] >= 2 else "within the last hour"
        lines.append(f"Latest real sensor reading ({fc['sensor']['name']}): {nr['pm25']:.0f} µg/m³ "
                     f"({nr['label']}), measured {age}.")
    acc = fc.get("accuracy")
    if acc:
        direction = "lower" if acc["bias"] < 0 else "higher"
        lines.append(f"In the latest {acc['hours']} hours with sensor readings, the forecast was off by "
                     f"{acc['mae']:.0f} µg/m³ on average and ran {abs(acc['bias']):.0f} {direction} than reality.")
    lines.append("Forecast PM2.5 (µg/m³) for 2-hour daylight slots, with temperature and rain:")
    for d in fc["slots"]:
        cells = []
        for s in d["slots"]:
            extra = f", rain {s['rain']}mm" if s["rain"] > 0.2 else ""
            cells.append(f"{s['start'][:2]}-{s['end'][:2]} {s['pm25']:.0f} {s['label']} {s['temp']:.0f}°C{extra}")
        lines.append(f"{d['weekday']}: " + " | ".join(cells))
    return "\n".join(lines)


def _clean_history(history: list[dict] | None) -> list[dict]:
    out = []
    for m in (history or [])[-MAX_HISTORY:]:
        role, text = m.get("role"), str(m.get("content", ""))[:1500]
        if role in ("user", "assistant") and text.strip():
            out.append(dict(role=role, content=text))
    return out


def stream_answer(question: str, context: str, history: list[dict] | None = None) -> Iterator[str]:
    history = _clean_history(history)
    user_msg = f"{context}\n\nQuestion: {question.strip()[:800]}"
    if BACKEND == "openai":
        yield from _stream_openai(history, user_msg)
    else:
        yield from _stream_gemini(history, user_msg)


def _stream_gemini(history: list[dict], user_msg: str) -> Iterator[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        yield "The assistant is not configured yet: set GEMINI_API_KEY on the server."
        return
    contents = [dict(role="model" if m["role"] == "assistant" else "user", parts=[dict(text=m["content"])])
                for m in history]
    contents.append(dict(role="user", parts=[dict(text=user_msg)]))
    body = dict(contents=contents,
                systemInstruction=dict(parts=[dict(text=SYSTEM)]),
                generationConfig=dict(temperature=0.3, maxOutputTokens=600,
                                      thinkingConfig=dict(thinkingLevel="minimal")))
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
           f":streamGenerateContent?alt=sse")
    with requests.post(url, json=body, headers={"x-goog-api-key": key}, stream=True, timeout=90) as r:
        if r.status_code != 200:
            yield f"Sorry, Gemma is unavailable right now (HTTP {r.status_code}). Please try again in a minute."
            print("gemini error", r.status_code, r.text[:500], flush=True)
            return
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                js = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            for cand in js.get("candidates", []):
                for part in (cand.get("content") or {}).get("parts", []):
                    if part.get("text") and not part.get("thought"):
                        yield part["text"]


def _stream_openai(history: list[dict], user_msg: str) -> Iterator[str]:
    msgs = [dict(role="system", content=SYSTEM), *history, dict(role="user", content=user_msg)]
    body = dict(model=LLM_MODEL, messages=msgs, temperature=0.3, max_tokens=600, stream=True)
    headers = {"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}",
               # OpenRouter attribution headers (ignored by llama.cpp / Ollama)
               "HTTP-Referer": "https://github.com/starkhushi/clean-air-walk", "X-Title": "Clean Air Walk"}
    with requests.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers,
                       stream=True, timeout=300) as r:
        if r.status_code != 200:
            yield f"Sorry, the local Gemma server returned HTTP {r.status_code}."
            return
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta
