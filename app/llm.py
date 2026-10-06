"""Gemma assistant: turns the forecast into a plan for the person's actual day.

Pipeline ("Gemma understands, Python verifies"):
  1. Gemma reads the question (and earlier turns) and extracts constraints as
     JSON: which days, earliest start, latest end, duration, sensitive group.
  2. Python searches the forecast for the cleanest windows that satisfy those
     constraints exactly (scored by the worst 2-hour slot inside the window).
  3. Gemma explains the verified options in the person's own language.
Small models are good at language and bad at constraint arithmetic, so the
arithmetic never depends on the model.

Backends (LLM_BACKEND):
  gemini  - Gemma 4 served by Google AI Studio (GEMINI_API_KEY, free tier).
  openai  - any OpenAI-compatible Gemma server: OpenRouter, llama.cpp
            `llama-server`, Ollama (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY).
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator
from datetime import datetime

import requests

BACKEND = os.environ.get("LLM_BACKEND", "gemini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")   # Ollama default
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
MAX_HISTORY = 6
SENSITIVE_MAX = 60      # top of "Satisfactory"

EXTRACT = """You turn a request about going outside into JSON constraints. Output ONLY a JSON object, no prose.

Keys:
- "activity": short English description, e.g. "5 km run"
- "days": list of day labels copied exactly from the forecast table (e.g. ["Wed 07 Oct"]), or [] if any day is fine
- "earliest_hour": integer 0-23, earliest start the person can manage, or null
- "latest_end_hour": integer 1-24, when they must be finished, or null
- "duration_hours": number, how long the activity takes (default 1 for walks/runs, 3 for a cricket match)
- "sensitive": true if asthma, heart/lung problems, pregnancy, small children or elderly people are involved
- "language": "en", "hi" (Devanagari) or "hinglish" - the language and script of the LATEST message

Interpretation:
- "today", "kal"/"tomorrow", "weekend" (Sat and Sun), weekday names -> map to labels from the table, using the local date given.
- "morning"/"subah" -> earliest 6, latest_end 11. "afternoon"/"dopahar" -> earliest 12, latest_end 16.
  "evening"/"shaam" -> earliest 16, latest_end 20. "after college"/"after 4 pm" -> earliest 16.
- Carry constraints over from earlier messages unless the latest message changes them."""

EXPLAIN = """You are Clean Air Walk, a friendly assistant that helps people in India choose when to go outside, based on air quality.

You are given the person's question, the forecast, and VERIFIED OPTIONS computed from the forecast that already respect their constraints.
- Recommend option 1 as the main pick and option 2 (if any) as the backup. Never pick a slot that is not in the verified options and never invent numbers.
- Say times naturally, like a friend would: "Saturday, 4 to 6 pm" (or "शनिवार शाम 4 से 6 बजे"), then "PM2.5 around 38, Satisfactory". Do not copy the option lines verbatim.
- Use the start time given in each option ("start between 8 and 9 am"); do not compute your own.
- If the notes say no option meets the sensitive-group threshold, say so honestly and suggest a lighter activity, a well-fitted N95 mask, or the best other day that is listed.
- If the notes say no option fits their constraints at all, say so and offer the listed alternative.
- Mention heat above 33°C or rain if shown.
- Reply in the language given (en = English, hi = Hindi in Devanagari script, hinglish = Hindi in Latin script).
- Warm and brief: at most 80 words. No tables. Use **bold** for the main pick.
- Only if a health condition, child, elderly person or symptom is mentioned: add one short line to follow their doctor's advice. Otherwise no medical disclaimer.
- Only mention rain if it falls inside the recommended slots."""


# ------------------------------------------------------------------ context
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


# ------------------------------------------------------------------ step 1: constraints
def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _int_or_none(v, lo, hi):
    try:
        v = int(round(float(v)))
        return v if lo <= v <= hi else None
    except (TypeError, ValueError):
        return None


def extract_constraints(question: str, context: str, history: list[dict]) -> dict:
    convo = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in history)
    prompt = (f"{context}\n\nConversation so far:\n{convo or '(none)'}\n"
              f"LATEST MESSAGE: {question}\n\nJSON:")
    raw = _parse_json(complete(EXTRACT, prompt, max_tokens=300))
    lang = raw.get("language") if raw.get("language") in ("en", "hi", "hinglish") else None
    if lang is None:   # cheap script-based fallback
        lang = "hi" if re.search(r"[ऀ-ॿ]", question) else "en"
    dur = raw.get("duration_hours")
    try:
        dur = min(max(float(dur), 0.5), 10)
    except (TypeError, ValueError):
        dur = 1.0
    return dict(activity=str(raw.get("activity") or "going outside")[:80],
                days=[str(d) for d in (raw.get("days") or []) if isinstance(d, str)],
                earliest_hour=_int_or_none(raw.get("earliest_hour"), 0, 23),
                latest_end_hour=_int_or_none(raw.get("latest_end_hour"), 1, 24),
                duration_hours=dur, sensitive=bool(raw.get("sensitive")), language=lang)


# ------------------------------------------------------------------ step 2: verified search
def find_options(fc: dict, c: dict, k: int = 3) -> tuple[list[dict], list[str]]:
    """Cleanest windows that satisfy the constraints, scored by their worst slot."""
    notes = []
    n_slots = max(1, math.ceil(c["duration_hours"] / 2))
    wanted = {d.lower() for d in c["days"]}

    def search(respect_days: bool, respect_hours: bool) -> list[dict]:
        out = []
        for day in fc["slots"]:
            if respect_days and wanted and day["weekday"].lower() not in wanted:
                continue
            slots = day["slots"]
            for i in range(len(slots) - n_slots + 1):
                win = slots[i:i + n_slots]
                start_h, end_h = int(win[0]["start"][:2]), int(win[-1]["end"][:2])
                if any(int(b["start"][:2]) != int(a["end"][:2]) for a, b in zip(win, win[1:])):
                    continue    # not contiguous
                if respect_hours:
                    if c["earliest_hour"] is not None and start_h < c["earliest_hour"]:
                        continue
                    if c["latest_end_hour"] is not None and end_h > c["latest_end_hour"]:
                        continue
                if any(s["rain"] > 0.5 for s in win):
                    continue
                worst = max(win, key=lambda s: s["pm25"])
                out.append(dict(day=day["weekday"], start=win[0]["start"], end=win[-1]["end"],
                                pm25=worst["pm25"], band=worst["label"],
                                temp=max(s["temp"] for s in win),
                                mean=sum(s["pm25"] for s in win) / len(win)))
        out.sort(key=lambda o: (o["pm25"], o["mean"]))
        return out

    opts = search(True, True)
    if not opts:
        opts = search(True, False)
        if opts:
            notes.append("No window fits their time of day on the requested day(s); options below ignore the time limit.")
        else:
            opts = search(False, True) or search(False, False)
            notes.append("No window fits the requested day(s); options below are from other days.")
    if c["sensitive"] and opts and opts[0]["pm25"] > SENSITIVE_MAX:
        notes.append(f"Sensitive group: no option that fits is Good or Satisfactory (all above {SENSITIVE_MAX}).")
    # keep options on different days or clearly different times
    picked = []
    for o in opts:
        if all(not (o["day"] == p["day"] and o["start"] == p["start"]) for p in picked):
            picked.append(o)
        if len(picked) == k:
            break
    return picked, notes


# ------------------------------------------------------------------ step 3: explanation
def stream_answer(question: str, context: str, fc: dict, history: list[dict] | None = None) -> Iterator[str]:
    history = _clean_history(history)
    question = question.strip()[:800]
    c = extract_constraints(question, context, history)
    opts, notes = find_options(fc, c)
    if opts:
        lines = []
        for i, o in enumerate(opts):
            s, e = int(o["start"][:2]), int(o["end"][:2])
            latest = e - c["duration_hours"]
            start_txt = (f"start at {s:02d}:00" if latest <= s
                         else f"start between {s:02d}:00 and {int(latest):02d}:{int(round((latest % 1) * 60)):02d}")
            lines.append(f"{i + 1}) {o['day']}, window {o['start']}-{o['end']} ({start_txt} for a "
                         f"{c['duration_hours']:g}-hour activity): PM2.5 {o['pm25']:.0f} ({o['band']}), "
                         f"up to {o['temp']:.0f}°C")
    else:
        lines = ["(no forecast slots available)"]
    verified = ("VERIFIED OPTIONS (best first):\n" + "\n".join(lines)
                + ("\nNOTES: " + " ".join(notes) if notes else "")
                + f"\nConstraints understood: {json.dumps(c, ensure_ascii=False)}"
                + f"\nAnswer language: {c['language']}")
    print("constraints", json.dumps(c, ensure_ascii=False), "| options", lines[:2], flush=True)
    convo = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in history)
    prompt = (f"{context}\n\n{verified}\n\nConversation so far:\n{convo or '(none)'}\n"
              f"Question: {question}")
    yield from stream(EXPLAIN, prompt)


def _clean_history(history: list[dict] | None) -> list[dict]:
    out = []
    for m in (history or [])[-MAX_HISTORY:]:
        role, text = m.get("role"), str(m.get("content", ""))[:1500]
        if role in ("user", "assistant") and text.strip():
            out.append(dict(role=role, content=text))
    return out


# ------------------------------------------------------------------ backends
def complete(system: str, prompt: str, max_tokens: int = 300) -> str:
    return "".join(stream(system, prompt, max_tokens=max_tokens, temperature=0.0))


def stream(system: str, prompt: str, max_tokens: int = 600, temperature: float = 0.3) -> Iterator[str]:
    if BACKEND == "openai":
        yield from _stream_openai(system, prompt, max_tokens, temperature)
    else:
        yield from _stream_gemini(system, prompt, max_tokens, temperature)


def _sse_lines(r: requests.Response) -> Iterator[str]:
    r.encoding = "utf-8"          # SSE has no charset header; requests would assume latin-1
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            yield line[5:].strip()


def _stream_gemini(system: str, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        yield "The assistant is not configured yet: set GEMINI_API_KEY on the server."
        return
    body = dict(contents=[dict(role="user", parts=[dict(text=prompt)])],
                systemInstruction=dict(parts=[dict(text=system)]),
                generationConfig=dict(temperature=temperature, maxOutputTokens=max_tokens,
                                      thinkingConfig=dict(thinkingLevel="minimal")))
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
           f":streamGenerateContent?alt=sse")
    for attempt in range(2):     # one retry on transient 5xx
        with requests.post(url, json=body, headers={"x-goog-api-key": key}, stream=True, timeout=90) as r:
            if r.status_code >= 500 and attempt == 0:
                continue
            if r.status_code != 200:
                print("gemini error", r.status_code, r.text[:500], flush=True)
                yield f"Sorry, Gemma is unavailable right now (HTTP {r.status_code}). Please try again in a minute."
                return
            for payload in _sse_lines(r):
                try:
                    js = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for cand in js.get("candidates", []):
                    for part in (cand.get("content") or {}).get("parts", []):
                        if part.get("text") and not part.get("thought"):
                            yield part["text"]
            return


def _stream_openai(system: str, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
    msgs = [dict(role="system", content=system), dict(role="user", content=prompt)]
    body = dict(model=LLM_MODEL, messages=msgs, temperature=temperature, max_tokens=max_tokens, stream=True)
    headers = {"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}",
               # OpenRouter attribution headers (ignored by llama.cpp / Ollama)
               "HTTP-Referer": "https://github.com/starkhushi/clean-air-walk", "X-Title": "Clean Air Walk"}
    with requests.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers,
                       stream=True, timeout=300) as r:
        if r.status_code != 200:
            print("llm error", r.status_code, r.text[:500], flush=True)
            yield f"Sorry, the Gemma server returned HTTP {r.status_code}."
            return
        for payload in _sse_lines(r):
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
            if delta:
                yield delta
