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
import time
from collections.abc import Iterator
from datetime import datetime

import requests

BACKEND = os.environ.get("LLM_BACKEND", "gemini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-26b-a4b-it")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemma-4-31b-it")
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


_HINGLISH = re.compile(r"\b(kal|aaj|shaam|subah|jaana|hai|kya|mujhe|chahiye|dopahar|raat|baje|ghante|kab)\b", re.I)
_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_DAY_RE = {"mon": r"\b(mon|monday|somvar|somvaar)\b", "tue": r"\b(tue|tues|tuesday|mangalvar|mangalvaar)\b",
           "wed": r"\b(wed|wednesday|budhvar|budhvaar)\b", "thu": r"\b(thu|thur|thurs|thursday|guruvar|guruvaar)\b",
           "fri": r"\b(fri|friday|shukravar|shukravaar)\b", "sat": r"\b(sat|saturday|shanivar|shanivaar)\b",
           "sun": r"\b(sun|sunday|ravivar|ravivaar|itvaar)\b"}
_HI_DAYS = {"सोम": "mon", "मंगल": "tue", "बुध": "wed", "गुरु": "thu", "शुक्र": "fri", "शनि": "sat", "रवि": "sun"}


def heuristic_constraints(question: str, fc: dict, history: list[dict]) -> dict:
    """Keyword fallback when Gemma's extraction call fails (rate limits, outages)."""
    text = " ".join([m["content"] for m in history if m["role"] == "user"] + [question]).lower()
    q = question.lower()
    labels = [d["weekday"] for d in fc["slots"]]
    today = datetime.fromisoformat(fc["generated_local"]).date()
    days = []
    for lab in labels:
        d = datetime.strptime(f"{lab} {today.year}", "%a %d %b %Y").date()
        wd = _WEEKDAYS[d.weekday()]
        if (re.search(_DAY_RE[wd], q) or any(h in question and v == wd for h, v in _HI_DAYS.items())
                or (("today" in q or "aaj" in q or "आज" in question) and d == today)
                or (re.search(r"\b(tomorrow|kal)\b", q) or "कल" in question) and (d - today).days == 1
                or (("weekend" in q or "वीकेंड" in question) and wd in ("sat", "sun"))):
            days.append(lab)
    earliest = latest = None
    if re.search(r"morning|subah|सुबह", text):
        earliest, latest = 6, 11
    if re.search(r"afternoon|dopahar|दोपहर", text):
        earliest, latest = 12, 16
    if re.search(r"evening|shaam|शाम|after college", text):
        earliest, latest = 16, 20
    m = re.search(r"after (\d{1,2})\s*(pm|baje)?", text)
    if m:
        h = int(m.group(1))
        earliest = h + 12 if h < 8 else h
    dur = 1.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|ghante|घंटे)", text)
    if m:
        dur = float(m.group(1))
    elif re.search(r"cricket|football|match", text):
        dur = 3.0
    elif re.search(r"(\d+)\s*(min|minute)", text):
        dur = 0.5
    sensitive = bool(re.search(r"asthma|दमा|heart|lung|pregnan|child|kid|baby|year old|elderly|grand|dada|nana|दादा|नाना|बुजुर्ग|बच्च", text))
    lang = "hi" if re.search(r"[ऀ-ॿ]", question) else ("hinglish" if _HINGLISH.search(question) else "en")
    return dict(activity="going outside", days=days, earliest_hour=earliest, latest_end_hour=latest,
                duration_hours=dur, sensitive=sensitive, language=lang)


def extract_constraints(question: str, context: str, history: list[dict], fc: dict) -> dict:
    convo = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in history)
    prompt = (f"{context}\n\nConversation so far:\n{convo or '(none)'}\n"
              f"LATEST MESSAGE: {question}\n\nJSON:")
    raw = _parse_json(complete(EXTRACT, prompt, max_tokens=300))
    if not raw:
        c = heuristic_constraints(question, fc, history)
        c["source"] = "keywords"
        return c
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
                duration_hours=dur, sensitive=bool(raw.get("sensitive")), language=lang, source="gemma")


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
_answer_cache: dict[tuple, tuple[float, str]] = {}
ANSWER_TTL_S = 30 * 60


def _option_lines(opts: list[dict], c: dict) -> list[str]:
    lines = []
    for i, o in enumerate(opts):
        s, e = int(o["start"][:2]), int(o["end"][:2])
        latest = e - c["duration_hours"]
        o["start_txt"] = (f"start at {s:02d}:00" if latest <= s
                          else f"start between {s:02d}:00 and {int(latest):02d}:{int(round((latest % 1) * 60)):02d}")
        lines.append(f"{i + 1}) {o['day']}, window {o['start']}-{o['end']} ({o['start_txt']} for a "
                     f"{c['duration_hours']:g}-hour activity): PM2.5 {o['pm25']:.0f} ({o['band']}), "
                     f"up to {o['temp']:.0f}°C")
    return lines


def template_answer(opts: list[dict], notes: list[str], c: dict) -> str:
    """Plain answer from the verified options, used only if Gemma is unreachable."""
    if not opts:
        return "Sorry, I couldn't find a forecast slot for that. Try another day or place."
    a, b = opts[0], (opts[1] if len(opts) > 1 else None)
    warn = c["sensitive"] and a["pm25"] > SENSITIVE_MAX
    if c["language"] == "hi":
        txt = f"सबसे साफ़ समय: **{a['day']}, {a['start']}–{a['end']}**, PM2.5 लगभग {a['pm25']:.0f} ({a['band']})।"
        if b:
            txt += f" दूसरा विकल्प: {b['day']}, {b['start']}–{b['end']} (लगभग {b['pm25']:.0f})।"
        if warn:
            txt += " संवेदनशील लोगों के लिए यह समय भी बहुत अच्छा नहीं है; हल्की गतिविधि या N95 मास्क पर विचार करें।"
    elif c["language"] == "hinglish":
        txt = f"Sabse saaf time: **{a['day']}, {a['start']}–{a['end']}**, PM2.5 lagbhag {a['pm25']:.0f} ({a['band']})."
        if b:
            txt += f" Backup: {b['day']}, {b['start']}–{b['end']} (lagbhag {b['pm25']:.0f})."
        if warn:
            txt += " Sensitive logon ke liye yeh bhi ideal nahi hai; halki activity ya N95 mask rakhein."
    else:
        txt = f"Cleanest option: **{a['day']}, {a['start']}–{a['end']}**, PM2.5 around {a['pm25']:.0f} ({a['band']})."
        if b:
            txt += f" Backup: {b['day']}, {b['start']}–{b['end']} (around {b['pm25']:.0f})."
        if warn:
            txt += " Even this isn't Good or Satisfactory for sensitive groups; consider a lighter activity or an N95 mask."
    return txt + "\n\n(Gemma is busy right now, so this is a plain summary of the verified options.)"


def stream_answer(question: str, context: str, fc: dict, history: list[dict] | None = None) -> Iterator[str]:
    history = _clean_history(history)
    question = question.strip()[:800]
    key = (question.lower(), fc["lat"], fc["lon"], fc["generated_local"]) if not history else None
    if key:
        hit = _answer_cache.get(key)
        if hit and time.time() - hit[0] < ANSWER_TTL_S:
            yield hit[1]
            return

    c = extract_constraints(question, context, history, fc)
    opts, notes = find_options(fc, c)
    lines = _option_lines(opts, c) if opts else ["(no forecast slots available)"]
    verified = ("VERIFIED OPTIONS (best first):\n" + "\n".join(lines)
                + ("\nNOTES: " + " ".join(notes) if notes else "")
                + f"\nConstraints understood: {json.dumps(c, ensure_ascii=False)}"
                + f"\nAnswer language: {c['language']}")
    print("constraints", json.dumps(c, ensure_ascii=False), "| options", lines[:2], flush=True)
    convo = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in history)
    prompt = (f"{context}\n\n{verified}\n\nConversation so far:\n{convo or '(none)'}\n"
              f"Question: {question}")
    parts: list[str] = []
    try:
        for chunk in stream(EXPLAIN, prompt):
            parts.append(chunk)
            yield chunk
    except LLMError as e:
        print("explain failed:", e, flush=True)
        if not parts:
            yield template_answer(opts, notes, c)
        return
    if key and parts:
        if len(_answer_cache) > 500:
            _answer_cache.clear()
        _answer_cache[key] = (time.time(), "".join(parts))


def _clean_history(history: list[dict] | None) -> list[dict]:
    out = []
    for m in (history or [])[-MAX_HISTORY:]:
        role, text = m.get("role"), str(m.get("content", ""))[:1500]
        if role in ("user", "assistant") and text.strip():
            out.append(dict(role=role, content=text))
    return out


# ------------------------------------------------------------------ backends
class LLMError(RuntimeError):
    pass


RETRY_STATUS = {429, 500, 502, 503, 504}


def complete(system: str, prompt: str, max_tokens: int = 300) -> str:
    try:
        return "".join(stream(system, prompt, max_tokens=max_tokens, temperature=0.0))
    except LLMError as e:
        print("complete failed:", e, flush=True)
        return ""


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
        raise LLMError("GEMINI_API_KEY not set")
    body = dict(contents=[dict(role="user", parts=[dict(text=prompt)])],
                systemInstruction=dict(parts=[dict(text=system)]),
                generationConfig=dict(temperature=temperature, maxOutputTokens=max_tokens,
                                      thinkingConfig=dict(thinkingLevel="minimal")))
    # Free-tier Gemma endpoints throw sporadic 500/429s: retry with backoff, then try the
    # other Gemma 4 model (separate quota) before giving up.
    plan = [(GEMINI_MODEL, 0), (GEMINI_MODEL, 1.5), (GEMINI_FALLBACK_MODEL, 0), (GEMINI_FALLBACK_MODEL, 3)]
    last = None
    for model, wait in plan:
        if not model:
            continue
        time.sleep(wait)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        try:
            with requests.post(url, json=body, headers={"x-goog-api-key": key}, stream=True, timeout=60) as r:
                if r.status_code in RETRY_STATUS:
                    last = f"{model} HTTP {r.status_code}"
                    continue
                if r.status_code != 200:
                    raise LLMError(f"{model} HTTP {r.status_code}: {r.text[:300]}")
                got = False
                for payload in _sse_lines(r):
                    try:
                        js = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for cand in js.get("candidates", []):
                        for part in (cand.get("content") or {}).get("parts", []):
                            if part.get("text") and not part.get("thought"):
                                got = True
                                yield part["text"]
                if got:
                    return
                last = f"{model} returned no text"
        except requests.RequestException as e:
            last = f"{model} {type(e).__name__}"
    raise LLMError(last or "no Gemma model available")


def _stream_openai(system: str, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
    msgs = [dict(role="system", content=system), dict(role="user", content=prompt)]
    body = dict(model=LLM_MODEL, messages=msgs, temperature=temperature, max_tokens=max_tokens, stream=True)
    headers = {"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}",
               # OpenRouter attribution headers (ignored by llama.cpp / Ollama)
               "HTTP-Referer": "https://github.com/starkhushi/clean-air-walk", "X-Title": "Clean Air Walk"}
    for wait in (0, 2):
        time.sleep(wait)
        try:
            with requests.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", json=body, headers=headers,
                               stream=True, timeout=300) as r:
                if r.status_code in RETRY_STATUS:
                    continue
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                got = False
                for payload in _sse_lines(r):
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    if delta:
                        got = True
                        yield delta
                if got:
                    return
        except requests.RequestException:
            continue
    raise LLMError("Gemma server unavailable")
