"""Sky check: compare what your eyes (camera) see with what the forecast says.

Gemma 4 is multimodal. It looks at a photo of the sky or horizon and rates the
haze it sees. Python then compares that rating with the forecast band for this
hour. A photo cannot measure PM2.5 (fog, cloud, dusk light and camera exposure
all look like haze), so the result is framed as "eyes vs forecast", with the
reasons it might be wrong.
"""
from __future__ import annotations

import base64
import re

from . import llm

VISION = """You look at ONE photo that a person in India took outdoors to judge air pollution.
Output ONLY a JSON object, no prose:
{
 "is_outdoor_sky": true/false,          // does the photo show sky, horizon or a distant view?
 "haze": 0-4,                           // 0 none (crisp, blue, distant objects sharp), 1 light, 2 moderate (washed-out, greyish), 3 heavy (distant buildings fade), 4 severe (thick smog, visibility very short)
 "visibility": "short phrase, e.g. 'buildings about 1 km away are faint'",
 "confounders": ["fog" | "cloud" | "rain" | "night" | "sunset" | "indoor" | "too close" ...],  // things that make haze hard to judge
 "description_en": "one sentence describing the sky and haze",
 "description_hi": "the same sentence in Hindi (Devanagari)"
}
Be honest and conservative. If the photo is not an outdoor view, set is_outdoor_sky false."""

# Which haze ratings are plausible for each forecast band (Good .. Severe).
BAND_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]
EXPECTED_HAZE = {"Good": (0, 1), "Satisfactory": (0, 1), "Moderate": (1, 2), "Poor": (2, 3),
                 "Very poor": (3, 4), "Severe": (3, 4)}
HAZE_WORD = {"en": ["no haze", "light haze", "moderate haze", "heavy haze", "severe smog"],
             "hi": ["कोई धुंध नहीं", "हल्की धुंध", "मध्यम धुंध", "घनी धुंध", "बहुत घना स्मॉग"]}
MAX_BYTES = 2_500_000


def decode_image(data_url: str) -> tuple[str, str]:
    m = re.match(r"data:(image/(?:jpeg|png|webp));base64,(.+)", data_url or "", re.S)
    if not m:
        raise ValueError("Please send a JPEG, PNG or WebP photo.")
    mime, b64 = m.group(1), m.group(2)
    raw = base64.b64decode(b64, validate=True)
    if len(raw) > MAX_BYTES:
        raise ValueError("Photo is too large; please use a smaller one.")
    return mime, b64


def check(data_url: str, fc: dict, lang: str = "en") -> dict:
    lang = "hi" if lang == "hi" else "en"
    mime, b64 = decode_image(data_url)
    raw = llm._parse_json(llm.complete(VISION, "Rate the haze in this photo.", max_tokens=400,
                                       images=[(mime, b64)]))
    if not raw:
        return dict(ok=False, message=("Gemma couldn't look at the photo right now. Please try again in a minute."
                                       if lang == "en" else "Gemma अभी फ़ोटो नहीं देख पा रहा है। एक मिनट बाद फिर कोशिश करें।"))
    if not raw.get("is_outdoor_sky", True):
        return dict(ok=False, message=("That doesn't look like an outdoor view. Point the camera at the sky or the horizon."
                                       if lang == "en" else "यह बाहर का दृश्य नहीं लगता। कैमरा आसमान या दूर के दृश्य की ओर करें।"))
    haze = llm._int_or_none(raw.get("haze"), 0, 4)
    haze = 2 if haze is None else haze
    confounders = [str(c).lower() for c in (raw.get("confounders") or [])][:4]

    pm = fc["forecast"]["pm25"][0] if fc["forecast"]["pm25"] else None
    nr = fc.get("now_reading")
    if nr and nr["age_hours"] <= 3:
        pm, source = nr["pm25"], "sensor"
    else:
        source = "forecast"
    if pm is None:
        return dict(ok=False, message="No forecast is available for this hour yet.")
    band = next((b["label"] for b in fc["bands"] if b["max"] is None or (pm is not None and pm <= b["max"])), "Moderate")
    lo, hi = EXPECTED_HAZE.get(band, (1, 2))
    if haze > hi:
        verdict = "worse"
    elif haze < lo:
        verdict = "clearer"
    else:
        verdict = "matches"
    # Fog, cloud and dusk only matter when they could be mistaken for haze.
    unsure = haze >= 2 and any(c in " ".join(confounders) for c in ("fog", "cloud", "rain", "night", "sunset", "mist"))
    band_hi = {"Good": "अच्छा", "Satisfactory": "संतोषजनक", "Moderate": "मध्यम", "Poor": "ख़राब",
               "Very poor": "बहुत ख़राब", "Severe": "गंभीर"}.get(band, band)

    desc = raw.get("description_hi" if lang == "hi" else "description_en") or ""
    word = HAZE_WORD[lang][haze]
    src_en = "the nearest sensor" if source == "sensor" else "the forecast"
    src_hi = "पास का सेंसर" if source == "sensor" else "पूर्वानुमान"
    if lang == "en":
        head = {"worse": f"⚠️ Your photo looks worse than {src_en}.",
                "clearer": f"🌤️ Your photo looks clearer than {src_en}.",
                "matches": f"✅ Your photo matches {src_en}."}[verdict]
        body = (f"Gemma sees {word}. {desc} {src_en.capitalize()} says PM2.5 ≈ {pm:.0f} µg/m³ ({band}).")
        if verdict == "worse":
            body += " Trust your eyes: shorten the outing, go easier, or wait for a cleaner window."
        if unsure:
            body += f" Note: {', '.join(c for c in confounders if c in ('fog', 'cloud', 'rain', 'night', 'sunset', 'mist'))} can look like smog, so take this with a pinch of salt."
    else:
        head = {"worse": f"⚠️ आपकी फ़ोटो {src_hi} से ज़्यादा प्रदूषित दिखती है।",
                "clearer": f"🌤️ आपकी फ़ोटो {src_hi} से ज़्यादा साफ़ दिखती है।",
                "matches": f"✅ आपकी फ़ोटो {src_hi} से मेल खाती है।"}[verdict]
        body = f"Gemma के अनुसार फ़ोटो में {word} है। {desc} {src_hi} के अनुसार PM2.5 लगभग {pm:.0f} µg/m³ ({band_hi}) है।"
        if verdict == "worse":
            body += " अपनी आँखों पर भरोसा करें: कम समय बाहर रहें, हल्की गतिविधि करें या साफ़ समय का इंतज़ार करें।"
        if unsure:
            body += " ध्यान दें: कोहरा, बादल या शाम की रोशनी भी धुंध जैसी दिख सकती है।"
    return dict(ok=True, verdict=verdict, haze=haze, haze_word=word, band=band, pm25=pm, source=source,
                confounders=confounders, visibility=raw.get("visibility"), headline=head, message=body.strip())
