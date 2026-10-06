"""Exercise and mood guidance by air-quality band.

Sources (paraphrased, not invented):
- CPCB National AQI health statements for each band (India).
- US EPA AirNow "Air Quality Guide for Activity": when to reduce prolonged or heavy
  outdoor exertion, for sensitive groups and for everyone.
- Exercise as a mood support and Tele-MANAS (14416), India's free government
  mental-health helpline.
This is general guidance, not medical advice.
"""
from __future__ import annotations

from collections.abc import Iterator

from . import llm

BANDS = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"]

# CPCB health statement per band (paraphrased).
CPCB = {
    "en": {"Good": "Minimal health impact.",
           "Satisfactory": "Minor breathing discomfort for sensitive people.",
           "Moderate": "Breathing discomfort for people with asthma or lung disease, and for heart patients, children and older adults.",
           "Poor": "Breathing discomfort for most people on prolonged exposure.",
           "Very poor": "Respiratory illness on prolonged exposure.",
           "Severe": "Affects healthy people and seriously affects people with existing disease."},
    "hi": {"Good": "स्वास्थ्य पर न्यूनतम असर।",
           "Satisfactory": "संवेदनशील लोगों को साँस लेने में हल्की तकलीफ़।",
           "Moderate": "अस्थमा या फेफड़ों की बीमारी वाले लोगों, दिल के मरीज़ों, बच्चों और बुज़ुर्गों को साँस लेने में तकलीफ़।",
           "Poor": "लंबे समय तक बाहर रहने पर ज़्यादातर लोगों को साँस लेने में तकलीफ़।",
           "Very poor": "लंबे समय तक बाहर रहने पर साँस की बीमारी।",
           "Severe": "स्वस्थ लोगों पर भी असर, और बीमार लोगों पर गंभीर असर।"},
}

# Activity guidance, following the EPA activity guide's logic, mapped onto India's bands.
# level: what kind of outdoor exercise is reasonable. 3 = anything, 2 = moderate, 1 = light & short, 0 = indoors.
ACTIVITY = {
    #               general  sensitive
    "Good":         (3, 3),
    "Satisfactory": (3, 2),
    "Moderate":     (2, 1),
    "Poor":         (1, 0),
    "Very poor":    (0, 0),
    "Severe":       (0, 0),
}
ADVICE = {
    "en": {3: "Go for it: any outdoor exercise, including runs and sports.",
           2: "Outdoor exercise is fine, but swap long, hard sessions for moderate ones (brisk walk, easy cycling) and take breaks.",
           1: "Keep outdoor activity light and short (an easy walk). Do your real workout indoors.",
           0: "Exercise indoors today. Skip outdoor workouts and keep windows closed during the worst hours."},
    "hi": {3: "बेझिझक जाएँ: दौड़ और खेल समेत कोई भी बाहरी व्यायाम ठीक है।",
           2: "बाहर व्यायाम ठीक है, लेकिन लंबे और भारी सेशन की जगह मध्यम व्यायाम करें (तेज़ चलना, आराम से साइकिल) और बीच में आराम लें।",
           1: "बाहर सिर्फ़ हल्की और छोटी गतिविधि करें (आराम से टहलना)। असली व्यायाम घर के अंदर करें।",
           0: "आज घर के अंदर व्यायाम करें। बाहर वर्कआउट न करें और सबसे ख़राब घंटों में खिड़कियाँ बंद रखें।"},
}

# Indoor alternatives: (emoji, English, Hindi, minimum level of exertion 1 light .. 3 vigorous)
INDOOR = [
    ("🧘", "Yoga and stretching, 15–20 min", "योग और स्ट्रेचिंग, 15–20 मिनट", 1),
    ("🌬️", "Slow breathing: in for 4, out for 6, for 5 min", "धीमी साँस: 4 गिनती तक अंदर, 6 तक बाहर, 5 मिनट", 1),
    ("🚶", "Indoor walking or marching in place, 15 min", "घर में टहलना या एक जगह मार्च करना, 15 मिनट", 1),
    ("💃", "Dance to 4–5 songs", "4–5 गानों पर डांस", 2),
    ("🏋️", "Bodyweight circuit: squats, wall push-ups, lunges, plank (3 rounds)", "बॉडीवेट सर्किट: स्क्वाट, दीवार पुश-अप, लंज, प्लैंक (3 राउंड)", 2),
    ("🪜", "Stair climbs in your building, 10 min (if the stairwell isn't dusty)", "अपनी बिल्डिंग की सीढ़ियाँ, 10 मिनट (अगर सीढ़ियों में धूल न हो)", 3),
]

MOOD = {
    "en": ["Keep moving indoors: even 15 minutes of exercise lifts mood.",
           "Get daylight: sit by a window in the morning, even when it's hazy.",
           "Plan your next clean window. Having a walk to look forward to helps.",
           "Stay in touch: call a friend or invite family for indoor yoga.",
           "Don't doom-scroll pollution news. Check the forecast once, then plan.",
           "Sleep well: keep the bedroom air cleaner with windows shut at night on bad days."],
    "hi": ["घर में भी चलते-फिरते रहें: सिर्फ़ 15 मिनट का व्यायाम भी मूड बेहतर करता है।",
           "रोशनी लें: सुबह खिड़की के पास बैठें, धुंध हो तब भी।",
           "अगले साफ़ समय की योजना बनाएँ। आगे किसी सैर का इंतज़ार मन को अच्छा रखता है।",
           "जुड़े रहें: किसी दोस्त को फ़ोन करें या परिवार के साथ घर में योग करें।",
           "प्रदूषण की ख़बरें बार-बार न देखें। पूर्वानुमान एक बार देखें, फिर योजना बनाएँ।",
           "अच्छी नींद लें: ख़राब दिनों में रात को खिड़कियाँ बंद रखें।"],
}
HELPLINE = {
    "en": "If low mood, anxiety or stress lasts more than two weeks, talk to someone. Tele-MANAS, India's free government mental-health helpline, is available 24×7 on 14416 or 1-800-891-4416.",
    "hi": "अगर उदासी, घबराहट या तनाव दो हफ़्ते से ज़्यादा रहे, तो किसी से बात करें। Tele-MANAS, भारत सरकार की मुफ़्त मानसिक स्वास्थ्य हेल्पलाइन, 24×7 उपलब्ध है: 14416 या 1-800-891-4416।",
}


def band_of(pm: float, bands: list[dict]) -> str:
    return next((b["label"] for b in bands if b["max"] is None or pm <= b["max"]), "Severe")


def plan(fc: dict, profile: dict | None, lang: str = "en") -> dict:
    lang = "hi" if lang == "hi" else "en"
    p = llm.clean_profile(profile)
    sensitive = bool(p["conditions"])
    nr = fc.get("now_reading")
    if nr and nr["age_hours"] <= 3:
        pm_now, source = nr["pm25"], "sensor"
    else:
        pm_now, source = (fc["forecast"]["pm25"][0] if fc["forecast"]["pm25"] else None), "forecast"
    if pm_now is None:
        return dict(ok=False)
    band_now = band_of(pm_now, fc["bands"])
    level = ACTIVITY[band_now][1 if sensitive else 0]

    # Best upcoming windows (today, then tomorrow) and their level, so the card can say
    # "indoors now, but 17:00 today is fine for a brisk walk".
    upcoming = []
    for w in fc["windows"][:2]:
        b = w["best"]["label"]
        upcoming.append(dict(day=w["weekday"], start=w["best"]["start"], end=w["best"]["end"], pm25=w["best"]["pm25"],
                             band=b, level=ACTIVITY[b][1 if sensitive else 0],
                             advice=ADVICE[lang][ACTIVITY[b][1 if sensitive else 0]]))
    # Indoor air is usually cleaner, so anyone can work out indoors; sensitive groups stay light-to-moderate.
    cap = 2 if sensitive else 3
    indoor = [dict(icon=e, text=(hi if lang == "hi" else en), level=lv) for e, en, hi, lv in INDOOR if lv <= cap]
    return dict(ok=True, sensitive=sensitive, pm25=round(pm_now, 1), band=band_now, source=source,
                health=CPCB[lang][band_now], level=level, advice=ADVICE[lang][level],
                upcoming=upcoming, indoor=indoor, mood=MOOD[lang], helpline=HELPLINE[lang])


INDOOR_SYSTEM = """You are a friendly fitness buddy inside Clean Air Walk. The air outside is not great, so you design ONE short indoor routine.
Rules:
- 15 to 20 minutes total, with a 2-minute warm-up and a 2-minute cool-down. List each block with minutes.
- Use only bodyweight moves, yoga, stretching, marching, dancing or stairs. No equipment needed.
- If the person has asthma, a heart or lung condition, is pregnant, or is with an elderly person or children, keep it gentle and low-impact, avoid breath-holding, and say they can stop any time.
- End with one line on how moving helps mood on smoggy days.
- Reply in the requested language (en = English, hi = Hindi in Devanagari, hinglish = Hindi in Latin script). Max 130 words. Use short bullet lines. No medical claims."""


def stream_indoor_plan(fc: dict, profile: dict | None, lang: str = "en") -> Iterator[str]:
    info = plan(fc, profile, lang)
    p = llm.clean_profile(profile)
    who = llm.profile_summary(p) or "no special health notes"
    prompt = (f"Air now: PM2.5 about {info.get('pm25')} µg/m³ ({info.get('band')}). "
              f"Person: {who}. Language: {lang}.\nDesign today's indoor routine.")
    try:
        yield from llm.stream(INDOOR_SYSTEM, prompt, max_tokens=500)
    except llm.LLMError:
        yield ("\n".join(f"{i['icon']} {i['text']}" for i in info.get("indoor", []))
               + ("\n\n(Gemma is busy right now, so here are the standard indoor options.)" if lang != "hi"
                  else "\n\n(Gemma अभी व्यस्त है, इसलिए ये सामान्य घरेलू विकल्प हैं।)"))
