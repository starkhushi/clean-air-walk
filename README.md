# Clean Air Walk 🌿

**Tell Gemma what you want to do outside. It finds the cleanest time this week.**

October smog in Delhi NCR decides whether an evening run, a walk with grandparents or a
weekend cricket match is a good idea. Clean Air Walk reads the 5-day air-quality forecast
for your exact location, checks it against your nearest real sensor, and lets you ask in
plain English, Hindi or Hinglish:

> *"5 km run after college, free after 4 pm. I have mild asthma."*
> *"मुझे दादाजी के साथ सुबह पार्क में टहलना है। कौन सा दिन सबसे अच्छा है?"*

Gemma 4 answers with one specific day and time, the expected PM2.5, a backup, and a
short health note.

Built for the [Hacktoberfest 2026 DEV challenge](https://dev.to/challenges/hacktoberfest-week1-2026-10-05),
week 1: **Touch Grass**.

## How it works

1. **Forecast.** Hourly PM2.5 from the Copernicus CAMS global model plus local weather,
   both via [Open-Meteo](https://open-meteo.com) (no key needed).
2. **Reality check.** The nearest live PM2.5 sensor on [OpenAQ](https://openaq.org), often a
   government CPCB station. The page shows how far the forecast was from the sensor over the
   latest 72 hours of readings.
3. **Walk windows.** The cleanest dry 2-hour daylight window per day, banded with India's
   National AQI breakpoints.
4. **Gemma.** The forecast is turned into a compact table of 2-hour slots and given to
   **Gemma 4** (open weights, Apache 2.0) with strict rules: use only the table, respect the
   person's constraints, prefer Good/Satisfactory slots for sensitive groups, and reply in the
   person's language. Answers stream into the page.

### What I tried first, and why it isn't in the app

I first tried to make the forecast *better* than CAMS with TabPFN v2, using real CPCB sensors
in Ghaziabad, Delhi (Anand Vihar) and Noida (Sector 62), with 6 rolling 72-hour backtests per
station (Sep 18 to Oct 2, 2026):

| Method | Mean absolute error (µg/m³) |
|---|---|
| Raw CAMS | **19.9** |
| CAMS × sensor bias ratio | 20.7 |
| TabPFN, predicting PM2.5 | 23.1 |
| TabPFN, predicting the sensor/CAMS ratio | 22.3 |
| TabPFN + latest sensor reading (nowcast) | 22.4 |

CAMS won, so the app uses CAMS and is honest about its error instead of adding a model that
makes it worse.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...        # free at https://aistudio.google.com/apikey
export OPENAQ_API_KEY=...        # free at https://explore.openaq.org
uvicorn app.main:app --reload
```

### Run Gemma yourself instead

Gemma's weights are open, so the same app works with a model on your own machine:

```bash
ollama pull gemma3:4b             # or any Gemma GGUF with llama.cpp's llama-server
export LLM_BACKEND=openai
export LLM_BASE_URL=http://127.0.0.1:11434/v1
export LLM_MODEL=gemma3:4b
uvicorn app.main:app
```

## Deploy on Render

`render.yaml` is a blueprint. In Render choose **New → Blueprint**, pick this repo, and paste
`GEMINI_API_KEY` and `OPENAQ_API_KEY` when asked. The Starter plan is enough.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | none | Google AI Studio key for Gemma 4 |
| `GEMINI_MODEL` | `gemma-4-26b-a4b-it` | Or `gemma-4-31b-it` |
| `LLM_BACKEND` | `gemini` | `openai` for llama.cpp / Ollama |
| `LLM_BASE_URL`, `LLM_MODEL` | Ollama defaults | Self-hosted Gemma endpoint |
| `OPENAQ_API_KEY` | none | Enables the sensor reality check |

## Credits

- Gemma 4 by Google DeepMind (Apache 2.0).
- Air-quality forecast: Copernicus Atmosphere Monitoring Service via Open-Meteo.
- Ground measurements: OpenAQ and the agencies that publish to it, including CPCB.

Not medical advice. If you have asthma or heart conditions, follow your doctor's guidance.

## License

MIT
