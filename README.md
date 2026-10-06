# Clean Air Walk 🌿

**When is the air clean enough to go outside this week?**

Clean Air Walk finds the cleanest 2-hour daylight window for each of the next five days,
for any place with a nearby air-quality sensor. It was built for Delhi NCR, where October
smog decides whether an evening walk, run or cricket match is a good idea.

Built for the [Hacktoberfest 2026 DEV challenge](https://dev.to/challenges/hacktoberfest-week1-2026-10-05),
week 1: **Touch Grass**.

## How it works

1. **Global forecast.** The Copernicus CAMS model forecasts PM2.5 on a ~45 km grid
   (fetched from [Open-Meteo](https://open-meteo.com), no key needed).
2. **Your street.** The nearest real ground sensor on [OpenAQ](https://openaq.org)
   gives the last few weeks of measured PM2.5.
3. **Open-weight AI.** [TabPFN v2](https://github.com/PriorLabs/TabPFN), a tabular
   foundation model, learns how the real sensor relates to CAMS and local weather
   (wind, humidity, rain, temperature, mixing height, hour of day). It runs on CPU in
   seconds, needs no training loop, and returns uncertainty bands.
4. **Walk windows.** The corrected forecast is scanned for the cleanest dry 2-hour window
   between 06:00 and 20:00 each day, banded with India's National AQI breakpoints.
5. **Honest check.** Every forecast re-runs a backtest on the latest 72 observed hours and
   shows the error of TabPFN against raw CAMS and against a simple bias fix.

If no sensor is within 25 km, the app falls back to the raw CAMS forecast and says so.

## Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export OPENAQ_API_KEY=...        # free at https://explore.openaq.org
uvicorn app.main:app --reload
```

Open http://localhost:8000.

## Deploy on Render

The repo includes a `render.yaml` blueprint. In Render choose **New → Blueprint**, pick this
repo, and paste your `OPENAQ_API_KEY` when asked. The build downloads the TabPFN v2 weights
once so the first visitor does not wait.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAQ_API_KEY` | none | Enables the street-level TabPFN correction |
| `TABPFN_N_ESTIMATORS` | 4 | Ensemble size; higher is slower and slightly better |
| `MAX_TRAIN_ROWS` | 600 | Most recent sensor hours used as context |
| `TABPFN_MODEL_CACHE_DIR` | TabPFN default | Where model weights are cached |

## Credits

- Built with PriorLabs-TabPFN (v2 weights, Prior Labs License: Apache 2.0 with attribution).
- Air-quality forecast: Copernicus Atmosphere Monitoring Service via Open-Meteo.
- Ground measurements: OpenAQ and the agencies that publish to it, including CPCB.

Not medical advice. If you have asthma or heart conditions, follow your doctor's guidance.

## License

MIT
