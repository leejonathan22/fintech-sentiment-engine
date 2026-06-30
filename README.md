# Fintech Sentiment Engine

Pulls public App Store sentiment on BNPL and neobank apps threatening Chase's cobrand card interchange, clusters reviews into product themes with BERTopic, and generates a PM-ready narrative brief via Claude.

**Target apps:** Chime, Cash App, Affirm, Klarna vs. Chase (cobrand baseline)

## Project Phases

| Phase | Module | Status |
|-------|--------|--------|
| 1 — Data collection | `src/data_collection.py` | Active |
| 2 — Topic clustering | `src/clustering.py` | Stub |
| 3 — Brief generation | `src/brief_generator.py` | Stub |
| 4 — Streamlit UI | `app/streamlit_app.py` | Stub |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY
```

## Phase 1 — Validate data collection

Open `notebooks/01_explore.ipynb` and run cells top to bottom. The notebook pulls 50 Chime reviews as a smoke test, checks the schema, then you can uncomment `fetch_all()` to pull 500 reviews per app.

Raw CSVs land in `data/raw/<app_key>.csv` (gitignored).

## Structure

```
fintech-sentiment-engine/
├── notebooks/01_explore.ipynb   # data pull + BERTopic tuning scratch
├── src/
│   ├── data_collection.py       # App Store scraper + APP_REGISTRY
│   ├── clustering.py            # BERTopic pipeline (Phase 2)
│   └── brief_generator.py       # Claude API brief (Phase 3)
├── app/streamlit_app.py         # thin UI (Phase 4)
├── data/                        # gitignored
├── requirements.txt
└── .env.example
```
