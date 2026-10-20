# SentimentAutomated

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) ![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white) ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

A live sentiment-monitoring pipeline: ingest, score, store, alert. Live
Hacker News ingestion (via Algolia's public search API, no credentials
needed) feeds a quality gate that dedupes and filters text, batched
RoBERTa sentiment scoring, keyword and spaCy-based topic tagging, an
append-only SQLite store, and threshold-based alerting. Reddit and
NewsAPI ingestion are also supported. The dashboard shows sentiment by
topic, a sentiment-over-time heatmap, a topic-correlation view, what the
quality gate caught and why, and a PDF export - all from real, live data.

## Screenshots

![Home tab](screenshots/home.png)
![Try It tab, Hacker News source](screenshots/try_it_hn.png)
![Dashboard tab](screenshots/dashboard.png)

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
streamlit run app.py
```

Live Hacker News ingestion needs no credentials:

```bash
python3 hn_ingest.py "<search query>"
python3 scheduler.py "<search query>" <topic> [interval_seconds]
```

Tests: `pytest tests/`
