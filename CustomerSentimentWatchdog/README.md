# Customer Sentiment Watchdog (Advanced)

A polished Streamlit dashboard to analyze customer tickets with sentiment **and** emotions, keyword extraction, topic clustering, interactive charts, alerts, exports, and more.

## Features
- Upload CSV or paste tickets (multi-line)
- Multilingual sentiment (Transformers) with VADER fallback
- Emotion detection (anger, joy, sadness, fear, surprise, disgust) with fallback
- Keyword & keyphrase extraction (YAKE + TF-IDF)
- Topic clustering (scikit-learn KMeans)
- Trend over time (line + heatmap by day/hour)
- Customizable alert thresholds + real-time alert panel (email/Slack hooks ready)
- Summary cards and Customer Happiness Score (0-100)
- Drill-down: click charts to filter table
- Data filters: date range, sentiment, emotion, keyword, region, plan
- Export to CSV/Excel/PDF
- Light/Dark mode compatible
- Historical report archive
- Model selection: Fast (VADER) or Deep (Transformers)
- Segmentation by Region/Plan
- Smart recommendations for highly negative tickets
- Accessibility: text-to-speech for summaries

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV Format
Minimum column: `ticket` (text). Optional: `timestamp` (YYYY-MM-DD HH:MM), `region`, `plan`.

Sample provided: `sample_tickets.csv`

## Notes
- On first run, transformer models will download. If offline, choose **Fast (VADER)** in sidebar.
- PDF export uses ReportLab to generate a simple one-page summary PDF.
