import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from io import BytesIO
import logging

# Optional: enable internal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("watchdog")

# Local modules (should exist in src/)
from src.sentiment_analyzer import SentimentAnalyzer
from src.trend_tracker import TrendTracker
from src.keyword_extractor import KeywordExtractor
from src.clustering import cluster_texts
from src.utils import ensure_datetime, compute_happiness_score
from src.alerts import send_email_alert, send_slack_webhook
from src.pdf import export_summary_pdf

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Customer Sentiment Watchdog", layout="wide", page_icon="🚨")

# ---------------- Sidebar Controls ----------------
st.sidebar.title("⚙️ Settings")
neg_threshold = st.sidebar.slider("Alert threshold (negative %)", 10, 90, 40, step=5)
k_clusters = st.sidebar.slider("Topic Clusters (k)", 2, 8, 5)
date_from = st.sidebar.date_input("From date", value=None)
date_to = st.sidebar.date_input("To date", value=None)

st.sidebar.markdown("---")
st.sidebar.write("**Integrations (optional)**")
email_to = st.sidebar.text_input("Email to alert")
smtp_server = st.sidebar.text_input("SMTP server")
smtp_port = st.sidebar.number_input("SMTP port", value=465, step=1)
smtp_user = st.sidebar.text_input("SMTP username")
smtp_pass = st.sidebar.text_input("SMTP password", type="password")
slack_webhook = st.sidebar.text_input("Slack Incoming Webhook URL")

# Debug toggle
show_debug = st.sidebar.checkbox("Show debug tables", value=False)

# Title
st.title("🚨 Customer Sentiment Watchdog")
st.caption("Real-time sentiment & emotion insights for support teams")

# ---------------- Data Input ----------------
with st.expander("📥 Upload or Paste Tickets", expanded=True):
    uploaded = st.file_uploader(
        "Upload CSV (must include a 'ticket' column). Optional: 'timestamp','region','plan'.", 
        type=["csv"]
    )
    pasted = st.text_area("Or paste multiple tickets (one per line)", height=120)

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = pd.DataFrame(columns=["ticket"])
    else:
        df = pd.DataFrame(columns=["ticket"])

    if pasted and pasted.strip():
        extra = pd.DataFrame({"ticket": [x.strip() for x in pasted.splitlines() if x.strip()]})
        df = pd.concat([df, extra], ignore_index=True)

    if 'ticket' not in df.columns:
        st.warning("No 'ticket' column found yet. You can upload the provided sample file (sample_tickets.csv).")
    else:
        df['ticket'] = df['ticket'].fillna("").astype(str)

# ---------------- Process Button ----------------
process = st.button("🔎 Analyze Tickets", type="primary", use_container_width=True)

# ---------------- Analyzer Initialization ----------------
analyzer = SentimentAnalyzer(mode='fast')  # VADER only
tracker = TrendTracker()
kw = KeywordExtractor()

# ---------------- Main Analysis ----------------
if process and 'ticket' in df.columns and len(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = ensure_datetime(df['timestamp'])
    else:
        df['timestamp'] = pd.Timestamp.now()

    results = []
    for t in df['ticket'].tolist():
        try:
            r = analyzer.analyze_one(t)
        except Exception as e:
            logger.exception("Analyzer failure for ticket: %s", t)
            r = {'sentiment': 'neutral', 'sentiment_score': 0.0, 'emotion': 'neutral', 'emotion_score': 0.0}
        tracker.update(r.get('emotion'))
        results.append(r)

    res = pd.DataFrame(results)
    full = pd.concat([df.reset_index(drop=True), res.reset_index(drop=True)], axis=1)

    try:
        if date_from:
            full = full[full['timestamp'].dt.date >= date_from]
        if date_to:
            full = full[full['timestamp'].dt.date <= date_to]
    except Exception as e:
        st.warning("Date filtering skipped due to timestamp parsing issues.")
        logger.exception("Date filter error: %s", e)

    if full.shape[0] == 0:
        st.info("No tickets to analyze after applying filters. Try removing date filters or upload sample data.")
    else:
        # ---------------- Summary Cards ----------------
        c1, c2, c3, c4 = st.columns(4)
        pos = int((full['sentiment'] == 'positive').sum())
        neu = int((full['sentiment'] == 'neutral').sum())
        neg = int((full['sentiment'] == 'negative').sum())
        with c1: st.metric("✅ Positive", pos)
        with c2: st.metric("😐 Neutral", neu)
        with c3: st.metric("⚠️ Negative", neg)
        with c4:
            try:
                score = compute_happiness_score(pd.to_numeric(full['sentiment_score'], errors='coerce').fillna(0.0))
            except Exception:
                score = 0.0
            st.metric("😊 Happiness Score", f"{score:.1f}")

        # ---------------- Distribution Charts ----------------
        with st.container():
            c5, c6 = st.columns(2)
            with c5:
                fig = px.histogram(full, x="sentiment", title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with c6:
                fig2 = px.histogram(full, x="emotion", title="Emotion Distribution")
                st.plotly_chart(fig2, use_container_width=True)

        # ---------------- Trend Over Time ----------------
        try:
            full['date'] = pd.to_datetime(full['timestamp']).dt.date
            daily = full.groupby(['date', 'sentiment']).size().reset_index(name='count')
            min_date = pd.to_datetime(full['date']).min().date()
            max_date = pd.to_datetime(full['date']).max().date()
            all_dates = pd.date_range(start=min_date, end=max_date, freq='D').date
            all_sentiments = ['negative', 'neutral', 'positive']
            combos = pd.MultiIndex.from_product([all_dates, all_sentiments], names=['date', 'sentiment']).to_frame(index=False)
            daily_complete = combos.merge(daily, on=['date', 'sentiment'], how='left').fillna(0)
            daily_complete['count'] = daily_complete['count'].astype(int)
            daily_complete = daily_complete.sort_values(['date', 'sentiment']).reset_index(drop=True)

            if show_debug:
                st.subheader("Debug: daily_complete (first 20 rows)")
                st.dataframe(daily_complete.head(20))

            fig3 = px.line(
                daily_complete,
                x='date',
                y='count',
                color='sentiment',
                markers=True,
                title="📈 Trend of Sentiments Over Time",
                labels={'count': 'Number of Tickets', 'date': 'Date'}
            )
            fig3.update_traces(line=dict(width=3))
            fig3.update_layout(legend_title="Sentiment", hovermode="x unified")
            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            logger.exception("Trend plotting failed: %s", e)
            st.error("Trend Over Time failed to render. See debug table below for inspection.")

        # ---------------- Heatmap by Hour ----------------
        try:
            full['hour'] = pd.to_datetime(full['timestamp']).dt.hour
            heat = full.pivot_table(index='hour', columns='sentiment', values='ticket', aggfunc='count', fill_value=0).reset_index()
            if heat.shape[0] > 0 and heat.shape[1] > 1:
                fig4 = px.imshow(heat.set_index('hour').T, aspect='auto', title="Sentiment by Hour Heatmap")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Not enough data for hourly heatmap.")
        except Exception as e:
            logger.exception("Heatmap generation failed: %s", e)
            st.info("Heatmap skipped due to error.")

        # ---------------- Keywords ----------------
        try:
            top_yake = kw.extract_yake(full['ticket'].tolist())
            top_tfidf = kw.extract_tfidf(full['ticket'].tolist())
        except Exception as e:
            logger.exception("Keyword extraction failed: %s", e)
            top_yake, top_tfidf = [], []

        st.subheader("🔑 Top Keywords & Keyphrases")
        k1, k2 = st.columns(2)
        with k1:
            st.write("YAKE:", ", ".join(top_yake) if top_yake else "—")
        with k2:
            st.write("TF-IDF:", ", ".join(top_tfidf) if top_tfidf else "—")

        # ---------------- Clustering ----------------
        try:
            labels, cluster_terms = cluster_texts(full['ticket'].tolist(), k=min(max(2, k_clusters), 10))
            if len(full) and len(labels) == len(full):
                full['cluster'] = labels
            else:
                full['cluster'] = -1
            st.subheader("🧩 Topic Clusters")
            st.write({i: terms for i, terms in enumerate(cluster_terms)})
            fig5 = px.scatter(full.reset_index(), x='index', y='sentiment_score', color='cluster',
                              hover_data=['ticket'], title="Clusters vs Sentiment Score")
            st.plotly_chart(fig5, use_container_width=True)
        except Exception as e:
            logger.exception("Clustering failed: %s", e)
            st.info("Clustering skipped due to error.")

        # ---------------- Segmentation ----------------
        st.subheader("👥 Segmentation")
        if 'region' in full.columns:
            fig6 = px.histogram(full, x='region', color='sentiment', barmode='group', title="Sentiment by Region")
            st.plotly_chart(fig6, use_container_width=True)
        if 'plan' in full.columns:
            fig7 = px.histogram(full, x='plan', color='sentiment', barmode='group', title="Sentiment by Plan")
            st.plotly_chart(fig7, use_container_width=True)

        # ---------------- Alerts ----------------
        alert, ratio = tracker.check_alert(threshold=neg_threshold/100.0)
        if alert:
            st.error(f"⚠️ Negative sentiment spike detected! {ratio*100:.1f}% negative")
            alert_text = f"[Watchdog] Negative spike: {ratio*100:.1f}% negative"
            if slack_webhook:
                ok, msg = send_slack_webhook(slack_webhook, alert_text)
                st.caption(f"Slack: {msg}")
            if email_to and smtp_server and smtp_user and smtp_pass:
                ok, msg = send_email_alert(smtp_server, int(smtp_port), smtp_user, smtp_pass, email_to,
                                           "Watchdog Alert", alert_text)
                st.caption(f"Email: {msg}")
        else:
            st.success(f"✅ All clear. Negative emotions: {ratio*100:.1f}%")

        # ---------------- Drill-down Table ----------------
        st.subheader("📄 Tickets")
        sentiment_filter = st.multiselect("Filter by sentiment", options=['positive','neutral','negative'], default=[])
        emotion_filter = st.multiselect("Filter by emotion", options=sorted(full['emotion'].unique()), default=[])
        keyword_query = st.text_input("Search keyword")
        show_cols = ['timestamp','ticket','sentiment','sentiment_score','emotion','emotion_score']
        if 'region' in full.columns: show_cols.append('region')
        if 'plan' in full.columns: show_cols.append('plan')
        if 'cluster' in full.columns: show_cols.append('cluster')

        view = full.copy()
        if sentiment_filter:
            view = view[view['sentiment'].isin(sentiment_filter)]
        if emotion_filter:
            view = view[view['emotion'].isin(emotion_filter)]
        if keyword_query.strip():
            view = view[view['ticket'].str.contains(keyword_query.strip(), case=False, na=False)]
        st.dataframe(view[show_cols].sort_values('timestamp', ascending=False), use_container_width=True)

        # ---------------- Smart Recommendations ----------------
        st.subheader("🧠 Smart Recommendations")
        tips = []
        for _, row in view.iterrows():
            if row['sentiment']=='negative' or row['emotion'] in ['anger','fear','sadness','disgust']:
                text = row.get('ticket', "").lower()
                if 'refund' in text:
                    tips.append("Consider immediate refund or credit coupon.")
                if 'delay' in text or 'late' in text:
                    tips.append("Proactively notify user about revised ETA; offer expedited shipping.")
                if 'crash' in text or 'bug' in text:
                    tips.append("Create a bug ticket and prioritize hotfix; update status page.")
                if 'charge' in text or 'payment' in text:
                    tips.append("Escalate to billing; run duplicate charge audit and reassure user.")
        tips = list(dict.fromkeys(tips))[:6]
        if tips:
            st.write("• " + "\n• ".join(tips))
        else:
            st.write("No urgent recommendations detected.")

        # ---------------- Exports ----------------
        st.subheader("📤 Export")
        cexp1, cexp2, cexp3 = st.columns(3)
        with cexp1:
            csv_bytes = view.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv_bytes, file_name="watchdog_results.csv")
        with cexp2:
            to_excel = BytesIO()
            with pd.ExcelWriter(to_excel, engine='openpyxl') as writer:
                view.to_excel(writer, index=False, sheet_name="Results")
            st.download_button("Download Excel", data=to_excel.getvalue(), file_name="watchdog_results.xlsx")
        with cexp3:
            pdf_path = "watchdog_summary.pdf"
            export_summary_pdf(pdf_path, {
                "Positive": int(pos), "Neutral": int(neu), "Negative": int(neg),
                "Happiness Score": f"{score:.1f}", "Alert % Threshold": f"{neg_threshold}%"
            })
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Summary", data=f.read(), file_name="watchdog_summary.pdf")

else:
    st.info("Upload or paste tickets, then click **Analyze Tickets**. Or try the included `sample_tickets.csv`.")
