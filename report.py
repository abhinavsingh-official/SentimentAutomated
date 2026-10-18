"""The original PDF report (sentiment.py/sentiment_limited.py), refactored
to read from the SQLite store instead of a fresh CSV - now an optional
export, not the pipeline's primary output. Kept almost unchanged
chart-for-chart; only the data source changed."""

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud

import store

LABELS = ["Negative", "Neutral", "Positive"]
COLORS = {"Negative": "#F08080", "Neutral": "#F0E68C", "Positive": "#66CDAA"}


def generate_report(db_path=store.DB_PATH) -> bytes:
    posts = store.all_posts(db_path)
    df = pd.DataFrame(posts)
    if df.empty:
        raise ValueError("No posts stored yet - run the pipeline first.")

    sns.set(style="whitegrid")
    sentiment_counts = df["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    avg_conf = df["confidence"].mean()
    high_conf_count = (df["confidence"] > 0.9).sum()

    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        summary = f"""
Sentiment Analysis Report

Overview:
{len(df)} posts across {df['topic'].nunique()} topics and {df['source'].nunique()} source(s),
scored with cardiffnlp/twitter-roberta-base-sentiment.

Analysis Summary:
- Positive: {sentiment_counts['Positive']}
- Neutral:  {sentiment_counts['Neutral']}
- Negative: {sentiment_counts['Negative']}

Confidence:
- Average prediction confidence: {avg_conf:.2f}
- High-confidence predictions (> 0.9): {high_conf_count}

Dominant sentiment: {sentiment_counts.idxmax()}
        """
        plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.05, 0.95, summary, ha="left", va="top", wrap=True, fontsize=13, linespacing=1.6)
        pdf.savefig()
        plt.close()

        plt.figure()
        sentiment_counts.plot(kind="bar", color=[COLORS[l] for l in LABELS])
        plt.title("Sentiment Count")
        plt.ylabel("Posts")
        plt.xticks(rotation=0)
        pdf.savefig()
        plt.close()

        plt.figure()
        sentiment_counts.plot(kind="pie", autopct="%1.1f%%", ylabel="", colors=[COLORS[l] for l in LABELS])
        plt.title("Sentiment Proportion")
        pdf.savefig()
        plt.close()

        for topic in df["topic"].unique():
            topic_df = df[df["topic"] == topic]
            counts = topic_df["sentiment"].value_counts().reindex(LABELS, fill_value=0)
            plt.figure()
            counts.plot(kind="bar", color=[COLORS[l] for l in LABELS])
            plt.title(f"Sentiment - {topic}")
            plt.ylabel("Posts")
            plt.xticks(rotation=0)
            pdf.savefig()
            plt.close()

        for label in LABELS:
            texts = df[df["sentiment"] == label]["text"].dropna().astype(str)
            if texts.empty:
                continue
            words = " ".join(texts)
            wc = WordCloud(width=800, height=400, background_color="white").generate(words)
            plt.figure(figsize=(8, 4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud: {label}")
            pdf.savefig()
            plt.close()

    return buffer.getvalue()
