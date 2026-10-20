"""Streamlit dashboard for SentimentAutomated - reads from the SQLite
store built by pipeline.py, rather than a one-shot CSV-in/PDF-out script.
Includes a "Try it" tab that runs the real pipeline on a sample of the
bundled Twitter dataset live, so the project is interactive even without
Reddit/NewsAPI credentials configured."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import store
from alerts import check_alerts
from pipeline import run_pipeline
from process import SentimentModel
from topics import top_noun_chunks

st.set_page_config(page_title="SentimentAutomated", layout="wide")

STYLE = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
h1 { font-weight: 700; letter-spacing: -0.02em; color: #ff6f91; }
.sa-card {
    background: #ffe8d6; border: 1px solid #ff6f91; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.sa-feature-card {
    background: #ffe8d6; border: 1px solid #ff6f91; border-radius: 12px;
    padding: 1.3rem 1.4rem; height: 100%;
}
.sa-feature-card h4 { margin-top: 0; color: #c77d00; }
.sa-feature-card ul { margin: 0.4rem 0 0 0; padding-left: 1.2rem; color: #4a3b35; }
.sa-feature-card li { margin-bottom: 0.35rem; }
.sa-stat { font-size: 2rem; font-weight: 700; color: #ff6f91; }
.sa-stat-label { color: #7a655d; font-size: 0.9rem; }
.sa-alert {
    background: #fff3cd; border: 1px solid #ffc93c; border-radius: 10px;
    padding: 0.9rem 1.1rem; color: #6b5420; font-size: 0.92rem; margin-bottom: 0.6rem;
}
</style>
"""

ENTITY_CHOICES = [
    "Borderlands", "Amazon", "Google", "Microsoft", "Xbox(Xseries)",
    "PlayStation5(PS5)", "NBA2K", "Overwatch", "Fortnite", "CS-GO",
]


@st.cache_resource
def load_model() -> SentimentModel:
    return SentimentModel.load()


def home_tab():
    stats = store.stats()
    active_alerts = check_alerts()

    st.markdown(
        '<div class="sa-card" style="padding:2rem 2.2rem;">'
        "<h2 style='margin-top:0;'>A sentiment pipeline, not a one-shot script</h2>"
        "<p style='color:#4a3b35; font-size:1.05rem; max-width:760px; line-height:1.6;'>"
        "Ingest, score, store, and watch for trouble - not just run once on a "
        "CSV and print a PDF."
        "</p>"
        "<ul>"
        "<li><b>Ingest</b> - Reddit posts/comments, news headlines, or a batch CSV</li>"
        "<li><b>Score</b> - a RoBERTa model fine-tuned for sentiment, batched for real throughput</li>"
        "<li><b>Store</b> - SQLite, so history builds up instead of getting overwritten</li>"
        "<li><b>Alert</b> - flags a topic if its negative share crosses a threshold</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Where things stand right now")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="sa-feature-card"><div class="sa-stat">{stats["total_posts"]}</div>'
            '<div class="sa-stat-label">Posts stored</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="sa-feature-card"><div class="sa-stat">{stats["topic_count"]}</div>'
            '<div class="sa-stat-label">Topics tracked</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="sa-feature-card"><div class="sa-stat">{len(active_alerts)}</div>'
            '<div class="sa-stat-label">Active alerts</div></div>',
            unsafe_allow_html=True,
        )

    if active_alerts:
        st.markdown("#### Active alerts")
        for alert in active_alerts:
            st.markdown(
                f'<div class="sa-alert">"{alert.topic}" is {alert.negative_share:.0%} negative '
                f"over the last {alert.window_hours}h ({alert.sample_size} posts) - "
                f"above the {alert.threshold:.0%} threshold.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### What each tab does")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="sa-feature-card"><h4>Try It</h4>'
            "<ul>"
            "<li>Pick a topic from the bundled dataset and ingest a live sample</li>"
            "<li>No Reddit/NewsAPI credentials needed for this demo path</li>"
            "<li>Watch the pipeline accept/reject/score/store in real time</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="sa-feature-card"><h4>Dashboard</h4>'
            "<ul>"
            "<li>Sentiment trend per topic, and by source where more than one exists</li>"
            "<li>What people are actually talking about, not just overall sentiment</li>"
            "<li>A PDF export of the full report, generated from stored data</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:#7a655d; margin-top:1.6rem;">No data yet? Head to <b>Try It</b> to '
        "ingest a sample.</p>",
        unsafe_allow_html=True,
    )


def try_it_tab():
    st.markdown(
        '<p style="color:#7a655d;">Pulls a small batch from the bundled Twitter dataset for '
        "the topic you pick, runs it through the real pipeline (quality gate -> sentiment "
        "scoring -> storage), and shows you what happened at each stage.</p>",
        unsafe_allow_html=True,
    )

    # st.rerun() below re-executes the whole script, which would otherwise
    # wipe out an st.success() called just before it - stash the summary in
    # session_state so it survives the rerun and actually gets seen.
    if "last_ingest_summary" in st.session_state:
        summary = st.session_state.pop("last_ingest_summary")
        st.success(
            f"Ingested {summary['ingested']} - "
            f"{summary['accepted']} passed the quality gate, "
            f"{summary['rejected']} rejected, "
            f"{summary['stored']} newly stored (duplicates from a prior run are skipped)."
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        entity = st.selectbox("Topic", ENTITY_CHOICES)
    with col2:
        sample_size = st.number_input("Sample size", min_value=10, max_value=200, value=50, step=10)

    if st.button("Ingest sample"):
        with st.spinner(f"Ingesting and scoring {sample_size} posts about {entity}..."):
            # load_batch_csv() treats a whole file as one topic, which isn't
            # right here since this dataset covers many entities in one
            # file - filter to the chosen entity first instead.
            df = pd.read_csv("twitter_training.csv")
            subset = df[df["Entity"] == entity].head(sample_size)
            posts = [
                {"post_id": str(r["Tweet ID"]), "text": str(r["Tweet"]), "topic": entity.lower(), "created_at": None}
                for _, r in subset.iterrows()
            ]
            st.session_state["last_ingest_summary"] = run_pipeline(posts, source="batch_csv", model=load_model())
        st.rerun()


def dashboard_tab():
    posts = store.all_posts()
    if not posts:
        st.info("No data yet - head to the Try It tab to ingest a sample.")
        return

    df = pd.DataFrame(posts)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").fillna(pd.to_datetime(df["ingested_at"]))

    st.markdown("#### Sentiment by topic")
    topic_counts = df.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for sentiment, color in [("Positive", "#66CDAA"), ("Neutral", "#F0E68C"), ("Negative", "#F08080")]:
        if sentiment in topic_counts.columns:
            fig.add_trace(go.Bar(name=sentiment, x=topic_counts.index, y=topic_counts[sentiment], marker_color=color))
    fig.update_layout(barmode="stack", template="plotly_white", height=400, paper_bgcolor="#fffaf5", plot_bgcolor="#fffaf5")
    st.plotly_chart(fig, use_container_width=True)

    if df["source"].nunique() > 1:
        st.markdown("#### Sentiment by source")
        source_counts = df.groupby(["source", "sentiment"]).size().unstack(fill_value=0)
        fig2 = go.Figure()
        for sentiment, color in [("Positive", "#66CDAA"), ("Neutral", "#F0E68C"), ("Negative", "#F08080")]:
            if sentiment in source_counts.columns:
                fig2.add_trace(go.Bar(name=sentiment, x=source_counts.index, y=source_counts[sentiment], marker_color=color))
        fig2.update_layout(barmode="group", template="plotly_white", height=350, paper_bgcolor="#fffaf5", plot_bgcolor="#fffaf5")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### What people are actually talking about")
    st.markdown(
        '<p style="color:#7a655d;">Noun phrases pulled straight from stored text - open-ended, '
        "not limited to the topic labels above.</p>",
        unsafe_allow_html=True,
    )
    chunks = top_noun_chunks(df["text"].tolist(), top_n=15)
    if chunks:
        chunk_df = pd.DataFrame(chunks, columns=["phrase", "mentions"])
        st.bar_chart(chunk_df.set_index("phrase")["mentions"])

    st.markdown("#### Recent posts")
    recent = df.sort_values("created_at", ascending=False).head(50)
    st.dataframe(
        recent[["topic", "source", "sentiment", "confidence", "text"]],
        use_container_width=True, hide_index=True,
    )

    rejections = store.recent_rejections(limit=200)
    if rejections:
        st.markdown("#### What the quality gate rejected")
        reject_df = pd.DataFrame(rejections)
        reason_counts = reject_df["reason"].value_counts()
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.dataframe(reason_counts.rename("count"), use_container_width=True)
        with rc2:
            fig3 = go.Figure(go.Bar(x=reason_counts.index, y=reason_counts.values, marker_color="#ff6f91"))
            fig3.update_layout(template="plotly_white", height=280, paper_bgcolor="#fffaf5", plot_bgcolor="#fffaf5")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Export")
    if st.button("Generate PDF report"):
        from report import generate_report
        with st.spinner("Building report..."):
            pdf_bytes = generate_report()
        st.download_button("Download report.pdf", pdf_bytes, file_name="sentiment_report.pdf", mime="application/pdf")


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("SentimentAutomated")
    st.caption("A live sentiment pipeline: ingest, score, store, alert - across topics and sources.")
    tab0, tab1, tab2 = st.tabs(["Home", "Try It", "Dashboard"])
    with tab0:
        home_tab()
    with tab1:
        try_it_tab()
    with tab2:
        dashboard_tab()


if __name__ == "__main__":
    main()
