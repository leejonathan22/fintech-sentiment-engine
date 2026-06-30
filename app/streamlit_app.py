"""
Phase 4 — Streamlit UI: Fintech Sentiment Engine
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.brief_generator import (
    TOPIC_LABELS,
    _format_topic_context,
    SYSTEM_PROMPT,
    _USER_FRAMING,
    _DATA_NOTES,
    load_topic_data,
    MODEL,
)
from src.clustering import PROCESSED_DIR, run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fintech Sentiment Engine",
    page_icon="📊",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading topic data...")
def get_topic_data():
    summary_path = PROCESSED_DIR / "topic_summary.csv"
    reviews_path = PROCESSED_DIR / "reviews_with_topics.csv"

    if not summary_path.exists() or not reviews_path.exists():
        return None, None

    return load_topic_data(summary_path, reviews_path)


def run_clustering_pipeline():
    with st.spinner("Running BERTopic pipeline (~60s)..."):
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        run_pipeline()
    st.cache_data.clear()
    st.rerun()


# ── Brief streaming ───────────────────────────────────────────────────────────

def stream_brief(summary: pd.DataFrame, rep_docs: dict):
    from anthropic import Anthropic

    user_message = "\n\n".join([
        _USER_FRAMING,
        _format_topic_context(summary, rep_docs),
        _DATA_NOTES,
    ])

    client = Anthropic()
    with client.messages.stream(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


# ── Render ────────────────────────────────────────────────────────────────────

st.title("Fintech Sentiment Engine")
st.caption(
    "Google Play reviews · Chime · Cash App · Affirm · Klarna · Chase  "
    "| BERTopic clustering + Claude strategic brief"
)

summary, rep_docs = get_topic_data()

if summary is None:
    st.warning("No processed data found in `data/processed/`.")
    if st.button("Run clustering pipeline (~60s)"):
        run_clustering_pipeline()
    st.stop()

tab_topics, tab_brief = st.tabs(["Topics", "Strategic Brief"])

# ── Tab 1: Topics ─────────────────────────────────────────────────────────────

with tab_topics:
    st.subheader("Cluster overview")

    n_generic = 0
    full_summary_path = PROCESSED_DIR / "topic_summary.csv"
    if full_summary_path.exists():
        full = pd.read_csv(full_summary_path)
        generic_row = full[full["topic"] == -2]
        if not generic_row.empty:
            n_generic = int(generic_row["docs"].iloc[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Meaningful topics", len(summary))
    col2.metric("Reviews clustered", int(summary["docs"].sum()))
    col3.metric("Generic-positive excluded", n_generic)

    st.divider()

    display = summary[["topic", "docs", "avg_rating", "app_breakdown"]].copy()
    display.insert(1, "theme", display["topic"].map(TOPIC_LABELS))
    display["app_breakdown"] = display["app_breakdown"].str.replace("  ", " · ")
    display = display.set_index("topic")

    st.dataframe(
        display.style.background_gradient(
            subset=["avg_rating"], cmap="RdYlGn", vmin=1, vmax=5
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Representative reviews")

    reviews_path = PROCESSED_DIR / "reviews_with_topics.csv"
    if reviews_path.exists():
        reviews_df = pd.read_csv(reviews_path)
        for tid, theme in TOPIC_LABELS.items():
            subset = reviews_df[reviews_df["topic"] == tid].dropna(subset=["text"])
            if subset.empty:
                continue
            row = summary[summary["topic"] == tid]
            avg_r = float(row["avg_rating"].iloc[0]) if not row.empty else 0
            reps = (
                subset.assign(_len=subset["text"].str.len())
                .sort_values("_len", ascending=False)["text"]
                .head(3)
                .tolist()
            )
            label = f"Topic {tid} · {theme} · {avg_r}★"
            with st.expander(label):
                for i, doc in enumerate(reps, 1):
                    st.markdown(f"**[{i}]** {doc}")

# ── Tab 2: Strategic Brief ────────────────────────────────────────────────────

with tab_brief:
    st.subheader("PM-ready strategic brief")
    st.caption(f"Model: `{MODEL}` · ~400–600 words · synthesizes across all 9 topics")

    if "brief_text" not in st.session_state:
        st.session_state.brief_text = ""

    col_btn, col_clear = st.columns([1, 5])
    generate = col_btn.button("Generate brief", type="primary")
    if col_clear.button("Clear"):
        st.session_state.brief_text = ""

    if generate:
        st.session_state.brief_text = ""
        brief_placeholder = st.empty()
        streamed = ""
        for chunk in stream_brief(summary, rep_docs):
            streamed += chunk
            brief_placeholder.markdown(streamed + "▌")
        st.session_state.brief_text = streamed
        brief_placeholder.markdown(streamed)
    elif st.session_state.brief_text:
        st.markdown(st.session_state.brief_text)
    else:
        st.info("Click **Generate brief** to synthesize a strategic memo from the cluster data.")
