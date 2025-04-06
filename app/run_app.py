import os
import sys
import csv
from datetime import datetime

import streamlit as st
import pandas as pd
from textblob import TextBlob

# Project-level imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from emotion_toggle import get_cont_emotions
from models.retriever import rank_documents_by_query


# Ensure feedback directory exists
os.makedirs("data/raw", exist_ok=True)

# Emotion categories and their descriptions
emotion_descriptions = {
    "Neutral": "Flat or objective tone",
    "Hopeful": "Includes joy, optimism, anticipation",
    "Angry": "Includes anger, annoyance",
    "Fearful": "Includes fear",
    "Frustrated": "Includes disappointment",
    "Empowered": "Includes pride, gratitude, approval"
}

# Initialize session state
for key, default in {
    "query_text": "",
    "corrected_query_ready": False,
    "last_query": "",
    "last_emotion": "",
    "search_triggered": False,
    "use_emotion_filter": False,
    "use_cont_filter": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- UI Rendering ---

st.title("Emotion-Aware Social Justice Search")

# Autocorrect suggestion using TextBlob
if st.session_state.corrected_query_ready:
    st.session_state.query_text = st.session_state.corrected_query_value
    st.session_state.corrected_query_ready = False

query = st.text_input("Enter your search query:", key="query_text")

if st.session_state.query_text:
    corrected = str(TextBlob(st.session_state.query_text).correct())
    if corrected.lower() != st.session_state.query_text.lower():
        st.warning(f"Did you mean: **{corrected}**?")
        if st.button("Use corrected query"):
            st.session_state.corrected_query_value = corrected
            st.session_state.corrected_query_ready = True
            st.rerun()

query = st.session_state.query_text
emotion_options = ["— Select how you're feeling —"] + list(emotion_descriptions.keys())
emotion = st.selectbox("How are you feeling?", emotion_options)
valid_emotion_selected = emotion in emotion_descriptions

# Filter options
col1, col2 = st.columns(2)
with col1:
    st.checkbox("Apply emotion-based ranking", key="use_emotion_filter",
                on_change=lambda: st.session_state.update(use_cont_filter=False, search_triggered=False))
with col2:
    st.checkbox("Show contrasting emotion results", key="use_cont_filter",
                on_change=lambda: st.session_state.update(use_emotion_filter=False, search_triggered=False))

# Determine final emotion for filtering
if valid_emotion_selected and (st.session_state.use_emotion_filter or st.session_state.use_cont_filter):
    final_emotion = emotion
else:
    final_emotion = "Neutral"

# Trigger search
if st.button("Search"):
    if not valid_emotion_selected:
        st.error("Please select how you're feeling before searching.")
        st.stop()

    st.session_state.last_query = query
    st.session_state.last_emotion = final_emotion
    st.session_state.search_triggered = True

# --- Results Display ---
if st.session_state.search_triggered and st.session_state.last_query:
    query = st.session_state.last_query
    final_emotion = st.session_state.last_emotion

    st.success(f"Query: {query}")
    st.info(f"Emotion selected: {final_emotion}")
    st.markdown("---")

    # Determine retrieval mode
    if st.session_state.use_emotion_filter:
        st.caption(f"Retrieving results aligned with your emotion: *{final_emotion}* — {emotion_descriptions[final_emotion]}")
        mode = "aligned"
    elif st.session_state.use_cont_filter:
        contrasting = ', '.join(get_cont_emotions(final_emotion))
        st.caption(f"Retrieving contrasting results for *{final_emotion}* → Showing: {contrasting}")
        mode = "cont"
    else:
        st.caption(f"No emotion filter applied. Defaulting to *Neutral* — {emotion_descriptions['Neutral']}")
        mode = "aligned"

    # Load and rank corpus
    corpus_df = pd.read_csv("data/corpus/corpus.csv")
    alpha = st.slider("Relevance weight (α)", 0.0, 1.0, 0.7)
    beta = 1.0 - alpha

    ranked_docs = rank_documents_by_query(corpus_df, query, final_emotion, mode=mode, alpha=alpha, beta=beta)

    # Pagination controls
    results_per_page = 10
    total_pages = (len(ranked_docs) - 1) // results_per_page + 1
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (page_number - 1) * results_per_page
    end_idx = start_idx + results_per_page
    paged_docs = ranked_docs.iloc[start_idx:end_idx]

    # Render results with feedback
    st.subheader("Top-Ranked Results")
    for i, row in paged_docs.iterrows():
        st.write(f"### [{row['title']}]({row['url']})")
        st.write(f"**Emotion Match**: {row['emotion_match']:.2f} | **Relevance**: {row['relevance']:.2f}")
        st.write(row.get('content', row.get('title', '[No content available]')))
        st.write(f"**Final Score**: {row['final_score']:.2f}")
        st.caption(
            f"Score = α × {row['relevance']:.2f} + β × {row['emotion_match']:.2f} "
            f"(α = {alpha:.2f}, β = {beta:.2f})"
        )

        # Feedback form
        with st.form(key=f"feedback_form_{i}"):
            feedback_choice = st.radio(
                "Was this result relevant?",
                ["👍 Relevant", "👎 Not Relevant"],
                key=f"radio_{i}",
                horizontal=True
            )
            if st.form_submit_button("Submit Feedback"):
                feedback_path = "data/raw/feedback_log.csv"
                write_header = not os.path.exists(feedback_path) or os.path.getsize(feedback_path) == 0

                with open(feedback_path, "a", newline='', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([
                            "query", "emotion", "mode", "doc_title", "doc_content",
                            "feedback", "timestamp"
                        ])
                    writer.writerow([
                        query, final_emotion, mode, row['title'], row['title'],
                        feedback_choice, datetime.now().isoformat()
                    ])
                st.success("Feedback recorded.")

        st.markdown("---")
