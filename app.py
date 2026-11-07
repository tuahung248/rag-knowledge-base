# app.py — Streamlit UI for AI Engineering Knowledge Base RAG

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from dotenv import load_dotenv
from src.rag_chain import ask_with_sources

load_dotenv()

# Streamlit Community Cloud exposes secrets via st.secrets; copy them into
# the environment so the rest of the code can use os.getenv everywhere.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:
    pass  # no secrets file locally; .env is used instead

# --- Page config ---
st.set_page_config(
    page_title="AI Engineering Knowledge Base",
    page_icon="🤖",
    layout="centered",
)

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 AI Engineering KB")
    st.markdown(
        """
        A RAG-powered Q&A bot over AI engineering concepts.

        **Built with:**
        - LangChain LCEL
        - Voyage AI embeddings
        - Supabase pgvector
        - Google Gemini 3.5 Flash
        """
    )
    st.divider()
    st.markdown("**Try asking:**")
    example_questions = [
        "What is RAG and how does it work?",
        "How does RAGAS evaluate retrieval quality?",
        "What is the ReAct agent pattern?",
        "What observability tools exist for LLM systems?",
        "What are the chunking strategies for RAG?",
        "How do embeddings work?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["question_input"] = q

    st.divider()
    st.caption("P0 portfolio project — Hudson Bui")

# --- Main area ---
st.title("AI Engineering Knowledge Base")
st.markdown(
    "Ask anything about RAG, embeddings, prompt engineering, LLMOps, or AI agents."
)

# Question input
question = st.text_input(
    "Your question:",
    placeholder="What is retrieval-augmented generation?",
    key="question_input",
)

# Visitor's own Gemini key: only needed when asking a question
user_gemini_key = st.text_input(
    "Your Gemini API key (needed to ask):",
    type="password",
    key="user_gemini_key",
    help="Free at aistudio.google.com. Used only for this request; not stored.",
)

# Submit
col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)
with col2:
    st.empty()

# --- Answer ---
if submit and not user_gemini_key.strip():
    st.warning("Enter your Gemini API key above to ask a question. Free at aistudio.google.com.")
elif submit and question.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            answer, source_docs = ask_with_sources(
                question.strip(), gemini_api_key=user_gemini_key.strip()
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.markdown("### Answer")
    st.markdown(answer)

    # Sources
    st.markdown("---")
    with st.expander(f"📄 Sources ({len(source_docs)} chunks retrieved)"):
        for i, doc in enumerate(source_docs):
            source_file = Path(
                doc.metadata.get("source", "unknown")
            ).name
            st.markdown(f"**[{i+1}] `{source_file}`**")
            st.markdown(f"> {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")
            if i < len(source_docs) - 1:
                st.divider()

elif submit and not question.strip():
    st.warning("Please enter a question.")
