
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import settings
from src.rag_graph import answer_question

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚")

st.title("📚 Enterprise Knowledge Assistant")
st.caption(
    f"Ask questions about internal documents. Answers are grounded in the "
    f"knowledge base and cite their sources. · Model: `{settings.llm_model}`"
)


if "messages" not in st.session_state:
    st.session_state.messages = []  


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    page = f", page {src['page']}" if src.get("page") is not None else ""
                    st.markdown(f"- **{src['document']}**{page}")

if prompt := st.chat_input("Ask a question, e.g. 'What is the refund policy?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                result = answer_question(prompt, chat_history=history)
            except FileNotFoundError:
                st.error(
                    "No index found. Run `python -m src.ingest` to build it first."
                )
                st.stop()

        st.markdown(result["answer"])

        confidence = result["confidence"]
        st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence: {confidence:.0%}")

        if result["sources"]:
            with st.expander("📎 Sources"):
                for src in result["sources"]:
                    page = f", page {src['page']}" if src.get("page") is not None else ""
                    st.markdown(f"- **{src['document']}**{page}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
