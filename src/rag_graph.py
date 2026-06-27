"""The RAG pipeline, modelled as a LangGraph state machine.

Flow:  rewrite_query  ->  retrieve  ->  generate

    +----------------+     +------------+     +--------------------------+
    | rewrite_query  | --> |  retrieve  | --> |        generate          |
    | (resolve       |     | (FAISS top |     | (grounded answer + cite, |
    |  follow-ups)   |     |  k chunks) |     |  hallucination guard)    |
    +----------------+     +------------+     +--------------------------+

Why a graph instead of one big function: each step is an isolated, testable
node with explicit state. It makes the data flow obvious, is trivial to extend
(add re-ranking or a "grade documents" node between retrieve and generate), and
is exactly the shape this kind of agentic RAG is meant to be drawn as.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import settings
from src.ingest import load_index


NO_ANSWER = (
    "I could not find this information in the provided documents."
)

class GroundedAnswer(BaseModel):
    """The LLM's structured response."""

    sufficient_context: bool = Field(
        description="True only if the provided context actually contains the "
        "information needed to answer the question."
    )
    answer: str = Field(
        description="A concise answer grounded ONLY in the context, with inline "
        "citations like [1], [2]. If sufficient_context is false, explain that "
        "the answer is not available in the documents."
    )


class RAGState(TypedDict, total=False):
    question: str                       # original user question
    chat_history: list[dict]            # [{"role": "user"/"assistant", "content": ...}]
    query: str                          # standalone query after rewriting
    documents: list[Document]           # retrieved chunks
    relevances: list[float]             # per-chunk relevance in (0, 1]
    answer: str                         # final answer text
    sources: list[dict]                 # [{"document": ..., "page": ...}]
    confidence: float                   # 0..1


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite a user's latest question into a standalone search query "
            "using the conversation so far. Resolve pronouns and references. "
            "Return ONLY the rewritten query, nothing else.",
        ),
        ("placeholder", "{history}"),
        ("human", "Latest question: {question}\n\nStandalone query:"),
    ]
)

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise knowledge assistant. Answer the question using "
            "ONLY the numbered context passages below. Follow these rules strictly:\n"
            "1. Base every claim on the context. Do NOT use outside knowledge.\n"
            "2. Cite the passages you use with inline markers like [1], [2].\n"
            "3. Be concise and direct — answer the question, nothing more.\n"
            "4. If the context does not contain the answer, set sufficient_context "
            "to false and say the information is not available.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# --------------------------------------------------------------------------- #
# Lazily-built singletons (built once per process)
# --------------------------------------------------------------------------- #
@lru_cache
def _get_vectorstore() -> FAISS:
    return load_index()


@lru_cache
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )


def _format_context(documents: list[Document]) -> str:
    """Render retrieved chunks as a numbered, citation-friendly context block."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}, page {page}" if page is not None else source
        blocks.append(f"[{i}] (Source: {location})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def _dedupe_sources(documents: list[Document]) -> list[dict]:
    """Collapse retrieved chunks into unique (document, page) citations, in order."""
    sources: list[dict] = []
    seen: set[tuple[str, object]] = set()
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"document": source, "page": page})
    return sources


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
def rewrite_query(state: RAGState) -> RAGState:
    """Turn a possibly-contextual follow-up into a standalone query.

    Skipped entirely when there's no history — saves a round-trip on the common
    single-shot case.
    """
    question = state["question"]
    history = state.get("chat_history") or []
    if not history:
        return {"query": question}

    messages = [(m["role"], m["content"]) for m in history]
    chain = REWRITE_PROMPT | _get_llm()
    result = chain.invoke({"history": messages, "question": question})
    return {"query": result.content.strip() or question}


def retrieve(state: RAGState) -> RAGState:
    """Semantic search over FAISS, keeping a relevance score per chunk.

    We use raw L2 distance and map it to a (0, 1] relevance via 1/(1+distance):
    monotonic, always in range, and good enough to drive a confidence signal
    without depending on a normalised score function.
    """
    query = state.get("query") or state["question"]
    results = _get_vectorstore().similarity_search_with_score(query, k=settings.top_k)

    documents = [doc for doc, _ in results]
    relevances = [1.0 / (1.0 + float(distance)) for _, distance in results]
    return {"documents": documents, "relevances": relevances}


def generate(state: RAGState) -> RAGState:
    """Generate a grounded, cited answer and compute a confidence score."""
    documents = state.get("documents") or []

    # No documents retrieved at all -> short-circuit to "not available".
    if not documents:
        return {"answer": NO_ANSWER, "sources": [], "confidence": 0.0}

    context = _format_context(documents)
    structured_llm = _get_llm().with_structured_output(GroundedAnswer)
    chain = GENERATE_PROMPT | structured_llm
    result: GroundedAnswer = chain.invoke(
        {"context": context, "question": state["question"]}
    )

    relevances = state.get("relevances") or [0.0]
    # Confidence = how relevant the retrieved evidence was, gated by whether the
    # model judged that evidence sufficient. An unsupported answer is capped low.
    retrieval_conf = sum(relevances) / len(relevances)
    if result.sufficient_context:
        confidence = retrieval_conf
        sources = _dedupe_sources(documents)
        answer = result.answer
    else:
        confidence = min(retrieval_conf, 0.2)
        sources = []
        answer = result.answer or NO_ANSWER

    return {"answer": answer, "sources": sources, "confidence": round(confidence, 2)}


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
@lru_cache
def get_graph():
    """Build and compile the LangGraph pipeline (cached per process)."""
    builder = StateGraph(RAGState)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "rewrite_query")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def answer_question(question: str, chat_history: list[dict] | None = None) -> dict:
    """Public entry point used by the API, UI, and evaluation harness.

    Returns a dict: {question, answer, sources, confidence}.
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    final_state = get_graph().invoke(
        {"question": question.strip(), "chat_history": chat_history or []}
    )
    return {
        "question": question.strip(),
        "answer": final_state.get("answer", NO_ANSWER),
        "sources": final_state.get("sources", []),
        "confidence": final_state.get("confidence", 0.0),
    }


if __name__ == "__main__":
    import json

    demo = answer_question("What is the employee leave policy?")
    print(json.dumps(demo, indent=2))
