# System Design — Enterprise Knowledge Assistant

## 1. High-level architecture

The system has two clearly separated phases: an **offline ingestion** phase that
builds a searchable index, and an **online query** phase that answers questions.
This separation keeps the serving path fast (no re-indexing per request) and
makes ingestion a job you can run on a schedule.

```
                 ┌──────────────────── INGESTION (offline) ────────────────────┐
                 │                                                              │
  data/*.{md,pdf,txt} ─► Loaders ─► Chunker ─► OpenAI Embeddings ─► FAISS index │
                 │   (PyPDF/Text)  (Recursive,  (text-embedding-     (.faiss/)  │
                 │                  1000/150)    3-small)                        │
                 └──────────────────────────────────────────────────────────────┘

                 ┌──────────────────── QUERY (online, LangGraph) ──────────────┐
   question ────►│  rewrite_query ──► retrieve ──► generate                     │──► { answer,
   (+history)    │   resolve         FAISS         grounded answer,             │     sources,
                 │   follow-ups      top-k=4       [n] citations,               │     confidence }
                 │                                 confidence, "don't know"     │
                 └──────────────────────────────────────────────────────────────┘
       ▲                                                                   │
       └────────────── Streamlit UI · FastAPI /ask · CLI ──────────────────┘
```

## 2. Data flow

1. **Ingest** (`src/ingest.py`): each file is loaded into LangChain `Document`s.
   PDFs are loaded per page (page numbers preserved, normalised to 1-indexed) for
   precise citations; Markdown/text loaders attach the filename as `source`.
2. **Chunk**: `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) splits on
   document structure first (headings, paragraphs) so chunks stay coherent.
3. **Embed + index**: chunks are embedded with OpenAI `text-embedding-3-small`
   and stored in a FAISS index persisted to disk.
4. **Query** (`src/rag_graph.py`): the compiled LangGraph runs three nodes —
   `rewrite_query` → `retrieve` → `generate` — and returns the answer, citations,
   and a confidence score.
5. **Serve**: the same `answer_question()` entry point backs the Streamlit UI,
   the FastAPI endpoint, and the evaluation harness, so all three behave
   identically.

## 3. Component explanation

| Component | File | Responsibility |
|-----------|------|----------------|
| **Config** | `config.py` | One typed `Settings` object from `.env`; nothing else reads the environment directly. |
| **Ingestion** | `ingest.py` | Load → chunk → embed → persist FAISS. Also the single source of truth for the embedding model and index load/save. |
| **RAG graph** | `rag_graph.py` | The LangGraph state machine: query rewriting, retrieval, grounded generation, citation assembly, confidence. |
| **API** | `api.py` | FastAPI `/ask` + `/health`; typed request/response, actionable errors. |
| **UI** | `app.py` | Streamlit chat with memory, source expanders, confidence bar. |
| **Evaluation** | `evaluation/` | Test set + automated scoring harness. |

### Why LangGraph (not a single function or a plain chain)
Modelling RAG as a graph makes each step an **isolated, testable node** with
explicit state. The data flow is obvious, and the pipeline is trivial to extend:
dropping in a re-ranking node or a "grade documents → re-retrieve" loop is a
one-node change rather than a refactor. It's also the most natural thing to draw
and explain.

### Key design decisions
- **Chunking — 1000/150, structure-aware.** Large enough to preserve context for
  policy-style prose, small enough for precise retrieval; the 150-char overlap
  prevents facts from being split across a boundary. Splitting on headings first
  keeps chunks semantically whole.
- **Hallucination prevention — structured output guard.** The LLM returns a
  structured object `{ sufficient_context: bool, answer: str }`. "I don't know"
  becomes a **machine-readable outcome** instead of something we pattern-match.
  When context is insufficient, sources are dropped and confidence is capped.
- **Prompt design.** The system prompt pins the model to the numbered context,
  forbids outside knowledge, mandates `[n]` inline citations, and instructs it to
  decline when the answer isn't present. `temperature=0` keeps answers stable.
- **Citations are programmatic, not model-trusted.** The source list is built
  from the **metadata of the chunks actually retrieved** (deduped by document +
  page), so a citation can never point at a document that wasn't used.
- **Confidence score.** Derived from FAISS retrieval distance — each chunk's
  relevance is `1/(1+distance)`, averaged — then gated by the model's
  `sufficient_context` flag. Transparent and dependency-free, not a fabricated
  number from the LLM.

## 4. Evaluation approach

`evaluation/evaluate.py` runs a labelled question set (`test_cases.json`) through
the pipeline and measures three things:

- **Answer accuracy** — does an answerable question's response contain the
  expected fact(s)? (keyword match)
- **Citation accuracy** — is the correct source document cited?
- **Hallucination avoidance** — are out-of-scope questions (e.g. *"pet insurance
  policy?"*) correctly **refused** rather than answered?

This keyword/source-based grading is deterministic, free, and fast — ideal for
catching regressions in CI. Improvements attempted / planned:
- An **LLM-as-judge** pass scoring *faithfulness* (is every claim supported by the
  cited context?) and *answer relevance*, à la RAGAS.
- **Retrieval metrics** (hit-rate / MRR@k) by labelling the gold chunk per question.
- Expanding the set with paraphrases and adversarial/ambiguous questions.

## 5. Scalability considerations

The architecture is small by design but built to scale along clear seams:

- **Vector store.** FAISS (in-memory, local) is perfect for hundreds–thousands of
  chunks. For large or multi-tenant corpora, swap FAISS for a managed/distributed
  store (pgvector, Pinecone, Weaviate, Qdrant). Only `ingest.py` and one helper in
  `rag_graph.py` touch the store, so the swap is localized.
- **Ingestion.** Currently a full rebuild. At scale: incremental/delta ingestion
  keyed by document hash, batched embedding calls, and a background worker queue.
- **Serving.** The FastAPI app is stateless → horizontally scalable behind a load
  balancer. Embedding/LLM calls are the latency floor; add response caching for
  repeated questions and request-level concurrency limits.
- **Cost & latency.** `gpt-4o-mini` + `text-embedding-3-small` keep per-query cost
  low; `top_k`, chunk size, and model are all env-configurable to trade
  cost/quality. Query rewriting is skipped when there's no history to avoid an
  extra LLM round-trip on the common case.

## 6. Limitations

- No OCR — scanned/image PDFs won't extract text.
- In-memory FAISS is rebuilt wholesale and lives on one node (no HA out of the box).
- Confidence is a useful heuristic, not a calibrated probability.
- Retrieval is dense-only; purely keyword/code-like queries could benefit from
  hybrid search.
- No authentication/authorization or per-document access control yet.

## 7. Future improvements

- **Hybrid search** (BM25 + dense) via an ensemble retriever, plus a
  cross-encoder **re-ranker** for higher precision.
- **Streaming responses** in the API/UI for better perceived latency.
- **Authentication** and per-document ACLs for real enterprise deployment.
- **User feedback collection** (👍/👎) feeding an evaluation/labelling loop.
- **Observability** — tracing (LangSmith), latency/cost metrics, and dashboards.
- **Incremental ingestion** and a managed vector DB for production scale.
