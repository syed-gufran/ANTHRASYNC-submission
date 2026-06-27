# 📚 Enterprise Knowledge Assistant

A **Retrieval-Augmented Generation (RAG)** system that answers
natural-language questions about a company's internal documents — and **cites its
sources**. Built with **LangChain + LangGraph**, **OpenAI**, and **FAISS**.

> Ask *"What is the employee leave policy?"* → get *"Employees are eligible for
> 24 paid leaves annually… [1]"* with a citation to `HR_Policy.md`.

---

##  What it does

- **Document ingestion** — loads PDF / Markdown / text files, chunks them, embeds
  them, and stores them in a searchable FAISS index.
- **Semantic search** — finds the most relevant passages for a question.
- **Grounded answers** — the LLM answers **only** from retrieved context.
- **Source citations** — every answer lists the documents (and PDF page numbers)
  it relied on.
- **Hallucination guard** — if the answer isn't in the documents, it says so
  instead of making something up.
- **Three interfaces** — a Streamlit chat UI, a FastAPI `/ask` endpoint, and a CLI.

### Bonus features included
Conversation memory (follow-up questions) · query rewriting · confidence scoring ·
multi-document reasoning · an automated evaluation harness.

---

## 🏗️ Architecture

```
                          INGESTION (offline, one-off)
   data/*.md,*.pdf ──► load ──► chunk ──► embed (OpenAI) ──► FAISS index (.faiss/)

                          QUERY TIME (LangGraph pipeline)
   question ─►┌───────────────┐   ┌──────────┐   ┌───────────────────────────┐
              │ rewrite_query │──►│ retrieve │──►│         generate          │──► answer
              │ (follow-ups)  │   │ (FAISS,  │   │ grounded answer + [n]      │    + sources
              └───────────────┘   │  top-k)  │   │ citations + confidence,    │    + confidence
                                  └──────────┘   │ "I don't know" guard       │
                                                 └───────────────────────────┘
```

See [`docs/system_design.md`](docs/system_design.md) for the full design document.

---

## 🚀 Quickstart

### 1. Prerequisites
- Python **3.10 – 3.12** recommended.
- An **OpenAI API key**.

### 2. Install
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
```

### 4. Build the index
```bash
python -m src.ingest
```
This reads everything in `data/`, chunks and embeds it, and writes the FAISS
index to `.faiss/`. Re-run it whenever you add or change documents.

### 5. Run an interface

**Streamlit chat UI**
```bash
streamlit run src/app.py
```

**REST API**
```bash
uvicorn src.api:app --reload
```

**One-off CLI check**
```bash
python -m src.rag_graph   
```

---

## 🔌 API

```http
POST /ask
Content-Type: application/json

{ "question": "What is the refund policy?" }
```

```json
{
  "answer": "Customers can request a full refund within 30 days of purchase... [1]",
  "sources": [{ "document": "Customer_Refund_Policy.md", "page": null }],
  "confidence": 0.78
}
```

`curl` example:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many paid leaves do employees get?"}'
```

---

## 🧪 Evaluation

```bash
python -m evaluation.evaluate
```

Runs `evaluation/test_cases.json` through the pipeline and reports **answer
accuracy**, **citation accuracy**, and **hallucination avoidance** (out-of-scope
questions that should be refused). See the
[evaluation approach](docs/system_design.md#evaluation-approach) for details and
how it would extend to an LLM-judge.

---

## ⚙️ Configuration

All knobs live in `.env` (defaults in `.env.example`):

| Variable          | Default                  | Purpose                                  |
|-------------------|--------------------------|------------------------------------------|
| `OPENAI_API_KEY`  | —                        | **Required.** Used for LLM + embeddings. |
| `LLM_MODEL`       | `gpt-4o-mini`            | Chat model for answer generation.        |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for indexing/search.     |
| `CHUNK_SIZE`      | `1000`                   | Characters per chunk.                    |
| `CHUNK_OVERLAP`   | `150`                    | Overlap between chunks.                  |
| `TOP_K`           | `4`                      | Chunks retrieved per question.           |
| `TEMPERATURE`     | `0.0`                    | Lower = more deterministic answers.      |

---

## 📁 Project structure

```
.
├── data/                       # Sample enterprise documents (the knowledge base)
├── src/
│   ├── config.py               # Typed settings from .env
│   ├── ingest.py               # Load → chunk → embed → FAISS index
│   ├── rag_graph.py            # LangGraph RAG pipeline (the core)
│   ├── api.py                  # FastAPI /ask endpoint
│   └── app.py                  # Streamlit chat UI
├── evaluation/
│   ├── test_cases.json         # Question set with expected answers/sources
│   └── evaluate.py             # Automated scoring harness
├── docs/
│   └── system_design.md        # 1–2 page system design document
├── requirements.txt
└── .env.example
```

---

## 🧠 Technology choices (short version)

| Concern        | Choice                  | Why                                                       |
|----------------|-------------------------|-----------------------------------------------------------|
| Orchestration  | LangGraph               | Explicit, testable, extensible graph of RAG steps.        |
| LLM            | OpenAI `gpt-4o-mini`    | Strong instruction-following + structured output, low cost.|
| Embeddings     | `text-embedding-3-small`| Cheap, fast, high-quality; same provider as the LLM.       |
| Vector store   | FAISS                   | Fast, local, zero-infra; ideal for this scale.            |
| API / UI       | FastAPI / Streamlit     | Minimal, well-documented, fast to demo.                   |

Full rationale, limitations, and future work are in
[`docs/system_design.md`](docs/system_design.md).

---

## 📌 Assumptions

- The document set is small-to-medium (hundreds of docs) — FAISS in-memory is a
  good fit. For larger corpora, swap in a managed vector DB (see future work).
- Documents are text-extractable (no OCR step for scanned PDFs).
- Sample documents under `data/` are fictional and for demonstration only.
- No secrets/keys are committed; `.env` is git-ignored.
