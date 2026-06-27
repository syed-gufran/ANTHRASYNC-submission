"""Document ingestion: load -> chunk -> embed -> index into FAISS.

Run as a module:

    python -m src.ingest

This builds (or rebuilds) the FAISS index under ``settings.vector_path`` from
every supported document in ``settings.data_path``. Keeping ingestion separate
from query time means indexing is a one-off (or scheduled) job, and the API/UI
start up fast by just loading the prebuilt index.
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

# Map file extensions to the loader that handles them. Add more here to extend
# format support (e.g. ".docx" via Docx2txtLoader).
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}


def get_embeddings() -> OpenAIEmbeddings:
    """Single source of truth for the embedding model (used here and at query time)."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def _load_file(path: Path) -> list[Document]:
    """Load one file into LangChain Documents, attaching clean source metadata."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # PyPDFLoader yields one Document per page and sets metadata["page"]
        # (0-indexed), which we normalise to 1-indexed for human-friendly citations.
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            page = doc.metadata.get("page")
            if isinstance(page, int):
                doc.metadata["page"] = page + 1
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    # Normalise the source to just the filename — that's what we cite.
    for doc in docs:
        doc.metadata["source"] = path.name
    return docs


def load_documents() -> list[Document]:
    """Load every supported document from the data directory."""
    data_path = settings.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    documents: list[Document] = []
    for path in sorted(data_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(_load_file(path))

    if not documents:
        raise ValueError(
            f"No supported documents found in {data_path}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer splitting on structure (paragraphs, lines) before characters,
        # so chunks stay semantically coherent.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_index() -> FAISS:
    """End-to-end ingestion: load, chunk, embed, persist the FAISS index."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    print(f"📂 Loading documents from {settings.data_path} ...")
    documents = load_documents()
    print(f"   Loaded {len(documents)} document section(s).")

    print("✂️  Chunking ...")
    chunks = chunk_documents(documents)
    print(f"   Produced {len(chunks)} chunks "
          f"(size={settings.chunk_size}, overlap={settings.chunk_overlap}).")

    print(f"🧠 Embedding with '{settings.embedding_model}' and building FAISS index ...")
    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    settings.vector_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(settings.vector_path))
    print(f"✅ Index saved to {settings.vector_path}")
    return vectorstore


def load_index() -> FAISS:
    """Load a previously built FAISS index from disk."""
    if not settings.vector_path.exists():
        raise FileNotFoundError(
            f"No FAISS index at {settings.vector_path}. "
            f"Run `python -m src.ingest` first."
        )
    return FAISS.load_local(
        str(settings.vector_path),
        get_embeddings(),
        # Safe here: we created this index ourselves. Only set True for trusted files.
        allow_dangerous_deserialization=True,
    )


if __name__ == "__main__":
    try:
        build_index()
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard
        print(f"❌ Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)
