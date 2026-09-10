"""
Separated so "which embedding model" and "which vector DB" are decided in
exactly one place — everything else just calls build_vectorstore() or
get_vectorstore() and doesn't care how embeddings actually happen.
"""

import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma


def _get_embeddings():
    # FastEmbed runs on ONNX Runtime (no PyTorch), so it stays light enough
    # for a free-tier server — no API key, no HuggingFace token needed.
    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    return FastEmbedEmbeddings(model_name=model)


def build_vectorstore(chunks, persist_dir: str = None):
    """
    Embed `chunks` locally and persist them to a Chroma vector store on
    disk. Run this once, offline, via run_ingestion.py.
    """
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "vectorstore/chroma_db")
    embeddings = _get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def get_vectorstore(persist_dir: str = None):
    """
    Load an already-persisted Chroma store from disk (no re-embedding).
    """
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "vectorstore/chroma_db")
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(
            f"No vector store found at '{persist_dir}'. "
            "Run `python -m ingestion.run_ingestion` first."
        )
    embeddings = _get_embeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)