"""
Separated so "which embedding model" and "which vector DB" are decided in
exactly one place — everything else just calls build_vectorstore() or
get_vectorstore() and doesn't care how embeddings actually happen.
"""

import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma


_embeddings = None  # cached after first load, reused for every later request


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        _embeddings = FastEmbedEmbeddings(model_name=model)
    return _embeddings

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