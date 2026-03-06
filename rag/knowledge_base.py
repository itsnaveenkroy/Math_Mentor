"""
Loads markdown docs from the knowledge_base/ folder,
chunks them, and indexes into chromadb for retrieval.
"""

import os
import re
import hashlib
import chromadb

from rag.embeddings import get_embedding_function
from utils.config import CHROMA_PERSIST_DIR


_client = None
_collection = None

COLLECTION_NAME = "math_knowledge"
KB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")


def _get_collection():
    """Get or create the chromadb collection. Lazy init."""
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embed_fn = get_embedding_function()
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _chunk_markdown(text, source_file, chunk_size=500, overlap=50):
    """
    Split markdown text into overlapping chunks.
    Tries to break on section headers first, then falls back to
    splitting on double-newlines, then brute-force by size.
    """
    # try splitting on ## headers first
    sections = re.split(r'\n(?=##\s)', text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # break long sections into smaller pieces
            words = section.split()
            current = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > chunk_size and current:
                    chunk_text = " ".join(current)
                    chunks.append(chunk_text)
                    # keep some overlap
                    keep = max(1, len(current) * overlap // chunk_size)
                    current = current[-keep:]
                    current_len = sum(len(w) + 1 for w in current)
                current.append(word)
                current_len += len(word) + 1
            if current:
                chunks.append(" ".join(current))

    return chunks


def _doc_id(source, index):
    """Deterministic ID for a chunk so we don't double-index."""
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def index_knowledge_base(force=False):
    """
    Read all .md files from knowledge_base/ and index them.
    Skips if already indexed (unless force=True).
    """
    collection = _get_collection()

    if not force and collection.count() > 0:
        # already indexed, don't redo it
        return collection.count()

    if not os.path.isdir(KB_DIR):
        return 0

    all_chunks = []
    all_ids = []
    all_metas = []

    for fname in sorted(os.listdir(KB_DIR)):
        if not fname.endswith(".md"):
            continue

        fpath = os.path.join(KB_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        topic = fname.replace(".md", "").replace("_", " ")
        chunks = _chunk_markdown(content, fname)

        for i, chunk in enumerate(chunks):
            doc_id = _doc_id(fname, i)
            all_chunks.append(chunk)
            all_ids.append(doc_id)
            all_metas.append({
                "source": fname,
                "topic": topic,
                "chunk_index": i,
            })

    if all_chunks:
        # chromadb has batch size limits, so chunk it
        batch = 100
        for start in range(0, len(all_chunks), batch):
            end = start + batch
            collection.upsert(
                ids=all_ids[start:end],
                documents=all_chunks[start:end],
                metadatas=all_metas[start:end],
            )

    return len(all_chunks)
