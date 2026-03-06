"""
Retriever - queries the chromadb knowledge base for relevant
context chunks given a math problem.
"""

from rag.knowledge_base import _get_collection, index_knowledge_base
from utils.config import MAX_RETRIEVAL_K


def retrieve_context(query, k=None):
    """
    Find the most relevant knowledge base chunks for a given query.
    Returns a list of dicts with text / source / score.
    """
    if k is None:
        k = MAX_RETRIEVAL_K

    # make sure we have something indexed
    index_knowledge_base()

    collection = _get_collection()

    if collection.count() == 0:
        return []

    # clamp k to what's actually in the db
    actual_k = min(k, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    context_chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        # chromadb cosine distance: 0 = identical, 2 = opposite
        # convert to a similarity score (higher = better)
        similarity = 1.0 - (dist / 2.0)

        context_chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "topic": meta.get("topic", ""),
            "similarity": round(similarity, 3),
        })

    return context_chunks


def retrieve_context_text(query, k=None):
    """Convenience: just get the text chunks joined together."""
    chunks = retrieve_context(query, k=k)
    if not chunks:
        return ""
    return "\n\n---\n\n".join(c["text"] for c in chunks)
