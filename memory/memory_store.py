"""
Memory layer - stores past problems and their solutions so we can:
1. Find similar problems the user already asked (avoid repeat work)
2. Track what topics the user struggles with
3. Store feedback for self-improvement

Uses chromadb for similarity search + a json file for structured data.
"""

import os
import json
import time
import hashlib
import chromadb

from rag.embeddings import get_embedding_function
from utils.config import MEMORY_PERSIST_DIR


MEMORY_COLLECTION = "problem_memory"
FEEDBACK_FILE = os.path.join(MEMORY_PERSIST_DIR, "feedback_log.json")
STATS_FILE = os.path.join(MEMORY_PERSIST_DIR, "topic_stats.json")
CORRECTIONS_FILE = os.path.join(MEMORY_PERSIST_DIR, "corrections_log.json")

_client = None
_collection = None


def _get_memory_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    os.makedirs(MEMORY_PERSIST_DIR, exist_ok=True)
    _client = chromadb.PersistentClient(path=os.path.join(MEMORY_PERSIST_DIR, "chroma"))
    embed_fn = get_embedding_function()
    _collection = _client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def store_problem(problem_text, solution, topic="general", confidence=0.0, feedback=None,
                   context_used=None, verifier_outcome=None):
    """
    Save a solved problem to memory for future reference.
    Stores the full pipeline data: problem, solution, retrieved context, and verification.
    """
    collection = _get_memory_collection()

    doc_id = hashlib.md5(problem_text.encode()).hexdigest()

    metadata = {
        "topic": topic,
        "confidence": confidence,
        "timestamp": time.time(),
        "feedback": feedback or "",
        "verifier_correct": str(verifier_outcome.get("is_correct", "")) if verifier_outcome else "",
        "verifier_confidence": verifier_outcome.get("confidence", 0.0) if verifier_outcome else 0.0,
    }

    # Build full document with context for richer retrieval
    doc_parts = [f"Problem: {problem_text}", f"Solution: {solution}"]
    if context_used:
        sources = [c.get("source", "") for c in context_used[:3] if c.get("source")]
        if sources:
            doc_parts.append(f"Sources: {', '.join(sources)}")
    if verifier_outcome and verifier_outcome.get("issues"):
        doc_parts.append(f"Issues: {'; '.join(verifier_outcome['issues'][:3])}")

    collection.upsert(
        ids=[doc_id],
        documents=["\n\n".join(doc_parts)],
        metadatas=[metadata],
    )

    # update topic stats
    _update_topic_stats(topic, confidence, feedback)

    return doc_id


def find_similar_problems(query, k=3):
    """
    Look for previously solved problems that are similar to the query.
    Returns list of dicts with problem text and metadata.
    """
    collection = _get_memory_collection()

    if collection.count() == 0:
        return []

    actual_k = min(k, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    similar = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity = 1.0 - (dist / 2.0)
        if similarity > 0.6:  # only return actually similar stuff
            similar.append({
                "text": doc,
                "topic": meta.get("topic", ""),
                "confidence": meta.get("confidence", 0),
                "similarity": round(similarity, 3),
            })

    return similar


def store_feedback(problem_text, rating, comment=""):
    """
    Save user feedback on a solution. This feeds the self-learning loop.
    rating: 'helpful', 'wrong', 'confusing', etc.
    """
    os.makedirs(MEMORY_PERSIST_DIR, exist_ok=True)

    feedback_entry = {
        "problem": problem_text[:200],  # truncate for storage
        "rating": rating,
        "comment": comment,
        "timestamp": time.time(),
    }

    entries = _load_json(FEEDBACK_FILE, default=[])
    entries.append(feedback_entry)

    # keep last 500 entries max
    if len(entries) > 500:
        entries = entries[-500:]

    _save_json(FEEDBACK_FILE, entries)
    return True


def _update_topic_stats(topic, confidence, feedback):
    """Track per-topic performance so we know what the user struggles with."""
    stats = _load_json(STATS_FILE, default={})

    if topic not in stats:
        stats[topic] = {"count": 0, "total_confidence": 0.0, "feedback_counts": {}}

    s = stats[topic]
    s["count"] += 1
    s["total_confidence"] += confidence
    if feedback:
        s["feedback_counts"][feedback] = s["feedback_counts"].get(feedback, 0) + 1

    _save_json(STATS_FILE, stats)


def get_topic_stats():
    """Get aggregated stats about which topics have been asked about."""
    stats = _load_json(STATS_FILE, default={})
    result = {}
    for topic, s in stats.items():
        count = s["count"]
        avg_conf = s["total_confidence"] / count if count > 0 else 0
        result[topic] = {
            "count": count,
            "avg_confidence": round(avg_conf, 3),
            "feedback": s.get("feedback_counts", {}),
        }
    return result


def get_recent_feedback(n=10):
    """Get the n most recent feedback entries."""
    entries = _load_json(FEEDBACK_FILE, default=[])
    return entries[-n:]


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default if default is not None else {}


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def store_correction(original_text, corrected_text, source="ocr"):
    """
    Store a user correction (e.g. OCR or audio text that was fixed by the user).
    Used for self-learning: future OCR/audio outputs can be auto-corrected.
    """
    os.makedirs(MEMORY_PERSIST_DIR, exist_ok=True)

    entry = {
        "original": original_text[:300],
        "corrected": corrected_text[:300],
        "source": source,
        "timestamp": time.time(),
    }

    entries = _load_json(CORRECTIONS_FILE, default=[])
    entries.append(entry)

    # keep last 200 corrections max
    if len(entries) > 200:
        entries = entries[-200:]

    _save_json(CORRECTIONS_FILE, entries)
    return True


def get_corrections(source=None, n=50):
    """
    Retrieve past corrections. Optionally filter by source (ocr/audio).
    Used at runtime to apply known correction patterns.
    """
    entries = _load_json(CORRECTIONS_FILE, default=[])
    if source:
        entries = [e for e in entries if e.get("source") == source]
    return entries[-n:]
