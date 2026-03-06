"""
Embedding wrapper. We just use chromadb's built-in embedding function
so we don't need any API key for this part. It uses a small
sentence-transformer model under the hood (onnxruntime).
"""

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


# single instance so we're not loading the model repeatedly
_embed_fn = None

def get_embedding_function():
    """Get chromadb's default embedding function (all-MiniLM-L6-v2 via onnx)."""
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = DefaultEmbeddingFunction()
    return _embed_fn


def embed_texts(texts):
    """
    Embed a list of strings. Returns list of float vectors.
    Mostly here if we need to embed outside of chromadb queries.
    """
    fn = get_embedding_function()
    return fn(texts)
