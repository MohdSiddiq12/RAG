"""
embedding.py
------------
Embedding model factory.

Industry pattern: single-responsibility factory function behind a config constant.
Swap the MODEL_NAME constant to change models project-wide — nothing else changes.

Model upgrade: all-mpnet-base-v2  (420M params, 768-dim)
  vs old:      all-MiniLM-L6-v2   (22M params,  384-dim)

all-mpnet-base-v2 scores ~5-8 points higher on BEIR benchmarks for
semantic similarity tasks, which directly improves retrieval quality on
abstract / comparative queries.
"""

from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"   # stronger semantic model
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # keep as fallback comment


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached-safe embedding model instance."""
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},   # cosine-safe unit vectors
    )
    print(f"✅ Embedding model loaded: {MODEL_NAME.split('/')[-1]}")
    return embeddings