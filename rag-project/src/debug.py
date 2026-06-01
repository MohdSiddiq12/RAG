"""
debug.py
--------
Retrieval diagnostics toolkit.

Industry pattern: debug utilities are first-class code, not afterthoughts.
These functions let you inspect exactly what the retriever sees for any
query — essential for diagnosing retrieval failures before blaming the LLM.

Usage from main.py:
    debug_retriever("What are transformers?")
    debug_retriever("Compare attention mechanisms", retriever_type="mmr")
    compare_retrieval_strategies("What are the advantages of transformers?")
"""

from typing import Optional
from src.retriever import get_retriever, get_base_retriever, get_multi_query_retriever


def debug_retriever(
    query: str,
    k: int = 6,
    retriever_type: str = "multi_query",
    content_preview_len: int = 500,
) -> None:
    """
    Print detailed info about what the retriever fetches for a query.

    Args:
        query:               The question to test.
        k:                   Number of chunks to retrieve.
        retriever_type:      "multi_query" | "mmr"
        content_preview_len: How many chars of each chunk to show.
    """
    print(f"\n{'─'*70}")
    print(f"🔍 DEBUG  |  retriever={retriever_type}  |  k={k}")
    print(f"Query: {query}")
    print(f"{'─'*70}")

    retriever = get_retriever(retriever_type=retriever_type, k=k)
    docs = retriever.invoke(query)

    print(f"\n📄 Retrieved {len(docs)} chunks\n")

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "N/A")
        print(f"  ┌─ Chunk {i} ─────────────────────────────────────")
        print(f"  │  Source : {source}")
        print(f"  │  Page   : {page}")
        print(f"  │  Length : {len(doc.page_content)} chars")
        print(f"  │  Preview:")
        # indent content preview
        preview = doc.page_content[:content_preview_len].replace("\n", "\n  │    ")
        print(f"  │    {preview}{'...' if len(doc.page_content) > content_preview_len else ''}")
        print(f"  └{'─'*52}\n")


def compare_retrieval_strategies(query: str, k: int = 6) -> None:
    """
    Run the same query through both MMR and MultiQuery retrievers
    and compare what each strategy fetches.

    Useful for deciding which strategy works better for a given query type.
    """
    print(f"\n{'═'*70}")
    print(f"⚖️  STRATEGY COMPARISON")
    print(f"Query: {query}")
    print(f"{'═'*70}")

    for strategy in ["mmr", "multi_query"]:
        retriever = get_retriever(retriever_type=strategy, k=k)
        docs = retriever.invoke(query)
        sources = [
            f"{d.metadata.get('source','?')} p{d.metadata.get('page','')}"
            for d in docs
        ]
        unique_sources = list(dict.fromkeys(sources))   # preserve order, dedupe

        print(f"\n  [{strategy.upper()}]  → {len(docs)} chunks from {len(unique_sources)} unique sources")
        for s in unique_sources:
            print(f"    • {s}")

    print(f"\n{'═'*70}\n")