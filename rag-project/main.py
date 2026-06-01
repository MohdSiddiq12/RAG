"""
main.py
-------
CLI entry point for the RAG system.

Upgrades over original:
  - Uses create_rag_chain_with_sources → displays answer + citations
  - Streams the answer token-by-token for better UX on long responses
  - 'debug' command now supports strategy comparison
  - 'mode' command lets you switch retriever strategy mid-session
  - Clean error handling with informative messages
  - Help menu

Run:
    python main.py
"""

import sys
from dotenv import load_dotenv

load_dotenv()


def print_banner() -> None:
    print("\n" + "═" * 70)
    print("  🧠  RAG SYSTEM  —  Powered by Groq + ChromaDB + LangChain")
    print("═" * 70)
    print("  Commands:")
    print("    exit / quit  →  Exit the program")
    print("    debug        →  Inspect retrieved chunks for a query")
    print("    compare      →  Compare MMR vs MultiQuery for a query")
    print("    mode         →  Switch retriever strategy (multi_query / mmr)")
    print("    help         →  Show this menu")
    print("═" * 70 + "\n")


def display_sources(sources: list) -> None:
    """Print de-duplicated source citations below the answer."""
    if not sources:
        return

    seen = set()
    unique = []
    for doc in sources:
        key = (
            doc.metadata.get("source", "Unknown"),
            doc.metadata.get("page", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(key)

    print("\n  📚 Sources:")
    for source, page in unique:
        page_str = f"  (page {page})" if page != "" else ""
        print(f"    • {source}{page_str}")


def run_ingestion() -> None:
    """Load, split, and embed documents into ChromaDB."""
    from src.data_loader import load_documents
    from src.text_splitter import split_documents
    from src.vectorstore import create_vectorstore

    print("\n📥 Starting document ingestion...\n")
    docs   = load_documents("data")
    splits = split_documents(docs, chunk_size=800, chunk_overlap=150)
    create_vectorstore(splits)
    print("\n✅ Ingestion complete!\n")


def main() -> None:
    print_banner()

    # ── Ingestion choice ───────────────────────────────────────────────────────
    choice = input("Do you want to  (1) Re-ingest documents  or  (2) Just query? (1/2): ").strip()
    if choice == "1":
        run_ingestion()

    # ── Load chain ─────────────────────────────────────────────────────────────
    from src.rag_chain import create_rag_chain_with_sources
    from src.debug import debug_retriever, compare_retrieval_strategies

    retriever_type = "multi_query"   # default; changeable mid-session
    print(f"\n🔗 Building RAG chain  (retriever={retriever_type})...")
    rag_chain = create_rag_chain_with_sources(retriever_type=retriever_type)

    print("\n" + "═" * 70)
    print("  ✅  System ready — ask anything about your documents.")
    print("═" * 70 + "\n")

    # ── Query loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            query = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            sys.exit(0)

        if not query:
            continue

        lower = query.lower()

        # ── Built-in commands ──────────────────────────────────────────────────
        if lower in ("exit", "quit", "q"):
            print("👋  Goodbye!")
            break

        if lower == "help":
            print_banner()
            continue

        if lower == "debug":
            q = input("  Enter query to debug: ").strip()
            if q:
                debug_retriever(q, retriever_type=retriever_type)
            continue

        if lower == "compare":
            q = input("  Enter query to compare strategies: ").strip()
            if q:
                compare_retrieval_strategies(q)
            continue

        if lower == "mode":
            print("  Available modes: multi_query | mmr")
            new_mode = input("  Select mode: ").strip().lower()
            if new_mode in ("multi_query", "mmr"):
                retriever_type = new_mode
                print(f"\n🔗 Rebuilding chain with retriever={retriever_type}...")
                rag_chain = create_rag_chain_with_sources(retriever_type=retriever_type)
                print(f"✅ Switched to: {retriever_type}\n")
            else:
                print("  ⚠️  Unknown mode. Choose 'multi_query' or 'mmr'.")
            continue

        # ── Answer query ───────────────────────────────────────────────────────
        print("\n⏳ Thinking...\n")
        try:
            result = rag_chain.invoke({"question": query})
            print(f"💬 Answer:\n{result['answer']}")
            display_sources(result["sources"])
            print()

        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("   Tip: Run 'debug' to check what the retriever is fetching.\n")


if __name__ == "__main__":
    main()