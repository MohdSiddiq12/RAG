"""
retriever.py
------------
Advanced retriever with two industry-standard strategies:

1. MMR (Maximal Marginal Relevance)
   Fetches `fetch_k` candidates then re-ranks to maximise both relevance
   AND diversity.  Fixes the "4 near-duplicate chunks" problem.

2. MultiQueryRetriever
   Uses the LLM to auto-generate N paraphrased versions of the user's
   question, runs each through MMR retrieval, then deduplicates results.
   Fixes abstract / comparative queries that don't semantically match any
   single chunk.

Industry pattern:
  - Config constants at the top (one place to tune)
  - Factory functions (not classes) for stateless retriever creation
  - Retriever type selected via a string flag → easy to swap in tests / CLI
"""

import logging
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever

from src.embedding import get_embeddings
from src.llm import get_llm

# ── Config ─────────────────────────────────────────────────────────────────────
PERSIST_DIR      = "chroma_db"
COLLECTION_NAME  = "rag_collection"
DEFAULT_K        = 6      # final chunks returned to the chain
FETCH_K          = 20     # MMR candidate pool before diversity re-rank
LAMBDA_MULT      = 0.5    # MMR λ: 0 = max diversity, 1 = max relevance
N_QUERY_VARIANTS = 3      # how many query paraphrases MultiQuery generates

# Suppress the verbose LangChain multi-query log lines in production
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)


def _load_vectorstore(persist_directory: str = PERSIST_DIR) -> Chroma:
    """Load existing Chroma vectorstore from disk."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def get_base_retriever(
    persist_directory: str = PERSIST_DIR,
    k: int = DEFAULT_K,
    fetch_k: int = FETCH_K,
    lambda_mult: float = LAMBDA_MULT,
):
    """
    MMR retriever — diverse, high-relevance chunks.
    Good for: factual questions, concept explanations.
    """
    vectorstore = _load_vectorstore(persist_directory)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )
    print(f"✅ MMR Retriever ready  (k={k}, fetch_k={fetch_k}, λ={lambda_mult})")
    return retriever


def get_multi_query_retriever(
    persist_directory: str = PERSIST_DIR,
    k: int = DEFAULT_K,
    fetch_k: int = FETCH_K,
    lambda_mult: float = LAMBDA_MULT,
):
    """
    MultiQuery + MMR retriever — best for abstract, comparative, or
    broad questions that don't map cleanly to a single chunk.

    Pipeline:
        user question
            → LLM generates N paraphrases
            → each paraphrase hits MMR retriever
            → results are deduplicated
            → merged unique chunk set returned
    """
    base_retriever = get_base_retriever(persist_directory, k, fetch_k, lambda_mult)
    llm = get_llm()

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        # prompt is auto-generated; LangChain instructs the LLM to produce
        # N_QUERY_VARIANTS alternative phrasings of the question
    )
    print(
        f"✅ MultiQuery Retriever ready  "
        f"({N_QUERY_VARIANTS} query variants × MMR k={k})"
    )
    return multi_retriever


def get_retriever(
    retriever_type: str = "multi_query",
    persist_directory: str = PERSIST_DIR,
    k: int = DEFAULT_K,
):
    """
    Public factory — select retriever strategy via string flag.

    Args:
        retriever_type: "multi_query" (default) | "mmr"
        persist_directory: path to Chroma DB
        k: number of final chunks to return
    """
    if retriever_type == "multi_query":
        return get_multi_query_retriever(persist_directory=persist_directory, k=k)
    elif retriever_type == "mmr":
        return get_base_retriever(persist_directory=persist_directory, k=k)
    else:
        raise ValueError(
            f"Unknown retriever_type='{retriever_type}'. "
            "Choose 'multi_query' or 'mmr'."
        )