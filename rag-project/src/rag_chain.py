"""
rag_chain.py
------------
Production RAG chain using LangChain Expression Language (LCEL).

Key upgrades over the original:

1. SYSTEM + HUMAN message split (ChatPromptTemplate with roles)
   The original used a single flat string. Splitting into system/human
   gives the LLM clearer instruction boundaries and better follows
   chat-tuned models' expected format.

2. Stronger prompt engineering
   - Explicit synthesis instruction for cross-document questions
   - Numbered source attribution so the LLM cites evidence
   - Graceful degradation: partial info → use it, don't bail
   - "Think step by step" nudge for complex multi-hop queries

3. Source-aware output
   Returns both the answer text AND the source documents so callers
   can display citations.  Industry pattern: chains return structured
   dicts, not raw strings, whenever downstream consumers need metadata.

4. Two chain variants
   - create_rag_chain()         → simple str output (quick use)
   - create_rag_chain_with_sources() → dict {answer, sources} (full info)

LCEL pipeline shape:
    {context_docs, question}
        → prompt
        → llm
        → output parser
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from typing import List, Dict, Any

from src.llm import get_llm
from src.retriever import get_retriever


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert research assistant with deep analytical ability.

You will be given numbered source excerpts from one or more documents, followed by a question.

Your job:
1. Read ALL provided sources carefully before answering.
2. Synthesise information ACROSS sources when the question is comparative or broad.
3. For concept questions: explain clearly with definitions and examples from the sources.
4. For comparison questions: explicitly contrast what each source says.
5. For extraction questions: list all relevant concepts found across the sources.
6. If partial information exists, use it and note what is missing.
7. Cite sources by number (e.g. [Source 1]) when making specific claims.
8. Only say "insufficient information" if the context contains NOTHING relevant.
9. Think step by step for complex questions before giving the final answer.

Do NOT fabricate information. Ground every claim in the provided sources."""

HUMAN_PROMPT = """SOURCES:
{context}

QUESTION: {question}

ANSWER:"""


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
    ])


# ── Context Formatters ─────────────────────────────────────────────────────────

def format_docs_simple(docs: List[Document]) -> str:
    """Plain text join — used by the simple chain."""
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Numbered, source-labelled context block.
    Gives the LLM explicit [Source N] handles to cite in its answer.

    Example output:
        [Source 1] (transformers.pdf, page 3)
        The transformer architecture replaces recurrence with self-attention...

        [Source 2] (rnns.pdf, page 1)
        RNNs process sequences token by token, preventing parallelisation...
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "")
        header = f"[Source {i}] ({source}" + (f", page {page})" if page else ")")
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)


# ── Chain Factories ────────────────────────────────────────────────────────────

def create_rag_chain(retriever_type: str = "multi_query"):
    """
    Simple RAG chain → returns answer string.
    Good for: CLI use, quick prototyping.

    Args:
        retriever_type: "multi_query" (default) | "mmr"
    """
    llm       = get_llm()
    retriever = get_retriever(retriever_type=retriever_type)
    prompt    = _build_prompt()

    chain = (
        {
            "context":  retriever | RunnableLambda(format_docs_with_sources),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG Chain ready (simple output)")
    return chain


def create_rag_chain_with_sources(retriever_type: str = "multi_query"):
    """
    RAG chain → returns dict: {"answer": str, "sources": List[Document]}

    Industry pattern: structured output lets the caller render citations,
    log retrieved chunks, or run evaluation metrics without re-invoking.

    Args:
        retriever_type: "multi_query" (default) | "mmr"
    """
    llm       = get_llm()
    retriever = get_retriever(retriever_type=retriever_type)
    prompt    = _build_prompt()

    # Step 1: retrieve docs and pass them forward in the pipeline
    retrieve_step = RunnablePassthrough.assign(
        docs=RunnableLambda(lambda x: retriever.invoke(x["question"]))
    )

    # Step 2: format context from retrieved docs
    format_step = RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: format_docs_with_sources(x["docs"]))
    )

    # Step 3: generate answer
    answer_step = RunnablePassthrough.assign(
        answer=RunnableLambda(
            lambda x: (prompt | llm | StrOutputParser()).invoke(
                {"context": x["context"], "question": x["question"]}
            )
        )
    )

    # Step 4: return clean output dict
    output_step = RunnableLambda(
        lambda x: {
            "answer":  x["answer"],
            "sources": x["docs"],
        }
    )

    chain = retrieve_step | format_step | answer_step | output_step

    print("✅ RAG Chain ready (answer + sources output)")
    return chain