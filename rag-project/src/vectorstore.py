"""
vectorstore.py
--------------
Creates and persists a Chroma vectorstore from document chunks.

Why we wipe before re-creating:
    ChromaDB stores the embedding dimension alongside each collection.
    If you switch embedding models (e.g. 384-dim → 768-dim), upserting
    into the old collection raises InvalidArgumentError.

    When the user explicitly chooses "Re-ingest", they want a clean DB —
    so we delete the old persist directory first.  This is safe and correct.
"""

import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.embedding import get_embeddings

PERSIST_DIR     = "chroma_db"
COLLECTION_NAME = "rag_collection"


def create_vectorstore(
    docs: List[Document],
    persist_directory: str = PERSIST_DIR,
) -> Chroma:
    """
    Embed docs and persist them into a Chroma vectorstore.

    Always wipes the existing directory first so embedding-model switches
    (and therefore dimension changes) never cause a dimension-mismatch error.
    """
    db_path = Path(persist_directory)

    if db_path.exists():
        shutil.rmtree(db_path)
        print(f"🗑️  Cleared old vectorstore at '{persist_directory}'  (fresh start)")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )

    count = vectorstore._collection.count()
    print(f"✅ Vectorstore ready  ({count} vectors stored in '{persist_directory}')")
    return vectorstore