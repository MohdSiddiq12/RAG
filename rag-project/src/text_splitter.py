"""
text_splitter.py
----------------
Splits documents into overlapping chunks for embedding.

chunk_size=800, chunk_overlap=150 is a solid baseline for technical docs.
- Increase chunk_size if your documents have long dense paragraphs.
- Increase chunk_overlap if your answers span chunk boundaries.

No logic changes from original — this was correct.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Separators tried in order: paragraph → line → sentence → word → char.
    This preserves semantic units as much as possible before hard-splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    splits = splitter.split_documents(documents)
    print(f"✅ Split into {len(splits)} chunks  (size={chunk_size}, overlap={chunk_overlap})")
    return splits