"""
data_loader.py
--------------
Loads PDF, TXT, and DOCX files from a directory into LangChain Documents.

No changes to business logic from original — this was working correctly.
Added type hints and docstring for clarity.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


def load_documents(data_dir: str = "data") -> List[Document]:
    """
    Recursively load all supported documents from data_dir.

    Supported formats: .pdf, .txt, .docx
    Returns a flat list of LangChain Document objects.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: '{data_dir}'. Create it and add your documents.")

    documents: List[Document] = []

    for pdf_file in sorted(data_path.glob("**/*.pdf")):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        print(f"  📄 {pdf_file.name}  →  {len(docs)} pages")
        documents.extend(docs)

    for txt_file in sorted(data_path.glob("**/*.txt")):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs = loader.load()
        print(f"  📄 {txt_file.name}  →  {len(docs)} doc(s)")
        documents.extend(docs)

    for docx_file in sorted(data_path.glob("**/*.docx")):
        loader = Docx2txtLoader(str(docx_file))
        docs = loader.load()
        print(f"  📄 {docx_file.name}  →  {len(docs)} doc(s)")
        documents.extend(docs)

    if not documents:
        raise ValueError(f"No documents found in '{data_dir}'. Add .pdf/.txt/.docx files.")

    print(f"\n✅ Total loaded: {len(documents)} document chunks")
    return documents