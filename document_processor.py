"""
Multi-format document processor.
Handles loading and chunking for: PDF, TXT, DOCX, Markdown, CSV, HTML.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings, SUPPORTED_FILE_EXTENSIONS


def _get_loader(file_path: str):
    """Return the appropriate LangChain loader for a file extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")

    elif ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path)

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path)

    elif ext == ".md":
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        return UnstructuredMarkdownLoader(file_path)

    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(file_path, encoding="utf-8")

    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        return UnstructuredHTMLLoader(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}"
        )


def load_and_chunk(
    file_path: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Load a document from disk and split it into chunks.

    Args:
        file_path: Absolute path to the document.
        chunk_size: Override for chunk size (defaults to settings.chunk_size).
        chunk_overlap: Override for overlap (defaults to settings.chunk_overlap).

    Returns:
        List of Document chunks with metadata including source filename.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}"
        )

    loader = _get_loader(file_path)
    raw_documents = loader.load()

    # Add source filename to metadata
    filename = Path(file_path).name
    for doc in raw_documents:
        doc.metadata["source_file"] = filename

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(raw_documents)
    return chunks


def load_multiple(file_paths: List[str]) -> List[Document]:
    """
    Load and chunk multiple documents.

    Args:
        file_paths: List of file paths to process.

    Returns:
        Combined list of Document chunks from all files.
    """
    all_chunks: List[Document] = []
    for path in file_paths:
        chunks = load_and_chunk(path)
        all_chunks.extend(chunks)
    return all_chunks
