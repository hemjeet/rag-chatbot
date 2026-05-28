"""
Core RAG Engine — manages the vector store, retrievers, and query pipeline.
Designed to be initialized once at app startup via FastAPI lifespan.

Compatible with langchain >= 1.3 (uses LCEL Runnables, no deprecated chains).
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import AsyncIterator, List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import settings, LLM_PROVIDERS
from document_processor import load_and_chunk, load_multiple
from memory import memory_manager

logger = logging.getLogger("rag_engine")


# ──────────────────────────────────────────────
# System Prompt Template
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for a RAG (Retrieval-Augmented Generation) system.

If the user asks a factual question, answer it based ONLY on the following context.
If the context doesn't contain the answer, say "I cannot find this information in the uploaded documents."

You may, however, respond naturally to casual greetings (like "hi", "hello", "how are you").

Instructions:
1. Answer directly and concisely
2. Use only information from the context for factual questions
3. If unsure about a fact, say you don't know
4. Do not make up information or facts outside the context
5. When relevant, mention which source document the information comes from

{conversation_history_block}

Context:
{context}"""


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


class _EnsembleRetriever:
    """
    A simple ensemble retriever that merges results from FAISS and BM25.
    Replaces the removed langchain EnsembleRetriever.
    """

    def __init__(self, faiss_retriever, bm25_retriever, weights=(0.4, 0.6)):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.bm25_weight = weights[0]
        self.faiss_weight = weights[1]

    def invoke(self, query: str, **kwargs) -> List[Document]:
        """Retrieve and merge documents using reciprocal rank fusion."""
        bm25_docs = self.bm25_retriever.invoke(query)
        faiss_docs = self.faiss_retriever.invoke(query)

        # Reciprocal Rank Fusion
        rrf_k = 60
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:200]
            doc_scores[key] = doc_scores.get(key, 0) + self.bm25_weight / (rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(faiss_docs):
            key = doc.page_content[:200]
            doc_scores[key] = doc_scores.get(key, 0) + self.faiss_weight / (rrf_k + rank + 1)
            doc_map[key] = doc

        # Sort by score descending
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:settings.retriever_k]]

    async def ainvoke(self, query: str, **kwargs) -> List[Document]:
        """Async version — falls back to sync since BM25 is sync-only."""
        return self.invoke(query, **kwargs)


class RAGEngine:
    """
    Manages the full RAG lifecycle: document ingestion, indexing, retrieval, and querying.
    Intended to be created once during FastAPI lifespan startup.
    """

    def __init__(self):
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._llm: Optional[ChatOpenAI] = None
        self._faiss_store: Optional[FAISS] = None
        self._documents: List[Document] = []  # All chunks for BM25
        self._indexed_files: List[str] = []   # Track which files have been indexed
        self._current_provider: str = settings.default_llm_provider
        self._current_model: str = settings.default_model
        self._current_embedding_model: str = settings.default_embedding_model
        self._current_temperature: float = settings.default_temperature
        self._is_ready: bool = False

    # ──────────────────────────────────────────
    # Initialization (called from lifespan)
    # ──────────────────────────────────────────

    def initialize(self):
        """
        Initialize embeddings, LLM, and load existing vector store if available.
        Called once during FastAPI lifespan startup.
        """
        logger.info("🚀 Initializing RAG Engine...")
        settings.ensure_dirs()

        # Initialize embeddings (always OpenAI)
        self._embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=self._current_embedding_model,
        )

        # Initialize LLM
        self._init_llm()

        # Try to load existing FAISS index
        faiss_index_path = os.path.join(settings.vector_store_path, "index.faiss")
        bm25_docs_path = os.path.join(settings.vector_store_path, "bm25_docs.json")

        if os.path.exists(faiss_index_path):
            logger.info("📂 Loading existing FAISS vector store...")
            self._faiss_store = FAISS.load_local(
                settings.vector_store_path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )

            # Load BM25 documents
            if os.path.exists(bm25_docs_path):
                with open(bm25_docs_path, "r", encoding="utf-8") as f:
                    docs_data = json.load(f)
                    self._documents = [Document(**doc) for doc in docs_data]
                logger.info(f"📄 Loaded {len(self._documents)} document chunks from BM25 index")

            # Load indexed file list
            files_path = os.path.join(settings.vector_store_path, "indexed_files.json")
            if os.path.exists(files_path):
                with open(files_path, "r", encoding="utf-8") as f:
                    self._indexed_files = json.load(f)

            self._is_ready = True
            logger.info("✅ RAG Engine ready with existing index")
        else:
            logger.info("📭 No existing index found. Upload documents to get started.")

        return self

    def _init_llm(self):
        """Initialize or re-initialize the LLM based on current settings."""
        provider_config = LLM_PROVIDERS.get(self._current_provider, LLM_PROVIDERS["openai"])
        api_key = settings.get_api_key(self._current_provider)

        kwargs = {
            "model": self._current_model,
            "temperature": self._current_temperature,
            "api_key": api_key,
            "streaming": True,
        }

        if provider_config["base_url"]:
            kwargs["base_url"] = provider_config["base_url"]

        self._llm = ChatOpenAI(**kwargs)
        logger.info(f"🤖 LLM initialized: {self._current_provider}/{self._current_model}")

    # ──────────────────────────────────────────
    # Configuration Updates
    # ──────────────────────────────────────────

    def update_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: Optional[float] = None,
        openai_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update runtime configuration. Returns the updated config."""
        changed = []

        # Update API keys in settings if provided
        if openai_api_key is not None:
            settings.openai_api_key = openai_api_key
            changed.append("openai_api_key")
        if deepseek_api_key is not None:
            settings.deepseek_api_key = deepseek_api_key
            changed.append("deepseek_api_key")
        if nvidia_api_key is not None:
            settings.nvidia_api_key = nvidia_api_key
            changed.append("nvidia_api_key")

        # Update LLM config
        llm_changed = False
        if provider is not None and provider != self._current_provider:
            self._current_provider = provider
            changed.append("provider")
            llm_changed = True
        if model is not None and model != self._current_model:
            self._current_model = model
            changed.append("model")
            llm_changed = True
        if temperature is not None and temperature != self._current_temperature:
            self._current_temperature = temperature
            changed.append("temperature")
            llm_changed = True

        if llm_changed or "openai_api_key" in changed or "deepseek_api_key" in changed:
            self._init_llm()

        # Update embedding model (requires re-indexing if changed)
        if embedding_model is not None and embedding_model != self._current_embedding_model:
            self._current_embedding_model = embedding_model
            self._embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=self._current_embedding_model,
            )
            changed.append("embedding_model")
            logger.warning("⚠️ Embedding model changed. Re-index documents for best results.")

        return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            "provider": self._current_provider,
            "model": self._current_model,
            "embedding_model": self._current_embedding_model,
            "temperature": self._current_temperature,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retriever_k": settings.retriever_k,
            "is_ready": self._is_ready,
            "indexed_files": self._indexed_files,
            "total_chunks": len(self._documents),
            "has_openai_key": bool(settings.openai_api_key),
            "has_deepseek_key": bool(settings.deepseek_api_key),
            "has_nvidia_key": bool(settings.nvidia_api_key),
        }

    # ──────────────────────────────────────────
    # Document Ingestion
    # ──────────────────────────────────────────

    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process and index one or more documents.
        Adds to the existing index (does not replace it).
        """
        logger.info(f"📥 Ingesting {len(file_paths)} document(s)...")

        # Load and chunk all files
        new_chunks = load_multiple(file_paths)
        if not new_chunks:
            raise ValueError("No content could be extracted from the uploaded files.")

        # Track new filenames
        new_filenames = [Path(p).name for p in file_paths]

        # Add to BM25 document list
        self._documents.extend(new_chunks)

        # Build or extend FAISS index
        if self._faiss_store is None:
            self._faiss_store = FAISS.from_documents(new_chunks, self._embeddings)
        else:
            new_store = FAISS.from_documents(new_chunks, self._embeddings)
            self._faiss_store.merge_from(new_store)

        # Save everything to disk
        self._faiss_store.save_local(settings.vector_store_path)
        self._save_bm25_docs()
        self._indexed_files.extend(new_filenames)
        self._save_indexed_files()

        self._is_ready = True

        result = {
            "files_processed": new_filenames,
            "new_chunks": len(new_chunks),
            "total_chunks": len(self._documents),
            "total_files": len(self._indexed_files),
        }
        logger.info(f"✅ Ingestion complete: {result}")
        return result

    def _save_bm25_docs(self):
        """Persist BM25 document chunks to disk."""
        docs_data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self._documents
        ]
        path = os.path.join(settings.vector_store_path, "bm25_docs.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

    def _save_indexed_files(self):
        """Persist indexed file list to disk."""
        path = os.path.join(settings.vector_store_path, "indexed_files.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._indexed_files, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────
    # Retriever Setup
    # ──────────────────────────────────────────

    def _build_ensemble_retriever(self) -> _EnsembleRetriever:
        """Build the FAISS + BM25 ensemble retriever."""
        if self._faiss_store is None or not self._documents:
            raise RuntimeError("No documents indexed. Upload documents first.")

        faiss_retriever = self._faiss_store.as_retriever(
            search_kwargs={"k": settings.retriever_k}
        )
        bm25_retriever = BM25Retriever.from_documents(
            self._documents, k=settings.retriever_k
        )

        return _EnsembleRetriever(
            faiss_retriever=faiss_retriever,
            bm25_retriever=bm25_retriever,
            weights=(settings.bm25_weight, settings.faiss_weight),
        )

    def _rerank_documents(self, docs: List[Document], query: str) -> List[Document]:
        """Rerank documents using NVIDIA reranker."""
        if not settings.nvidia_api_key:
            logger.warning("⚠️ NVIDIA API key not set, skipping reranking")
            return docs

        try:
            reranker = NVIDIARerank(
                model="nv-rerank-qa-mistral-4b:1",
                api_key=settings.nvidia_api_key,
            )
            reranked = reranker.compress_documents(docs, query)
            return list(reranked)
        except Exception as e:
            logger.warning(f"⚠️ Reranking failed, using original order: {e}")
            return docs

    # ──────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────

    def _build_prompt(self, session_id: Optional[str] = None) -> ChatPromptTemplate:
        """Build the RAG prompt, optionally injecting conversation history."""
        history_block = ""
        if session_id:
            session = memory_manager.get_session(session_id)
            if session:
                history_text = session.get_history_as_text()
                if history_text:
                    history_block = (
                        f"\nPrevious conversation:\n{history_text}\n\n"
                        "Use the conversation above for context when answering follow-up questions."
                    )

        system_content = SYSTEM_PROMPT.replace(
            "{conversation_history_block}", history_block
        )

        return ChatPromptTemplate.from_messages([
            ("system", system_content),
            ("human", "{input}"),
        ])

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        rerank: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronous query — returns the full answer at once.
        Uses LCEL Runnables (no deprecated chains).
        """
        if not self._is_ready:
            raise RuntimeError("No documents indexed. Upload documents first.")

        # Retrieve documents
        retriever = self._build_ensemble_retriever()
        docs = retriever.invoke(question)

        # Optional reranking
        if rerank:
            docs = self._rerank_documents(docs, question)

        # Build prompt with history
        prompt = self._build_prompt(session_id)

        # Build LCEL chain: prompt → LLM → parse output
        chain = prompt | self._llm | StrOutputParser()

        # Invoke with context and input
        context_str = _format_docs(docs)
        answer = chain.invoke({"context": context_str, "input": question})

        sources = list({
            doc.metadata.get("source_file", "unknown")
            for doc in docs
        })

        # Save to memory
        if session_id:
            session = memory_manager.get_or_create_session(session_id)
            session.add_user_message(question)
            session.add_assistant_message(answer)

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
        }

    async def query_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        rerank: bool = False,
    ) -> AsyncIterator[str]:
        """
        Async streaming query — yields a JSON string of sources first, then answer tokens.
        """
        if not self._is_ready:
            raise RuntimeError("No documents indexed. Upload documents first.")

        # Retrieve documents (sync — BM25 has no async)
        retriever = self._build_ensemble_retriever()
        docs = retriever.invoke(question)

        # Optional reranking
        if rerank:
            docs = self._rerank_documents(docs, question)

        # Build prompt with history
        prompt = self._build_prompt(session_id)

        # Build LCEL chain
        chain = prompt | self._llm | StrOutputParser()

        context_str = _format_docs(docs)
        
        # 1. Yield the sources first as a special JSON marker
        sources = list({doc.metadata.get("source_file", "unknown") for doc in docs})
        yield json.dumps({"__sources__": sources})

        # 2. Stream the response tokens
        full_answer = ""
        async for token in chain.astream({"context": context_str, "input": question}):
            full_answer += token
            yield token

        # Save to memory after streaming completes
        if session_id:
            session = memory_manager.get_or_create_session(session_id)
            session.add_user_message(question)
            session.add_assistant_message(full_answer)

    # ──────────────────────────────────────────
    # Knowledge Base Management
    # ──────────────────────────────────────────

    def list_documents(self) -> Dict[str, Any]:
        """List all indexed documents."""
        return {
            "files": self._indexed_files,
            "total_chunks": len(self._documents),
            "is_ready": self._is_ready,
        }

    def clear_index(self) -> Dict[str, str]:
        """Clear the entire knowledge base."""
        import shutil

        # Reset in-memory state
        self._faiss_store = None
        self._documents = []
        self._indexed_files = []
        self._is_ready = False

        # Remove files on disk
        store_path = settings.vector_store_path
        if os.path.exists(store_path):
            shutil.rmtree(store_path)
            os.makedirs(store_path, exist_ok=True)

        logger.info("🗑️ Knowledge base cleared")
        return {"status": "cleared", "message": "Knowledge base has been cleared."}

    @property
    def is_ready(self) -> bool:
        return self._is_ready
