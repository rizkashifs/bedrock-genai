"""
Vector store and retrieval service for PDF chunks.

Uses AWS Bedrock Titan Embeddings (amazon.titan-embed-text-v2:0) for encoding
and scikit-learn cosine_similarity for nearest-neighbour search.

Preserves the same Chunk / RetrievedChunk / RetrievalService interface as
rag-pdf-expert so all callers are unaffected by the embedding back-end swap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config.settings import settings
from app.models.bedrock_client import bedrock_client
from app.services.chunking import Chunk
from app.utils.logger import logger


# ── Embedding helper ───────────────────────────────────────────────────────

def get_titan_embedding(text: str) -> Optional[List[float]]:
    """Embed a single text string using Bedrock Titan Embeddings V2."""
    try:
        response_body = bedrock_client.invoke_model(
            body={"inputText": text},
            model_id=settings.titan_embed_model_id,
        )
        return response_body["embedding"]
    except Exception as exc:
        logger.error("Titan embedding failed: %s", exc)
        return None


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


# ── In-memory vector store ─────────────────────────────────────────────────

class VectorStore:
    """
    In-memory vector store backed by Titan Embeddings + cosine similarity.

    Public interface mirrors the FAISS-based store in rag-pdf-expert:
        add_chunks(chunks)
        search(query, k) -> List[RetrievedChunk]
        get_chunk_by_id(chunk_id) -> Optional[Chunk]
    """

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._embeddings: List[List[float]] = []

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        logger.info("Embedding %d chunks with Titan...", len(chunks))
        for chunk in chunks:
            embedding = get_titan_embedding(chunk.text)
            if embedding is None:
                logger.warning("Skipping chunk %s — embedding failed", chunk.id)
                continue
            self._chunks.append(chunk)
            self._embeddings.append(embedding)
        logger.info("VectorStore now holds %d chunks", len(self._chunks))

    def search(self, query: str, k: int = 10) -> List[RetrievedChunk]:
        if not self._chunks:
            return []
        query_embedding = get_titan_embedding(query)
        if query_embedding is None:
            logger.error("Query embedding failed — returning empty results")
            return []
        matrix = np.array(self._embeddings)
        q_vec = np.array(query_embedding).reshape(1, -1)
        scores = cosine_similarity(q_vec, matrix)[0]
        top_k = min(k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        for chunk in self._chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def clear(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()


# ── High-level retrieval service ───────────────────────────────────────────

class RetrievalService:
    """High-level retrieval API consumed by chat orchestration."""

    def __init__(self, store: Optional[VectorStore] = None) -> None:
        self._store = store or VectorStore()

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._store.add_chunks(chunks)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = k or settings.max_chunks_per_query
        return self._store.search(query, k=top_k)

    def retrieve_with_images(
        self,
        query: str,
        k: Optional[int] = None,
        include_linked_images: bool = True,
    ) -> List[RetrievedChunk]:
        """Retrieve text chunks and optionally expand with linked image chunks."""
        results = self.retrieve(query, k)
        if not include_linked_images:
            return results
        expanded: List[RetrievedChunk] = []
        seen: set = set()
        for retrieved in results:
            chunk = retrieved.chunk
            if chunk.id not in seen:
                expanded.append(retrieved)
                seen.add(chunk.id)
            if chunk.chunk_type == "text":
                for img_id in chunk.metadata.get("linked_image_ids", []):
                    if img_id in seen:
                        continue
                    img_chunk = self._store.get_chunk_by_id(img_id)
                    if img_chunk:
                        expanded.append(RetrievedChunk(chunk=img_chunk, score=retrieved.score))
                        seen.add(img_id)
        return expanded

    def retrieve_by_type(
        self,
        query: str,
        chunk_type: str = "text",
        k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        top_k = k or settings.max_chunks_per_query
        all_results = self._store.search(query, k=top_k * 3)
        return [r for r in all_results if r.chunk.chunk_type == chunk_type][:top_k]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        return self._store.get_chunk_by_id(chunk_id)


# ── Process-wide singleton ─────────────────────────────────────────────────

_global_retrieval_service: Optional[RetrievalService] = None


def get_global_retrieval_service() -> RetrievalService:
    """Shared RetrievalService so indexed chunks are reused across requests."""
    global _global_retrieval_service
    if _global_retrieval_service is None:
        _global_retrieval_service = RetrievalService()
    return _global_retrieval_service
