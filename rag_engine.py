"""
Local retrieval-augmented generation for ResearchRAG.

Embeddings: sentence-transformers all-MiniLM-L6-v2
Search: FAISS inner-product over L2-normalized vectors (cosine similarity)
Generation: Ollama qwen2.5:3b, answering only from retrieved passages
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from document_processor import TextChunk

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "qwen2.5:3b"
DEFAULT_TOP_K = 5
OLLAMA_HOST_HINT = "http://localhost:11434"

INDEX_FILENAME = "papers.index"
METADATA_FILENAME = "chunks.json"


class RAGEngine:
    """Build a FAISS index from PDF chunks and answer questions from retrieved text."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self.llm_model = llm_model

        self._embedder: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunks: list[TextChunk] = []

    @property
    def index_path(self) -> Path:
        return self.data_dir / INDEX_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / METADATA_FILENAME

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def load_embedder(self) -> SentenceTransformer:
        """Load the embedding model once and reuse it."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def build_index(self, chunks: list[TextChunk]) -> None:
        """Create a new FAISS index from chunks and save it to disk."""
        if not chunks:
            raise ValueError("Cannot build an index without text chunks.")

        texts = [chunk.text for chunk in chunks]
        vectors = self._embed_texts(texts)

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        self.index = index
        self.chunks = list(chunks)
        self.save()

    def save(self) -> None:
        if self.index is None:
            raise ValueError("There is no index to save.")

        faiss.write_index(self.index, str(self.index_path))
        payload = [chunk.to_dict() for chunk in self.chunks]
        self.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> bool:
        """Load a previously saved index. Returns False if files are missing."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        self.index = faiss.read_index(str(self.index_path))
        raw_chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.chunks = [TextChunk.from_dict(item) for item in raw_chunks]
        return self.is_ready

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Return the most similar chunks for a question."""
        if not self.is_ready:
            raise RuntimeError("The knowledge index is empty. Process PDFs first.")

        cleaned = question.strip()
        if not cleaned:
            return []

        query_vector = self._embed_texts([cleaned])
        k = max(1, min(top_k, len(self.chunks)))
        scores, indices = self.index.search(query_vector, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "score": float(score),
                }
            )
        return results

    def generate_answer(
        self,
        question: str,
        retrieved: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Ask Ollama to answer using only retrieved passages.

        Source labels are retrieval metadata (filename + page), not independently
        verified academic citations.
        """
        if not retrieved:
            return {
                "answer": (
                    "I could not find relevant passages in the processed papers "
                    "for this question."
                ),
                "ok": True,
                "error": None,
            }

        prompt = self._build_prompt(question, retrieved)
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful research assistant. Use only the "
                            "provided evidence. If the evidence is not enough, say so. "
                            "Do not invent papers, page numbers, or quotes. "
                            "When you refer to a source, use the filename and page "
                            "from the evidence labels. Those labels come from PDF "
                            "retrieval, not from a citation-verification system."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            return {
                "answer": "",
                "ok": False,
                "error": self._format_ollama_error(exc),
            }

        answer = _extract_chat_text(response).strip()
        if not answer:
            return {
                "answer": "",
                "ok": False,
                "error": "Ollama returned an empty response.",
            }

        return {"answer": answer, "ok": True, "error": None}

    def check_ollama(self) -> dict[str, Any]:
        """Report whether Ollama is reachable and whether the model is installed."""
        try:
            listing = ollama.list()
        except Exception as exc:
            return {
                "ok": False,
                "connected": False,
                "model_found": False,
                "models": [],
                "message": (
                    f"Could not connect to Ollama at {OLLAMA_HOST_HINT}. "
                    "Start Ollama, then pull the model with: ollama pull "
                    f"{self.llm_model}. Details: {exc}"
                ),
            }

        models = _model_names(listing)
        model_found = any(
            name == self.llm_model or name.startswith(self.llm_model)
            for name in models
        )
        if not model_found:
            return {
                "ok": False,
                "connected": True,
                "model_found": False,
                "models": models,
                "message": (
                    f"Ollama is running, but '{self.llm_model}' was not found. "
                    f"Install it with: ollama pull {self.llm_model}"
                ),
            }

        return {
            "ok": True,
            "connected": True,
            "model_found": True,
            "models": models,
            "message": f"Ollama is ready with {self.llm_model}.",
        }

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        embedder = self.load_embedder()
        vectors = embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def _build_prompt(self, question: str, retrieved: list[dict[str, Any]]) -> str:
        evidence_blocks = []
        for i, item in enumerate(retrieved, start=1):
            evidence_blocks.append(
                f"[{i}] {item['filename']}, page {item['page_number']}\n{item['text']}"
            )
        evidence = "\n\n".join(evidence_blocks)
        return (
            "Answer the question using only the evidence below. "
            "Quote or paraphrase the evidence. If it does not contain the answer, "
            "say that the papers do not provide enough information.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Answer:"
        )

    def _format_ollama_error(self, exc: Exception) -> str:
        text = str(exc).lower()
        if "not found" in text or "404" in text:
            return (
                f"The Ollama model '{self.llm_model}' is missing. "
                f"Run: ollama pull {self.llm_model}"
            )
        if "connection" in text or "refused" in text or "10061" in text:
            return (
                f"Could not reach Ollama. Make sure it is running on {OLLAMA_HOST_HINT}."
            )
        return f"Ollama error: {exc}"


def _extract_chat_text(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        return str(message.get("content", ""))
    message = getattr(response, "message", None)
    if message is None:
        return ""
    return str(getattr(message, "content", "") or "")


def _model_names(listing: Any) -> list[str]:
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models", [])

    names: list[str] = []
    for item in models or []:
        name = getattr(item, "model", None) or getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("model") or item.get("name")
        if name:
            names.append(str(name))
    return names
