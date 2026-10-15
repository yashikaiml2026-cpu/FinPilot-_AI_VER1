from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb

from src.config.settings import settings


class PolicyKnowledgeBase:
    def __init__(self, knowledge_dir: Path | None = None, persist_dir: Path | None = None) -> None:
        self.knowledge_dir = knowledge_dir or settings.knowledge_base_dir
        self.persist_dir = persist_dir or settings.chroma_path
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="banking_policy_kb",
            metadata={"hnsw:space": "cosine"},
        )

    def _split_text(self, text: str, source_name: str) -> list[dict[str, str]]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        chunks: list[dict[str, str]] = []
        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            aggregate = " "
            for sentence in sentences:
                if len(aggregate) + len(sentence) > 500:
                    chunks.append({"source": source_name, "text": aggregate.strip()})
                    aggregate = sentence
                else:
                    aggregate = f"{aggregate} {sentence}".strip()
            if aggregate.strip():
                chunks.append({"source": source_name, "text": aggregate.strip()})
        return chunks

    def _fallback_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        values = []
        for idx in range(384):
            chunk = digest[(idx * 2) % len(digest):(idx * 2) % len(digest) + 2]
            numeric = int(chunk or "00", 16)
            values.append((numeric / 255.0) - 0.5)
        norm = (sum(value * value for value in values) ** 0.5) or 1.0
        return [(value / norm) for value in values]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._fallback_embedding(text) for text in texts]

    def load_and_index(self) -> None:
        if self.collection.count() > 0:
            return

        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        ids: list[str] = []

        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for chunk_index, chunk in enumerate(self._split_text(text, path.name)):
                documents.append(chunk["text"])
                metadatas.append({"source": chunk["source"], "document": path.name})
                ids.append(f"{path.stem}-{chunk_index}")

        if not documents:
            return

        embeddings = self._embed_texts(documents)
        self.collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

    def search(self, question: str, k: int = 4) -> list[dict[str, Any]]:
        self.load_and_index()
        stored = self.collection.get(include=["documents", "metadatas"])
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        query_terms = self._terms(question)
        if not documents or not query_terms:
            return []

        document_terms = [self._terms(document) for document in documents]
        document_frequency = {
            term: sum(term in terms for terms in document_terms)
            for term in query_terms
        }
        scored = []
        for document, metadata, terms in zip(documents, metadatas, document_terms):
            overlap = query_terms.intersection(terms)
            if not overlap:
                continue
            score = sum(
                1 + (len(documents) / (1 + document_frequency[term]))
                for term in overlap
            ) / max(len(query_terms), 1)
            scored.append(
                {
                    "content": document,
                    "source": metadata.get("document", metadata.get("source", "unknown")),
                    "score": round(score, 4),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:k]

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop_words = {"a", "an", "and", "are", "for", "in", "is", "of", "on", "or", "the", "to", "when", "with"}
        terms = set()
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) < 3 or word in stop_words:
                continue
            if word.startswith("escal"):
                word = "escal"
            elif word.endswith("ing") and len(word) > 5:
                word = word[:-3]
            elif word.endswith("ed") and len(word) > 4:
                word = word[:-2]
            elif word.endswith("s") and len(word) > 4:
                word = word[:-1]
            terms.add(word)
        return terms


policy_kb = PolicyKnowledgeBase()


def query_policy_knowledge_base(question: str, k: int = 4) -> list[dict[str, Any]]:
    return policy_kb.search(question, k=k)
