"""Chroma wrapper. Swap this file to use Qdrant or pgvector."""
from functools import lru_cache

import chromadb

from app.config import settings
from app.retrieval.evidence_pool import Chunk


class VectorStore:
    def __init__(self, collection: str = "docs"):
        self.client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
        self.collection_name = collection
        self.col = self._open()

    def _open(self):
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.col = self._open()

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        for i in range(0, len(ids), 256):
            self.col.upsert(
                ids=ids[i:i + 256],
                documents=texts[i:i + 256],
                metadatas=metadatas[i:i + 256],
            )

    def query(self, text: str, k: int | None = None, where: dict | None = None) -> list[Chunk]:
        k = k or settings.top_k
        count = self.col.count()
        if count == 0:
            return []
        res = self.col.query(query_texts=[text], n_results=min(k, count), where=where or None)
        return [
            Chunk(id=i, text=d, source=m.get("source", ""), metadata=m, score=1 - dist)
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get(self, where: dict | None = None) -> list[Chunk]:
        res = self.col.get(where=where or None, include=["documents", "metadatas"])
        return [
            Chunk(id=i, text=d, source=m.get("source", ""), metadata=m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()