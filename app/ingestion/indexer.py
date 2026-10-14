"""Embed + store chunks with metadata, and dump a BM25 corpus.
Usage: python -m app.ingestion.indexer --source data/docs --reset
"""
import argparse
import json
from pathlib import Path

from app.config import settings
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_documents
from app.retrieval.vector_store import get_store


def build_index(source_dir: Path, reset: bool = False) -> int:
    store = get_store()
    if reset:
        store.reset()

    ids, texts, metas = [], [], []
    for doc in load_documents(source_dir):
        for cid, text, meta in chunk_document(doc, settings.chunk_size, settings.chunk_overlap):
            ids.append(cid)
            texts.append(text)
            metas.append(meta)
    if ids:
        store.add(ids, texts, metas)

    # BM25 corpus mirrors everything in the store
    corpus = [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in store.get()]
    (settings.vector_db_dir / "bm25_corpus.json").write_text(json.dumps(corpus), encoding="utf-8")

    from app.tools import keyword_search
    keyword_search.clear_cache()
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(settings.docs_dir))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    n = build_index(Path(args.source), reset=args.reset)
    print(f"Indexed {n} chunks from {args.source}")


if __name__ == "__main__":
    main()