"""Vector search restricted by metadata (source, doc type, date range)."""
from app.retrieval.evidence_pool import Chunk
from app.tools import vector_search

SPEC = {
    "name": "filtered_search",
    "description": (
        "Semantic search restricted by metadata. Use when the question mentions a specific "
        "file, document type (md/txt/pdf), or a date range."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "source": {"type": "string", "description": "Exact source file name/path"},
            "doc_type": {"type": "string", "description": "md, txt, or pdf"},
            "date_from": {"type": "string", "description": "YYYY-MM-DD"},
            "date_to": {"type": "string", "description": "YYYY-MM-DD"},
        },
        "required": ["query"],
    },
}


def _to_int(date: str) -> int:
    return int(date.replace("-", ""))


def build_where(source=None, doc_type=None, date_from=None, date_to=None) -> dict | None:
    conds = []
    if source:
        conds.append({"source": source})
    if doc_type:
        conds.append({"doc_type": doc_type})
    if date_from:
        conds.append({"date": {"$gte": _to_int(date_from)}})
    if date_to:
        conds.append({"date": {"$lte": _to_int(date_to)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def run(query, source=None, doc_type=None, date_from=None, date_to=None, k=None) -> list[Chunk]:
    where = build_where(source, doc_type, date_from, date_to)
    return vector_search.run(query, k=k, where=where)