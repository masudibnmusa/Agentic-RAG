"""Tool definitions exposed to the LLM, plus a safe dispatcher."""
from app.retrieval.evidence_pool import Chunk
from app.tools import (
    fetch_document,
    hybrid_search,
    keyword_search,
    metadata_filter,
    sql_tool,
    vector_search,
    web_search_tool,
)

# tool name -> (module, source it belongs to)
_TOOLS = {
    "vector_search": (vector_search, "docs"),
    "keyword_search": (keyword_search, "docs"),
    "hybrid_search": (hybrid_search, "docs"),
    "filtered_search": (metadata_filter, "docs"),
    "fetch_document": (fetch_document, "docs"),
    "sql_query": (sql_tool, "sql"),
    "web_search": (web_search_tool, "web"),
}


def _spec(module) -> dict:
    return module.get_spec() if hasattr(module, "get_spec") else module.SPEC


def tool_specs(sources: set[str]) -> list[dict]:
    return [_spec(module) for module, src in _TOOLS.values() if src in sources]


def run_tool(name: str, args: dict) -> tuple[list[Chunk], str | None]:
    entry = _TOOLS.get(name)
    if not entry:
        return [], f"unknown tool: {name}"
    try:
        return entry[0].run(**args), None
    except Exception as e:  # tool failures must never crash the agent
        return [], f"{type(e).__name__}: {e}"