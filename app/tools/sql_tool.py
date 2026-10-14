"""Read-only SQLite lookups for structured data."""
import hashlib
import re
import sqlite3
from contextlib import closing

from app.config import settings
from app.retrieval.evidence_pool import Chunk

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b",
    re.I,
)
MAX_ROWS = 50


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{settings.sql_db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def schema_text() -> str:
    if not settings.sql_db_path.exists():
        return "No database configured."
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(r[0] for r in rows if r[0])


def get_spec() -> dict:
    return {
        "name": "sql_query",
        "description": (
            "Run a single read-only SQLite SELECT query for structured or numeric data. "
            "Schema:\n" + schema_text()
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A SQLite SELECT statement"}},
            "required": ["query"],
        },
    }


def _validate(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise ValueError("Only SELECT queries are allowed")
    if ";" in sql or _FORBIDDEN.search(sql):
        raise ValueError("Query contains forbidden statements")
    return sql


def run(query: str) -> list[Chunk]:
    sql = _validate(query)
    with closing(_connect()) as conn:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS)
    lines = [" | ".join(cols)] + [" | ".join(str(v) for v in row) for row in rows]
    cid = "sql::" + hashlib.md5(sql.encode()).hexdigest()[:10]
    return [
        Chunk(
            id=cid,
            text=f"SQL: {sql}\n" + "\n".join(lines),
            source="sql",
            metadata={"query": sql},
            score=0.8,
        )
    ]