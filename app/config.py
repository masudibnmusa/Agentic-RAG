"""Central configuration. Override any value via environment variables / .env."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # API keys
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Models: strong model for planning/drafting/revision, cheap model for grading/extraction
    planner_model: str = os.getenv("PLANNER_MODEL", "claude-sonnet-5-5")
    grader_model: str = os.getenv("GRADER_MODEL", "claude-haiku-5-5")

    # Budgets and limits
    max_steps_per_subquestion: int = int(os.getenv("MAX_STEPS_PER_SUBQ", 3))
    max_total_steps: int = int(os.getenv("MAX_TOTAL_STEPS", 10))
    max_tokens_budget: int = int(os.getenv("MAX_TOKENS_BUDGET", 60000))
    max_revisions: int = 2
    max_subquestions: int = 4
    no_progress_patience: int = 2
    max_evidence_chunks: int = 12

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", 8))
    rerank_top_n: int = 6
    reranker: str = os.getenv("RERANKER", "none")        # none | cross_encoder | cohere
    grader_mode: str = os.getenv("GRADER_MODE", "llm")   # llm | reranker
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", 0.5))
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Sources the router may choose from: docs, sql, web
    enabled_sources: list = field(
        default_factory=lambda: os.getenv("ENABLED_SOURCES", "docs").split(",")
    )

    # Feature toggles (used by ablations)
    enable_routing: bool = _bool("ENABLE_ROUTING", True)
    enable_rewriting: bool = _bool("ENABLE_REWRITING", True)
    enable_decomposition: bool = _bool("ENABLE_DECOMPOSITION", True)
    enable_grading: bool = _bool("ENABLE_GRADING", True)
    enable_verification: bool = _bool("ENABLE_VERIFICATION", True)
    use_hyde: bool = _bool("USE_HYDE", False)
    use_query_expansion: bool = _bool("USE_QUERY_EXPANSION", False)

    # Paths
    docs_dir: Path = ROOT / "data" / "docs"
    vector_db_dir: Path = ROOT / "data" / "vector_db"
    traces_dir: Path = ROOT / "data" / "traces"
    sql_db_path: Path = ROOT / "data" / "structured.db"

    def __post_init__(self):
        for p in (self.docs_dir, self.vector_db_dir, self.traces_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()