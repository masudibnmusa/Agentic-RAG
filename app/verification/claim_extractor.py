"""Break an answer into atomic claims."""
from dataclasses import dataclass, field

from app.agent.llm_client import get_llm
from app.config import settings

SYSTEM = """Split the answer into atomic factual claims. Each claim must be a single, self-contained
statement. Keep the citation labels (like S1) that the answer attached to that claim.
Skip hedges, transitions, and statements about missing information.
Return JSON: {"claims": [{"text": "...", "citations": ["S1"]}]}"""


@dataclass
class Claim:
    text: str
    citations: list = field(default_factory=list)


def extract_claims(answer: str) -> list[Claim]:
    data = get_llm().ask_json(SYSTEM, answer, model=settings.grader_model, max_tokens=900)
    claims = []
    for item in data.get("claims", []):
        if isinstance(item, dict) and item.get("text"):
            claims.append(Claim(item["text"], [c for c in item.get("citations", []) if isinstance(c, str)]))
    return claims