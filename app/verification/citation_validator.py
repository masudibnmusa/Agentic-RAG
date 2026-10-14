"""Deterministic citation checks: unknown labels and uncited factual sentences."""
import re
from dataclasses import dataclass, field

from app.retrieval.evidence_pool import EvidencePool

CITE = re.compile(r"\[(S\d+)\]")


@dataclass
class CitationReport:
    invalid_labels: list = field(default_factory=list)
    uncited_sentences: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.invalid_labels and not self.uncited_sentences


def validate_citations(answer: str, pool: EvidencePool) -> CitationReport:
    labels = set(CITE.findall(answer))
    invalid = sorted(l for l in labels if pool.get(l) is None)
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    uncited = [s for s in sentences if len(s.split()) > 6 and not CITE.search(s)]
    return CitationReport(invalid, uncited)