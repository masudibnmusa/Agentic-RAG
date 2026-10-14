"""Chunk type + the evidence pool that accumulates and dedupes chunks across iterations."""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    label: str = ""  # short citation label (S1, S2, ...) assigned by the pool

    def render(self, max_chars: int | None = None) -> str:
        text = self.text
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "..."
        return f'<chunk id="{self.label}" source="{self.source}">\n{text}\n</chunk>'


class EvidencePool:
    def __init__(self):
        self._chunks: dict[str, Chunk] = {}
        self._by_label: dict[str, Chunk] = {}
        self._relevant: dict[str, set[str]] = {}  # sub-question -> relevant chunk ids
        self._counter = 0

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> list[Chunk]:
        """Add chunks (deduped by id). Returns the pooled instances, which carry labels."""
        stored = []
        for chunk in chunks:
            existing = self._chunks.get(chunk.id)
            if existing:
                existing.score = max(existing.score, chunk.score)
                stored.append(existing)
            else:
                self._counter += 1
                chunk.label = f"S{self._counter}"
                self._chunks[chunk.id] = chunk
                self._by_label[chunk.label] = chunk
                stored.append(chunk)
        return stored

    def get(self, label: str) -> Chunk | None:
        return self._by_label.get(label)

    def mark_relevant(self, sub_question: str, chunk_ids: set[str]) -> int:
        """Record relevant chunks for a sub-question. Returns how many are new."""
        known = self._relevant.setdefault(sub_question, set())
        new = chunk_ids - known
        known |= chunk_ids
        return len(new)

    def relevant_for(self, sub_question: str) -> list[Chunk]:
        ids = self._relevant.get(sub_question, set())
        return sorted((self._chunks[i] for i in ids), key=lambda c: c.score, reverse=True)

    def all_relevant(self, limit: int | None = None) -> list[Chunk]:
        ids = set().union(*self._relevant.values()) if self._relevant else set()
        ranked = sorted((self._chunks[i] for i in ids), key=lambda c: c.score, reverse=True)
        return ranked[:limit] if limit else ranked