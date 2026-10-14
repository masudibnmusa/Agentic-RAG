"""Load .txt / .md / .pdf files into Document objects."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(source_dir: Path) -> list[Document]:
    source_dir = Path(source_dir)
    docs = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        text = _read(path).strip()
        if not text:
            continue
        rel = str(path.relative_to(source_dir))
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        docs.append(
            Document(
                doc_id=rel,
                text=text,
                metadata={
                    "source": rel,
                    "doc_type": path.suffix.lower().lstrip("."),
                    "date": int(modified.strftime("%Y%m%d")),  # int so Chroma can range-filter
                },
            )
        )
    return docs