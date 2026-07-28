"""
Stage 1: Ingest
Loads raw documents from disk. Swap DOCS_DIR or this whole module
if you move to a different document collection later.
"""
from pathlib import Path
from dataclasses import dataclass

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


@dataclass
class RawDocument:
    doc_id: str          # filename without extension, used as the citation key
    filename: str
    text: str


def load_documents(docs_dir: Path = DOCS_DIR) -> list[RawDocument]:
    """Load every .md file in docs_dir into a RawDocument."""
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(RawDocument(doc_id=path.stem, filename=path.name, text=text))
    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}")
    for d in docs[:5]:
        print(f"  - {d.filename} ({len(d.text)} chars)")
