"""
Stage 1b: Chunk
Section-aware chunking: split each document on markdown headings so each
chunk is a coherent, self-contained section rather than an arbitrary
fixed-size slice. This is a defensible strategy per the brief (Section 2.2)
because Vue's docs are already organized as heading -> explanation -> example.
"""
import re
from dataclasses import dataclass
from .ingest import RawDocument, load_documents

MIN_CHUNK_CHARS = 200   # merge tiny sections into neighbors
MAX_CHUNK_CHARS = 1800  # split oversized sections further


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str


def _clean(raw: str) -> str:
    """Strip YAML frontmatter, Vue <script setup> blocks, and doc-tool
    container syntax that would just add noise to the embedding/TF-IDF space."""
    raw = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<script setup>.*?</script>", "", raw, flags=re.DOTALL)
    raw = re.sub(r":::\w*.*?:::", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\{#[\w-]+\}", "", raw)  # strip {#custom-anchor-id}
    return raw.strip()


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on '#' / '##' headings. Returns list of (heading, body)."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Introduction"
    current_body: list[str] = []
    for line in lines:
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, current_body))
    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def chunk_document(doc: RawDocument) -> list[Chunk]:
    cleaned = _clean(doc.text)
    sections = _split_sections(cleaned)
    doc_title = sections[0][0] if sections else doc.doc_id

    chunks: list[Chunk] = []
    buffer_heading, buffer_text = None, ""
    idx = 0

    def flush():
        nonlocal buffer_heading, buffer_text, idx
        if buffer_text.strip():
            idx += 1
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{idx}",
                doc_id=doc.doc_id,
                doc_title=doc_title,
                heading=buffer_heading or doc_title,
                text=buffer_text.strip(),
            ))
        buffer_heading, buffer_text = None, ""

    for heading, body in sections:
        candidate = (buffer_text + "\n\n" + body).strip() if buffer_text else body
        if len(candidate) > MAX_CHUNK_CHARS and buffer_text:
            flush()
            buffer_heading, buffer_text = heading, body
        elif len(candidate) < MIN_CHUNK_CHARS:
            buffer_heading = buffer_heading or heading
            buffer_text = candidate
        else:
            buffer_heading, buffer_text = heading, candidate
            flush()
    flush()

    # further split any chunk still over MAX_CHUNK_CHARS on paragraph breaks
    final: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= MAX_CHUNK_CHARS:
            final.append(c)
            continue
        parts = c.text.split("\n\n")
        buf = ""
        sub_idx = 0
        for p in parts:
            if len(buf) + len(p) > MAX_CHUNK_CHARS and buf:
                sub_idx += 1
                final.append(Chunk(f"{c.chunk_id}.{sub_idx}", c.doc_id, c.doc_title, c.heading, buf.strip()))
                buf = p
            else:
                buf = (buf + "\n\n" + p).strip()
        if buf.strip():
            sub_idx += 1
            final.append(Chunk(f"{c.chunk_id}.{sub_idx}", c.doc_id, c.doc_title, c.heading, buf.strip()))
    return final


def build_corpus(docs_dir=None) -> list[Chunk]:
    docs = load_documents(docs_dir) if docs_dir else load_documents()
    all_chunks: list[Chunk] = []
    for d in docs:
        all_chunks.extend(chunk_document(d))
    return all_chunks


if __name__ == "__main__":
    chunks = build_corpus()
    lengths = [len(c.text) for c in chunks]
    print(f"{len(chunks)} chunks from the corpus")
    print(f"avg length: {sum(lengths)//len(lengths)} chars, min: {min(lengths)}, max: {max(lengths)}")
    print("\nSample chunk:")
    print(f"  doc={chunks[5].doc_id}  heading={chunks[5].heading!r}")
    print(f"  text={chunks[5].text[:200]!r}...")
