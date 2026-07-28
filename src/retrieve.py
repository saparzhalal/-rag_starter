"""
Stage 2-4: Vectorize -> Vector Store -> Retrieve
Checkpoint 2: real embeddings via sentence-transformers, replacing the
TF-IDF baseline (kept in retrieve_tfidf.py for comparison). Same
RagIndex/SearchResult/build_index() interface as before, so app.py and
everything upstream of this file are untouched.

First run on a new machine downloads the model (~90MB, one-time, cached
locally afterward by sentence-transformers/huggingface_hub) -- needs
internet access for that first download only.
"""
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunk import Chunk, build_corpus

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, local -- per the brief's suggested stack
NO_MATCH_THRESHOLD = 0.3  # cosine similarity on real embeddings sits higher than TF-IDF's;
                          # re-tune this once you see real scores on your machine (see README)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class RagIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.model = SentenceTransformer(MODEL_NAME)

        texts = [
            f"{c.doc_title}\n{c.heading}\n\n{c.text}"
            for c in chunks
        ]

        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec  # cosine similarity (both sides normalized)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        results = [SearchResult(self.chunks[i], float(scores[i])) for i in ranked_idx]
        # graceful failure: drop results indistinguishable from noise
        results = [r for r in results if r.score >= NO_MATCH_THRESHOLD]
        return results


def build_index() -> RagIndex:
    chunks = build_corpus()
    return RagIndex(chunks)


if __name__ == "__main__":
    index = build_index()
    print(f"Indexed {len(index.chunks)} chunks with {MODEL_NAME}")

    for q in [
        "how does the composition api work",
        "how do I use v-model on a component",
        "what is teleport",
        "why would my page flicker when it loads",
    ]:
        print(f"\nQuery: {q!r}")
        for r in index.search(q, top_k=3):
            print(f"  [{r.score:.3f}] {r.chunk.doc_title} / {r.chunk.heading}  ({r.chunk.chunk_id})")
