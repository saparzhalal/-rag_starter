"""
TF-IDF + cosine similarity retrieval -- the original Checkpoint 0 baseline.
Kept here (not deleted) so retrieve.py's real embeddings can be compared
against it for the evaluation write-up (Section 8: "how much did real
embeddings actually improve retrieval over TF-IDF?").
Not imported by app.py anymore -- see retrieve.py for the active version.
"""
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunk import Chunk, build_corpus

NO_MATCH_THRESHOLD = 0.05  # below this, we say "nothing relevant found"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class RagIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.85)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = [SearchResult(self.chunks[i], float(scores[i])) for i in ranked_idx]
        # graceful failure: drop results indistinguishable from noise
        results = [r for r in results if r.score >= NO_MATCH_THRESHOLD]
        return results


def build_index() -> RagIndex:
    chunks = build_corpus()
    return RagIndex(chunks)


if __name__ == "__main__":
    index = build_index()
    print(f"Indexed {len(index.chunks)} chunks, vocab size {len(index.vectorizer.vocabulary_)}")

    for q in ["how does the composition api work", "how do I use v-model on a component", "what is teleport"]:
        print(f"\nQuery: {q!r}")
        for r in index.search(q, top_k=3):
            print(f"  [{r.score:.3f}] {r.chunk.doc_title} / {r.chunk.heading}  ({r.chunk.chunk_id})")
