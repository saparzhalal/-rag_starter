# Vue 3 / Nuxt Docs — RAG Search (CS382 Final Project)

A Retrieval-Augmented Generation search system over the official Vue 3 guide docs.
Ask a question, get back the most relevant passages with sources and similarity scores.

**Status: Checkpoint 3** — real embeddings + retrieval + Gemini-generated,
cited answers, all wired together in the Streamlit UI.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then paste your real Gemini API key into .env
streamlit run app.py
```

Get a free Gemini API key (no card required) at aistudio.google.com → "Get API key".

Opens at `http://localhost:8501`.

> First run also downloads the embedding model (~90MB, one-time, then
> cached locally) — needs a live internet connection for that first
> `streamlit run`. After that it works offline (generation still needs
> internet each time, since that's a live API call).

## Architecture

```
data/docs/*.md          52 Vue 3 guide docs (official docs, cloned from vuejs/docs)
      │
src/ingest.py            Stage 1: load raw markdown files
      │
src/chunk.py              Stage 1b: strip frontmatter/script tags, split into
      │                    section-aware chunks on markdown headings (579 chunks)
      │
src/retrieve.py            Stage 2-4: sentence-transformers (all-MiniLM-L6-v2)
      │                    embeds the chunk corpus, cosine-similarity search, rank top-k
      │                    (src/retrieve_tfidf.py: the old TF-IDF baseline, kept for
      │                     the evaluation write-up's before/after comparison)
      │
app.py                     Stage 6: Streamlit UI — query box, adjustable top_k,
                            answer area, expandable sources panel
      │
src/generate.py            Stage 5: builds a grounded prompt from the retrieved
                            chunks, calls Gemini (gemini-3.1-flash-lite),
                            returns a cited answer
```

Each stage is isolated in its own module on purpose: retrieval and generation
are separate files behind separate functions, so either can be swapped
(different embedding model, different LLM provider) without touching the rest.

## Known limitations (current stage)

- **Similarity threshold (`NO_MATCH_THRESHOLD` in retrieve.py) is a starting
  guess.** Watch the scores shown in the Sources panel on your own queries and
  adjust the constant if it's filtering out good matches or letting weak ones
  through.
- **No answer caching** — every submit re-calls the Gemini API, even for a
  repeated query.
- **In-memory index**, rebuilt on first load and cached — fine at this corpus
  size (579 chunks), would need a real vector store (FAISS/Chroma) at scale.
- **Single LLM call, no retry.** If the Gemini call fails (bad key, rate limit,
  network), the UI shows the error and falls back to the raw top passage
  instead of crashing — but it doesn't automatically retry.

## Document collection

52 markdown files from Vue 3's official guide (`vuejs/docs`, `src/guide/`),
chunked into 579 sections. Swap `data/docs/` for a different domain if needed —
`ingest.py` just reads whatever `.md` files are in that folder.
