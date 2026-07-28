"""
Stage 6: Interface
Checkpoint 3: retrieval (real embeddings) + generation (Gemini) wired
together, with a working Streamlit UI (query box, answer area, expandable
sources with doc name + similarity score, adjustable top_k).
"""
import streamlit as st
from dotenv import load_dotenv
from src.retrieve import build_index
from src.generate import generate_answer, GenerationError

load_dotenv()  # pulls GEMINI_API_KEY from a local .env file, if present

st.set_page_config(page_title="Vue Docs RAG Search", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Building index over the doc collection...")
def get_index():
    return build_index()


try:
    index = get_index()
    index_error = None
except Exception as e:  # non-functional requirement: basic error handling
    index = None
    index_error = str(e)

st.title("🔍 Vue 3 Docs — RAG Search")
st.caption("CS382 Final Project · Checkpoint 3: embeddings + LLM generation, cited answers")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=5)
    st.divider()
    if index:
        st.metric("Documents indexed", len({c.doc_id for c in index.chunks}))
        st.metric("Chunks indexed", len(index.chunks))
    st.divider()
    st.caption(
        "Pipeline stage: **Ingest → Chunk → Embed → Cosine Similarity → Rank → Generate (Gemini)**."
    )

if index_error:
    st.error(f"Failed to build the index: {index_error}")
    st.stop()

query = st.text_input("Ask a question about Vue 3:", placeholder="e.g. how does v-model work on a custom component?")
submitted = st.button("Search", type="primary")

if submitted:
    if not query or not query.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Retrieving..."):
            results = index.search(query, top_k=top_k)

        st.subheader("Answer")
        if not results:
            st.info("Nothing relevant found in the doc collection for that query. Try rephrasing it.")
        else:
            with st.spinner("Generating..."):
                try:
                    answer = generate_answer(query, results)
                    st.write(answer)
                except GenerationError as e:
                    # basic error handling: show a clean message, don't crash,
                    # and still fall back to the top passage so the demo keeps working
                    st.error(str(e))
                    best = results[0]
                    st.caption(f"Falling back to the closest matching passage, from **{best.chunk.doc_title}**:")
                    st.write(best.chunk.text[:600] + ("..." if len(best.chunk.text) > 600 else ""))

        st.subheader(f"Sources ({len(results)})")
        for r in results:
            with st.expander(f"{r.chunk.doc_title} — {r.chunk.heading}  ·  similarity {r.score:.3f}"):
                st.caption(f"doc: {r.chunk.doc_id}.md  ·  chunk: {r.chunk.chunk_id}")
                st.write(r.chunk.text)
