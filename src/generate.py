"""
Stage 5: Generate
Takes the query + retrieved chunks, builds a grounded prompt, and calls
Gemini to produce a cited answer. Isolated behind generate_answer() so
app.py doesn't need to know which LLM backend is in use -- swapping to
Claude or GPT later just means rewriting this file.
"""
import os
from google import genai

MODEL_NAME = "gemini-3.1-flash-lite"  # free-tier eligible, fast -- fine for a live demo

SYSTEM_INSTRUCTION = (
   
    "You are a helpful assistant that answers questions about Vue 3 using ONLY "
    "the documentation excerpts provided. Answer the user's question directly "
    "and naturally. Combine information across multiple excerpts when needed. "
    "If the answer is only partially supported by the excerpts, clearly say so. "
    "Do NOT use outside knowledge or invent facts. Support every answer with "
    "inline citations such as [1], [2], etc."
)



class GenerationError(Exception):
    """Raised on missing key / API failure so app.py can show a clean error
    instead of crashing (Non-Functional Expectation: basic error handling)."""


def _build_prompt(query: str, results) -> str:
    excerpts = "\n\n".join(
        f"[{i+1}] (from {r.chunk.doc_title} / {r.chunk.heading})\n{r.chunk.text}"
        for i, r in enumerate(results)
    )
    return (
       
    f"Documentation excerpts:\n\n{excerpts}\n\n"
    f"Question: {query}\n\n"
    "Instructions:\n"
    "- Answer the question directly.\n"
    "- Combine information from multiple excerpts when appropriate.\n"
    "- If the excerpts only partially answer the question, explain that clearly.\n"
    "- Do not use knowledge outside the provided excerpts.\n"
    "- Cite every factual statement using [1], [2], etc.\n\n"
    "Answer:"
)
    


def generate_answer(query: str, results) -> str:
    """results: list[SearchResult] from retrieve.py's index.search().
    Returns generated answer text. Raises GenerationError on failure --
    caller decides how to display that (see app.py)."""
    if not results:
        return "Nothing relevant found in the doc collection for that question."

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GenerationError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and paste "
            "your key in, then restart the app."
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=_build_prompt(query, results),
            config={"system_instruction": SYSTEM_INSTRUCTION},
        )
        return response.text
    except Exception as e:
        raise GenerationError(f"Generation call failed: {e}") from e
