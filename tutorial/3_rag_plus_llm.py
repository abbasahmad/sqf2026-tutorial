"""
==============================================================================
 STEP 3 — Put it together: RAG + a local LLM
==============================================================================

Now we combine everything:

    Step 1 (1_local_llm.py) runs a local model we can ask questions
    Step 2 (2_rag.py)       finds the document chunks relevant to a question

Remember Step 1: the model listed the seven testing principles from memory and
got them wrong. The RAG idea fixes this: instead of asking the model from its
general knowledge alone, we first RETRIEVE the relevant pieces of the real
document, then HAND THEM to the model inside the prompt and say:
"Answer using ONLY this context." The answer is now grounded and correct.

The flow:

    question ->  retrieve top chunks  ->  build a prompt with those chunks
             ->  send prompt to local model  ->  grounded answer + sources

------------------------------------------------------------------------------
 Make sure you have done the setup from the first two files:
   - Ollama installed + `ollama pull llama3.2:3b` + `pip install ollama` (Step 1)
   - the embedding model downloads automatically the first time (Step 2)

 Then run:
   python 3_rag_plus_llm.py
==============================================================================
"""

import os
import importlib.util
import ollama


# ----------------------------------------------------------------------------
# Reuse the functions we already wrote in 2_rag.py.
# Normally we would just write `import rag`, but our file is named "2_rag.py"
# and Python can't import a name that starts with a number. So we load it by
# its file path instead. (The result `rag` behaves like any imported module.)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rag", os.path.join(HERE, "2_rag.py"))
rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag)


MODEL_NAME = "llama3.2:3b"


# ----------------------------------------------------------------------------
# Build the prompt that we send to the model.
# This is the heart of RAG: we paste the retrieved chunks into the prompt as
# "context" and instruct the model to rely on it.
# ----------------------------------------------------------------------------
def build_prompt(question, retrieved_chunks):
    # Join the retrieved chunks into one block of context text, each labelled
    # with the document it came from so the model (and we) can trace sources.
    context_blocks = []
    for result in retrieved_chunks:
        chunk = result["chunk"]
        context_blocks.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    # The actual instruction sent to the model.
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return prompt


def ask_llm(prompt):
    """Send the prompt to the local model and return its answer (see Step 2)."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful software testing assistant."},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.4,  # low = factual, good for grounded answers
            # Make the model's context window big enough to actually read all
            # the retrieved chunks (the default is small and would cut them off).
            "num_ctx": 8192,
        },
    )
    return response["message"]["content"]


def main():
    # --- Prepare the knowledge base (Step 1) -----------------------------
    # Load the embedding model, then get the chunk vectors. The first time this
    # builds embeddings.json from the PDF; afterwards it just loads that file.
    print("Preparing the document index...\n")
    model = rag.load_embedding_model()
    chunks, chunk_vectors = rag.get_embeddings(model)

    # --- The question we want a grounded answer to -----------------------
    # The SAME question the model got wrong on its own in Step 1.
    question = ("According to the ISTQB® Certified Tester Foundation Level v4.0, "
                "what are the seven testing principles?")
    # We SEARCH the document with focused key words (not the whole sentence),
    # but we ASK the model the full question. Retrieval likes short key terms.
    search_query = "seven testing principles"
    print(f"QUESTION: {question}\n")

    # --- RETRIEVE the most relevant chunks (Step 2) ----------------------
    retrieved = rag.retrieve(search_query, chunks, chunk_vectors, model, top_k=4)
    print("Retrieved these chunks as context:")
    for rank, result in enumerate(retrieved, start=1):
        print(f"  [{rank}] {result['chunk']['source']}  (score={result['score']:.3f})")
    print()

    # --- BUILD the prompt and ASK the local model (Steps 2 + 3) ----------
    prompt = build_prompt(question, retrieved)
    print("Sending the question + retrieved context to the local model...")
    print("(The first call may take a few seconds.)\n")
    answer = ask_llm(prompt)

    print("GROUNDED ANSWER:")
    print("=" * 70)
    print(answer)
    print("=" * 70)
    print("\nThis answer is based on the ISTQB syllabus in docs/, produced by a")
    print("model running entirely on your laptop. That is RAG + a local LLM.")


if __name__ == "__main__":
    main()
