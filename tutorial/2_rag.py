"""
==============================================================================
 STEP 2 — How RAG works, from scratch, with NO API and NO LLM
==============================================================================

In Step 1 the local model answered about the seven ISTQB testing principles
from memory, and we could not trust it. Now we tackle the "Retrieval" idea: how
to PULL the real document and automatically find the exact part that answers a
question. No AI model is involved yet -- this is pure retrieval.

RAG = "Retrieval-Augmented Generation", and this file is the *Retrieval* part.

Our knowledge base is the official **ISTQB CTFL syllabus PDF** in the docs/
folder. The whole idea of RAG in five steps:

    1. LOAD     read the PDF document(s) in docs/
    2. CHUNK    cut the text into pieces ("chunks"), one per document SECTION
    3. EMBED    turn each chunk into a list of numbers (a "vector") using a
                small model that runs on YOUR laptop (no internet after the
                first download, no API key, no cloud)
    4. SAVE     store those vectors in a file (here: embeddings.json) so we
                don't have to re-embed the whole PDF every single time
    5. RETRIEVE turn the user's question into a vector too, then find the
                chunks whose vectors are the "closest" to it, using a simple
                formula called COSINE SIMILARITY

Run this file and read the printed output top to bottom:

    python 2_rag.py

The first run downloads a tiny embedding model (~90 MB) one time and builds
embeddings.json. Later runs reuse embeddings.json and are much faster.
==============================================================================
"""

import os                       # to work with file paths and folders
import re                       # to clean repeating boilerplate out of the PDF
import json                     # to save/load the embeddings file
import numpy as np              # to do the maths on vectors (lists of numbers)
from pypdf import PdfReader     # to read text out of PDF files
from sentence_transformers import SentenceTransformer  # the local embedding model


# ----------------------------------------------------------------------------
# Where things live (all relative to this file's folder).
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")                  # the PDF(s) to read
EMBEDDINGS_FILE = os.path.join(HERE, "embeddings.json")  # our saved vectors

# The name of the embedding model we download from the internet ONCE.
# "all-MiniLM-L6-v2" is small (~90 MB), fast, runs on a normal CPU, and turns
# any piece of text into a vector of 384 numbers.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================================
# STEP 1 — LOAD the documents (PDF and, if present, Markdown)
# ============================================================================
def clean_text(text):
    """Remove the page footer that repeats on EVERY page of the syllabus.

    That footer ("Certified Tester Foundation Level v4.0.1 Page N of 78 ...
    International Software Testing Qualifications Board") would otherwise be
    glued onto every chunk and pollute our search results.
    """
    text = re.sub(r"Certified Tester\s+Foundation Level\s+v4\.\s*0\.1", " ", text)
    text = re.sub(r"Page \d+ of 78", " ", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)   # the date stamp
    text = re.sub(r"©?\s*International Software Testing Qualifications Board",
                  " ", text)
    return text


def load_documents(docs_dir):
    """Read every PDF (and .md) file in the docs folder.

    Returns a list of dictionaries, one per file, like:
        [{"source": "ISTQB_CTFL_Syllabus_v4.0.1.pdf", "text": "..."}, ...]
    """
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, filename)

        if filename.lower().endswith(".pdf"):
            # Read a PDF: extract every page, then clean the repeating footer.
            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = clean_text("\n".join(pages))
            documents.append({"source": filename, "text": text})

        elif filename.lower().endswith(".md"):
            # Plain text files are even simpler.
            with open(path, "r", encoding="utf-8") as f:
                documents.append({"source": filename, "text": f.read()})

    return documents


# ============================================================================
# STEP 2 — CHUNK each document into pieces, one per SECTION
# ============================================================================
# WHY chunk at all? A whole document is too big and mixes many topics. Smaller
# pieces let us retrieve ONLY the part that answers the question.
#
# A naive way is to cut every N characters. But that can slice a list of items
# (like "the seven testing principles") right down the middle, so the answer is
# split across two chunks and neither one is complete.
#
# Instead we cut at the document's SECTION HEADINGS (like "1.3", "2.2.2"). Each
# chunk then holds one whole topic. We merge tiny sections together up to
# `max_chars` so chunks aren't too small.
def chunk_text(text, max_chars=2800):
    """Split text into chunks that follow the document's sections."""
    text = " ".join(text.split())        # tidy up the messy whitespace from the PDF

    # Cut the text right BEFORE every heading like "1.3 Title" or "2.2.2 Title".
    sections = re.split(r"(?=\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Z][a-z])", text)

    chunks = []
    buffer = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # If a single section is bigger than max_chars, cut it down to size.
        pieces = ([section[i:i + max_chars] for i in range(0, len(section), max_chars)]
                  if len(section) > max_chars else [section])
        for piece in pieces:
            if len(buffer) + len(piece) < max_chars:
                buffer = (buffer + " " + piece).strip()   # keep filling the chunk
            else:
                if buffer:
                    chunks.append(buffer)                 # chunk is full, start a new one
                buffer = piece
    if buffer:
        chunks.append(buffer)
    return chunks


def build_chunks(documents):
    """Turn the list of documents into a flat list of chunks.

    Each chunk remembers which document it came from (so we can cite sources).
    Returns a list like:
        [{"source": "ISTQB...pdf", "text": "Testing principles ..."}, ...]
    """
    all_chunks = []
    for doc in documents:
        for piece in chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": piece})
    return all_chunks


# ============================================================================
# STEP 3 — EMBED: turn text into vectors (lists of numbers)
# ============================================================================
def load_embedding_model():
    """Load the small local embedding model (downloads it on the first run)."""
    print(f"Loading the embedding model '{EMBEDDING_MODEL_NAME}' ...")
    print("(The very first time, this downloads ~90 MB. After that it's instant.)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model ready.\n")
    return model


def embed_texts(model, texts):
    """Convert a list of strings into a NumPy array of vectors.

    If we pass 80 chunks, we get back 80 vectors of 384 numbers each, i.e. an
    array of shape (80, 384).
    """
    return model.encode(texts, show_progress_bar=True)


# ============================================================================
# STEP 4 — SAVE / LOAD the embeddings (our little "vector database")
# ============================================================================
# We store the chunks and their vectors in a plain JSON file. JSON is perfect
# for LEARNING: you can open embeddings.json and literally read what is inside.
#
# IN A REAL PROJECT you would not use a JSON file. You would store these vectors
# in a VECTOR DATABASE that can search millions of them quickly, for example:
#     - PostgreSQL with the "pgvector" extension
#     - MongoDB Atlas Vector Search
#     - Dedicated engines like Chroma, Qdrant, Pinecone or FAISS
# The idea is exactly the same as here (store vectors, search by similarity) —
# the database just makes it fast and scalable.
def build_embeddings_file(model):
    """Read the docs, chunk + embed them, and save everything to embeddings.json."""
    documents = load_documents(DOCS_DIR)
    print(f"Loaded {len(documents)} document(s) from the 'docs/' folder.")

    chunks = build_chunks(documents)
    print(f"Cut them into {len(chunks)} small chunks.")
    print("Embedding every chunk (this is the slow part — done only once):")

    vectors = embed_texts(model, [c["text"] for c in chunks])

    # Build a simple structure and write it to disk. NumPy vectors are turned
    # into plain Python lists with .tolist() so they fit in a JSON file.
    data = {
        "model": EMBEDDING_MODEL_NAME,
        "chunks": [
            {"source": c["source"], "text": c["text"], "vector": vec.tolist()}
            for c, vec in zip(chunks, vectors)
        ],
    }
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Saved {len(chunks)} embeddings to {os.path.basename(EMBEDDINGS_FILE)}.\n")

    return chunks, vectors


def load_embeddings_file():
    """Load chunks and their vectors back from embeddings.json."""
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = [{"source": c["source"], "text": c["text"]} for c in data["chunks"]]
    vectors = np.array([c["vector"] for c in data["chunks"]])  # back to NumPy
    return chunks, vectors


def get_embeddings(model):
    """Return (chunks, vectors): load them from disk, or build them the first time."""
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Found {os.path.basename(EMBEDDINGS_FILE)} — loading saved embeddings.\n")
        return load_embeddings_file()
    print("No saved embeddings yet — building them now.\n")
    return build_embeddings_file(model)


# ============================================================================
# STEP 5 — RETRIEVE: find the chunks closest to the question
# ============================================================================
# COSINE SIMILARITY measures how similar the *direction* of two vectors is.
#   - result close to 1.0  -> very similar meaning
#   - result close to 0.0  -> unrelated
# The formula is just: (A . B) / (|A| * |B|)
#   A . B   = the "dot product" (multiply matching numbers, then add them up)
#   |A|     = the length of vector A
# We write it out by hand here so you can SEE exactly what is happening.
def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    length_a = np.linalg.norm(vec_a)
    length_b = np.linalg.norm(vec_b)
    return dot_product / (length_a * length_b)


def retrieve(query, chunks, chunk_vectors, model, top_k=4):
    """Find the `top_k` chunks most relevant to the search query.

    Steps:
      1. embed the query into a vector
      2. compare it to EVERY chunk vector with cosine similarity
      3. sort by score and return the best few
    """
    # 1. Embed the query (note: encode expects a list, so we pass [query]
    #    and take the first result).
    query_vector = model.encode([query])[0]

    # 2. Score every chunk against the query.
    scored = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(query_vector, chunk_vector)
        scored.append({"score": score, "chunk": chunk})

    # 3. Sort from highest score to lowest, keep the top_k.
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


# ============================================================================
# PUT IT ALL TOGETHER — run this file directly to see RAG retrieval in action
# ============================================================================
def main():
    # Load the model (needed both to build embeddings and to embed questions).
    model = load_embedding_model()

    # Steps 1-4: get our chunks and their vectors (build once, then reuse).
    chunks, chunk_vectors = get_embeddings(model)

    # Show what a single chunk looks like, and what its embedding looks like.
    print("Example of one chunk:")
    print("-" * 70)
    print(chunks[0]["text"])
    print("-" * 70)
    print(f"\nEach chunk is a vector of {chunk_vectors.shape[1]} numbers.")
    print("First 8 numbers of the first chunk's embedding:")
    print(chunk_vectors[0][:8], "\n")

    # Step 5: retrieve.
    # This is the SAME topic we asked the model in Step 1. Notice we search the
    # document with a few KEY WORDS, not a long sentence. Retrieval works best
    # with focused terms; extra words like "According to the ISTQB Certified
    # Tester Foundation Level v4.0" would just match the cover and title pages.
    search_query = "seven testing principles"
    print(f"Searching the syllabus for: \"{search_query}\"\n")

    results = retrieve(search_query, chunks, chunk_vectors, model, top_k=4)

    print("TOP MATCHING CHUNKS (highest cosine similarity first):\n")
    for rank, result in enumerate(results, start=1):
        score = result["score"]
        chunk = result["chunk"]
        preview = chunk["text"].replace("\n", " ")[:300]
        print(f"[{rank}] score={score:.3f}  from {chunk['source']}")
        print(f"    {preview}...\n")

    print("Notice: the top chunk contains the actual list of testing principles,")
    print("found WITHOUT any LLM and WITHOUT any API call. That is 'Retrieval'.")
    print("\nNext: 3_rag_plus_llm.py feeds this retrieved text to the local")
    print("model, so it can finally answer the question correctly.")


if __name__ == "__main__":
    main()
