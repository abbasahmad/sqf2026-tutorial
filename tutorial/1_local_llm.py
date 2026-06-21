"""
==============================================================================
 STEP 1 — Run a local LLM (Llama 3B) and make an API call to it
==============================================================================

Let's start with the AI model itself: a real Large Language Model (LLM) that
runs ENTIRELY on your own laptop. No cloud, no API key, no data leaving your
machine.

We will ask it a precise ISTQB question. Keep an eye on the answer: the model
only knows what it learned during training, so its answer about the official
syllabus may be INCOMPLETE or WRONG. That is exactly the problem we will fix in
the next steps with RAG (giving the model the real document to read).

We use a free tool called OLLAMA. Think of Ollama as a little program that:
    - downloads a ready-to-use model for you, and
    - runs it as a small local server on your machine.

Your Python code then talks to that server by making an "API call" -- exactly
like calling a cloud AI, except the server is on localhost (your own computer).

------------------------------------------------------------------------------
 ONE-TIME SETUP (do this before running this file)
------------------------------------------------------------------------------
 1. Install Ollama (free): https://ollama.com/download
    - macOS / Windows: download and run the installer.
    - After installing, Ollama runs quietly in the background.

 2. Download the Llama 3B model (this is the ~2 GB "ready-to-use" model).
    Open a terminal and run:

        ollama pull llama3.2:3b

 3. Install the Python package that talks to Ollama:

        pip install ollama

------------------------------------------------------------------------------
 Then run this file:

        python 1_local_llm.py
==============================================================================
"""

import ollama   # the small Python library that sends requests to local Ollama


# The model we downloaded with "ollama pull llama3.2:3b".
# 3b = 3 billion parameters: small enough for a laptop, smart enough to be useful.
MODEL_NAME = "llama3.2:3b"


def ask_llm(prompt, temperature=0.4):
    """Send one prompt to the local model and return its text answer.

    This is the "API call". Under the hood, the `ollama` library sends an HTTP
    request to the Ollama server running on your machine (at localhost:11434)
    and waits for the model's reply.

    `temperature` controls creativity:
        - low  (0.0 - 0.5) -> focused, consistent, factual  (good for testing)
        - high (0.6 - 1.0) -> more varied and creative
    """
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            # "messages" is a conversation. Each message has a role:
            #   - "system" : instructions that set the model's behaviour
            #   - "user"   : what we are asking
            {"role": "system", "content": "You are a helpful software testing assistant."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature},
    )
    # The reply text lives inside the response object.
    return response["message"]["content"]


def main():
    print(f"Talking to the local model '{MODEL_NAME}' via Ollama.\n")
    print("(If you get a connection error, make sure Ollama is installed and")
    print(" that you ran:  ollama pull llama3.2:3b )\n")

    # A precise question, with no documents and no retrieval -- just the model.
    question = ("According to the ISTQB® Certified Tester Foundation Level v4.0, "
                "what are the seven testing principles?")
    print(f"PROMPT: {question}\n")
    print("Thinking... (the first call may take a few seconds)\n")

    answer = ask_llm(question)

    print("MODEL ANSWER:")
    print("-" * 70)
    print(answer)
    print("-" * 70)

    print("\nThis answer came from a model running on YOUR laptop.")
    print("But notice: the model answered from its own general memory. It never")
    print("saw the real ISTQB syllabus, so this answer may be incomplete or")
    print("simply WRONG. We have no way to trust it or trace it to a source.")
    print("\nNext: 2_rag.py shows how to pull the real document and find the")
    print("exact part that answers this question. Then 3_rag_plus_llm.py feeds")
    print("that part to the model so it finally gives a correct, grounded answer.")


if __name__ == "__main__":
    main()
