"""
GenAI API — SOLUTION (Python / Flask)

A progressive lab that walks you through using the OpenAI and Anthropic APIs —
from a plain /ping route, to counting tokens, to a stateful multi-turn chatbot.

This is the completed version. Compare it against todo/app.py as you work
through exercise.md.

Run:   python app.py        (server starts on http://localhost:3000)
"""

import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response

from openai import OpenAI, BadRequestError
from anthropic import Anthropic
import tiktoken

# Load OPENAI_API_KEY / ANTHROPIC_API_KEY from the .env file in this folder.
load_dotenv()

app = Flask(__name__)

# The clients automatically pick up their API keys from the environment
# (OPENAI_API_KEY and ANTHROPIC_API_KEY), loaded above from .env.
openai = OpenAI()
claude = Anthropic()

# ── MODEL NAMES ───────────────────────────────────────────────────────────────
# Kept in one place so they are easy to swap. Use a model your API key can access.
GPT_MODEL = "gpt-4o"            # standard chat model — supports temperature
REASONING_MODEL = "o3-mini"    # any OpenAI reasoning model (o1, o3, o3-mini, ...)
CLAUDE_MODEL = "claude-sonnet-4-6"  # current Claude Sonnet
IMAGE_MODEL = "gpt-image-2"       # image generation


# ── HELPER ────────────────────────────────────────────────────────────────────
def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```html ... ```) that models sometimes add
    around their output even when instructed not to."""
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.rstrip().endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


# ── EXERCISE 1 — Ping / Pong ──────────────────────────────────────────────────
# The simplest possible route: prove the server is wired up before adding any AI.
# GET /ping  →  "pong"
@app.get("/ping")
def ping():
    print("\n[/ping] -> pong")
    return "pong"


# ── EXERCISE 2 — Token counting ───────────────────────────────────────────────
# Tokens are the units models use to read and generate text — roughly 4
# characters each in English. Counting them before a request lets you estimate
# cost and check the prompt fits in the model's context window.
# We use tiktoken — OpenAI's own tokeniser — so the count is exact, no API call.
@app.get("/tokens")
def tokens():
    prompt = (request.args.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Missing 'prompt' query parameter."), 400

    enc = tiktoken.encoding_for_model("gpt-4o")
    token_ids = enc.encode(prompt)
    print(f'[/tokens] "{prompt}" -> {len(token_ids)} tokens')

    return jsonify(prompt=prompt, token_count=len(token_ids))


# ── EXERCISE 3 — First GPT call (GET /ask) ────────────────────────────────────
# Send the prompt to GPT-4o and return its reply as plain text.
# The reply lives at completion.choices[0].message.content.
@app.get("/ask")
def ask():
    prompt = (request.args.get("prompt") or "").strip()
    if not prompt:
        return "Missing 'prompt' query parameter.", 400

    print(f'\n[/ask] -> Prompt sent to {GPT_MODEL}:\n  "{prompt}"')

    completion = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    reply = completion.choices[0].message.content
    usage = completion.usage
    print(f'[/ask] <- Response:\n  "{reply}"')
    print(f"[/ask] tokens — in: {usage.prompt_tokens} | out: {usage.completion_tokens} | total: {usage.total_tokens}")

    return reply


# ── EXERCISE 4 — Temperature (GET /temperature) ───────────────────────────────
# Temperature (0–2) controls randomness:
#   0 = deterministic, 1 = balanced (default), 2 = highly creative.
# Run the same URL a few times at temp=0 then temp=2 and compare.
@app.get("/temperature")
def temperature():
    prompt = (request.args.get("prompt") or "").strip()
    # Read temp carefully: temp=0 is valid, so check for None rather than falsiness.
    temp_arg = request.args.get("temp")
    temp = float(temp_arg) if temp_arg is not None else 1.0

    if not prompt:
        return jsonify(error="Missing 'prompt' query parameter."), 400

    print(f'\n[/temperature] -> {GPT_MODEL} (temperature={temp}):\n  "{prompt}"')

    completion = openai.chat.completions.create(
        model=GPT_MODEL,
        temperature=temp,
        max_tokens=1000,  # cap output so high temperatures can't run away
        messages=[{"role": "user", "content": prompt}],
    )

    reply = completion.choices[0].message.content
    print(f'[/temperature] <- "{reply}"')

    return jsonify(temperature=temp, prompt=prompt, response=reply)

    # KEY FINDING: temperature only works on standard models like gpt-4o.
    # Reasoning models (o1, o3, ...) do NOT support temperature — instead you
    # control how hard they think with reasoning.effort. See /gpt/reasoning.


# ── EXERCISE 4b — Reasoning model (GET /gpt/reasoning) ────────────────────────
# Reasoning models think internally before answering. You cannot set temperature;
# instead you set reasoning.effort ('low' | 'medium' | 'high').
# They use the Responses API: openai.responses.create(). The reply is at
# response.output_text (no .choices[0] nesting). summary='auto' returns a
# summary of the model's internal reasoning.
@app.get("/gpt/reasoning")
def gpt_reasoning():
    prompt = (request.args.get("prompt") or "").strip()
    effort = request.args.get("effort") or "medium"
    if not prompt:
        return jsonify(error="Missing 'prompt' query parameter."), 400

    print(f'\n[/gpt/reasoning] -> {REASONING_MODEL} (effort={effort}):\n  "{prompt}"')

    # summary="auto" returns a readable summary of the model's internal reasoning,
    # but it requires a VERIFIED OpenAI organization:
    #   https://platform.openai.com/settings/organization/general
    # If the org isn't verified, OpenAI rejects the request — so we catch that and
    # retry without the summary, and the route still works (reasoning_summary=None).
    try:
        response = openai.responses.create(
            model=REASONING_MODEL,
            reasoning={"effort": effort, "summary": "auto"},
            input=[{"role": "user", "content": prompt}],
        )
    except BadRequestError:
        print("[/gpt/reasoning] reasoning summary unavailable (org not verified) — retrying without it")
        response = openai.responses.create(
            model=REASONING_MODEL,
            reasoning={"effort": effort},
            input=[{"role": "user", "content": prompt}],
        )

    reply = response.output_text

    # response.output is a list of items (message, reasoning, ...). Find the
    # 'reasoning' item to read the model's reasoning summary, if present.
    reasoning_summary = None
    for item in response.output:
        if getattr(item, "type", None) == "reasoning" and getattr(item, "summary", None):
            reasoning_summary = item.summary[0].text
            break

    print(f'[/gpt/reasoning] <- "{reply}"')
    if reasoning_summary:
        print(f'[/gpt/reasoning] reasoning summary: "{reasoning_summary[:200]}..."')

    return jsonify(
        model=REASONING_MODEL,
        effort=effort,
        prompt=prompt,
        response=reply,
        reasoning_summary=reasoning_summary,
    )


# ── EXERCISE 5 — First Claude call (GET /claude) ──────────────────────────────
# Anthropic's API differs from OpenAI:
#   - method is claude.messages.create()
#   - max_tokens is REQUIRED
#   - the reply lives at message.content[0].text
@app.get("/claude")
def claude_ask():
    prompt = (request.args.get("prompt") or "").strip()
    if not prompt:
        return "Missing 'prompt' query parameter.", 400

    print(f'\n[/claude] -> Prompt sent to Claude:\n  "{prompt}"')

    message = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    reply = message.content[0].text
    usage = message.usage
    print(f'[/claude] <- "{reply}"')
    print(f"[/claude] tokens — in: {usage.input_tokens} | out: {usage.output_tokens}")

    return reply


# ── EXERCISE 6 — GPT HTML page (GET /gpt/html) ────────────────────────────────
# A system message sets the model's role for the whole conversation. Ask GPT-4o
# to act as a web developer and output a complete HTML page. Always run the
# output through strip_code_fences() — models sometimes wrap HTML in fences.
@app.get("/gpt/html")
def gpt_html():
    subject = (request.args.get("subject") or "").strip()
    if not subject:
        return "Missing 'subject' query parameter.", 400

    user_prompt = (
        f"Create a complete, self-contained, single-page HTML website about: {subject}. "
        "Output only raw HTML starting with <!DOCTYPE html>. Include inline CSS. "
        "Make it visually appealing."
    )

    print(f'\n[/gpt/html] -> {GPT_MODEL}:\n  "{user_prompt}"')

    completion = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert web developer. Output only valid, complete HTML — nothing else."},
            {"role": "user", "content": user_prompt},
        ],
    )

    html = strip_code_fences(completion.choices[0].message.content)
    print(f"[/gpt/html] <- {len(html)} characters of HTML")
    return Response(html, mimetype="text/html")


# ── EXERCISE 7 — Claude HTML page (GET /claude/html) ──────────────────────────
# Same task as Exercise 6 but using Claude. With Anthropic, `system` is a
# TOP-LEVEL field, not inside messages[]. Compare the result with /gpt/html.
@app.get("/claude/html")
def claude_html():
    subject = (request.args.get("subject") or "").strip()
    if not subject:
        return "Missing 'subject' query parameter.", 400

    user_prompt = (
        f"Create a complete, self-contained, single-page HTML website about: {subject}. "
        "Output only raw HTML starting with <!DOCTYPE html>. Include inline CSS. "
        "Make it visually appealing."
    )

    print(f'\n[/claude/html] -> Claude:\n  "{user_prompt}"')

    message = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=10000,
        system="You are an expert web developer. Output only valid, complete HTML — nothing else.",
        messages=[{"role": "user", "content": user_prompt}],
    )

    html = strip_code_fences(message.content[0].text)
    print(f"[/claude/html] <- {len(html)} characters of HTML")
    return Response(html, mimetype="text/html")


# ── EXERCISE 8 — Image generation (GET /gpt/image) ────────────────────────────
# openai.images.generate() with the gpt-image-2 model returns the image as
# base64 data (result.data[0].b64_json), NOT a temporary URL. We embed that
# base64 straight into an <img> tag as a "data URI" so it renders in the browser.
# (DALL-E 3 used to return a URL — gpt-image models always return base64.)
@app.get("/gpt/image")
def gpt_image():
    prompt = (request.args.get("prompt") or "").strip()
    if not prompt:
        return "Missing 'prompt' query parameter.", 400

    print(f'\n[/gpt/image] -> {IMAGE_MODEL}:\n  "{prompt}"')

    result = openai.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",  # also: 1536x1024, 1024x1536, or "auto"
    )

    # gpt-image models return base64-encoded image data, not a URL.
    b64 = result.data[0].b64_json
    print(f"[/gpt/image] <- received {len(b64)} base64 chars of image data")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Generated Image</title>
  <style>
    body {{ margin: 0; background: #111; display: flex; flex-direction: column;
           align-items: center; justify-content: center; min-height: 100vh;
           color: #fff; font-family: sans-serif; }}
    img {{ max-width: 90vw; max-height: 90vh; border-radius: 8px; }}
    p {{ margin-top: 1rem; opacity: 0.6; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <img src="data:image/png;base64,{b64}" alt="{prompt}" />
  <p>{prompt}</p>
</body>
</html>"""
    return Response(html, mimetype="text/html")


# ── EXERCISE 9 — POST route (POST /gpt/ask) ───────────────────────────────────
# Real APIs use POST, not GET, for AI calls: the prompt travels in the JSON
# request BODY instead of the URL. Flask parses it with request.get_json().
@app.post("/gpt/ask")
def gpt_ask_post():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Missing 'prompt' in request body."), 400

    print(f'\n[POST /gpt/ask] -> {GPT_MODEL}:\n  "{prompt}"')

    completion = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    reply = completion.choices[0].message.content
    print(f'[POST /gpt/ask] <- "{reply}"')
    return jsonify(prompt=prompt, response=reply)


# ── EXERCISE 10 — Multi-turn chat (POST /gpt/chat) ────────────────────────────
# The model is stateless — it has NO memory. The client keeps the full messages
# array and re-sends it every turn. The server passes it to GPT-4o, appends the
# reply, and returns the updated array for the client to store and resend.
@app.post("/gpt/chat")
def gpt_chat():
    body = request.get_json(silent=True) or {}
    messages = body.get("messages")

    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify(error="'messages' must be a non-empty array."), 400

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    print(f'\n[POST /gpt/chat] -> last user message: "{last_user.get("content") if last_user else "(none)"}"')
    print(f"[POST /gpt/chat] sending {len(messages)} message(s) of history to {GPT_MODEL}")

    completion = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
    )

    reply = completion.choices[0].message.content
    print(f'[POST /gpt/chat] <- "{reply}"')

    updated = messages + [{"role": "assistant", "content": reply}]
    return jsonify(response=reply, messages=updated)


# ── START SERVER ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Server running on http://localhost:3000")
    app.run(port=3000, debug=True)
