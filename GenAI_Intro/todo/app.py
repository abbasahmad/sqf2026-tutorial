"""
GenAI API — TO-DO (Python / Flask) — the learner version

Implement each exercise one by one. Follow exercise.md in this folder, and
compare your work against ../solution/app.py when you get stuck.

Exercise 1 (/ping) is already done for you as a worked example. Everything
after it is a stub with TODO instructions — fill them in.

Run:   python app.py        (server starts on http://localhost:3000)
"""

import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response

from openai import OpenAI
from anthropic import Anthropic
import tiktoken

# Load OPENAI_API_KEY / ANTHROPIC_API_KEY from the .env file in this folder.
load_dotenv()

app = Flask(__name__)

# The clients automatically read their API keys from the environment.
openai = OpenAI()
claude = Anthropic()

# ── MODEL NAMES ───────────────────────────────────────────────────────────────
# Use a model your API key can access.
GPT_MODEL = "gpt-4o"
REASONING_MODEL = "o3-mini"        # any OpenAI reasoning model (o1, o3, o3-mini, ...)
CLAUDE_MODEL = "claude-sonnet-4-6"
IMAGE_MODEL = "gpt-image-2"


# ── HELPER ────────────────────────────────────────────────────────────────────
# You will need this in exercises 6 and 7: some models wrap their HTML output in
# markdown code fences (```html ... ```) even when told not to. This strips them.
def strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.rstrip().endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


# ── EXERCISE 1 — Ping / Pong (DONE — your worked example) ─────────────────────
# The simplest possible route. Build and test this first to confirm Flask works
# before adding any AI. A @app.get("/ping") handler returns plain text.
#
# Test:  open http://localhost:3000/ping  ->  pong
@app.get("/ping")
def ping():
    print("\n[/ping] -> pong")
    return "pong"


# ── EXERCISE 2 — Token counting (GET /tokens) ─────────────────────────────────
# Tokens are the units models read/write — ~4 characters each in English.
# Counting them before a call estimates cost and checks the prompt fits.
# tiktoken is OpenAI's tokeniser, so the count is exact with no API call.
#
# TODO: Add GET /tokens?prompt=...
#   1. Read request.args.get("prompt")
#   2. enc = tiktoken.encoding_for_model("gpt-4o")
#   3. token_ids = enc.encode(prompt)
#   4. Return jsonify(prompt=prompt, token_count=len(token_ids))
#
# Test:  /tokens?prompt=Hello+world   ->   {"prompt": "Hello world", "token_count": 2}


# ── EXERCISE 3 — First GPT call (GET /ask) ────────────────────────────────────
# Send the prompt to GPT-4o and return its reply as plain text.
#
#   completion = openai.chat.completions.create(
#       model=GPT_MODEL,
#       messages=[{"role": "user", "content": prompt}],
#   )
#   reply = completion.choices[0].message.content
#
# Bonus: log completion.usage (prompt_tokens / completion_tokens / total_tokens).
#
# TODO: Add GET /ask?prompt=...  returning the reply as plain text.
# Test:  /ask?prompt=What+is+the+capital+of+France


# ── EXERCISE 4 — Temperature (GET /temperature) ───────────────────────────────
# Temperature (0–2) controls randomness: 0 = deterministic, 1 = default,
# 2 = very creative. Read it carefully — temp=0 is valid, so default ONLY when
# the value is missing (None), not when it is falsy.
#
# TODO: Add GET /temperature?prompt=...&temp=...
#   1. temp_arg = request.args.get("temp"); temp = float(temp_arg) if temp_arg is not None else 1.0
#   2. Call openai.chat.completions.create() with model=GPT_MODEL,
#      temperature=temp, max_tokens=1000, messages=[{"role": "user", "content": prompt}]
#   3. Return jsonify(temperature=temp, prompt=prompt, response=reply)
#
# Test:  /temperature?prompt=Give+me+one+word&temp=0   (run 3x -> same answer?)
# Then:  /temperature?prompt=Give+me+one+word&temp=2   (run 3x -> more varied?)
#
# Key finding: temperature only works on standard models. Reasoning models use
# reasoning.effort instead — see Exercise 4b.


# ── EXERCISE 4b — Reasoning model (GET /gpt/reasoning) ────────────────────────
# Reasoning models think internally before answering. No temperature — instead
# set reasoning.effort ('low' | 'medium' | 'high'). They use the Responses API:
#
#   response = openai.responses.create(
#       model=REASONING_MODEL,
#       reasoning={"effort": effort, "summary": "auto"},
#       input=[{"role": "user", "content": prompt}],
#   )
#   reply = response.output_text          # note: no .choices[0] here
#
# The reasoning summary lives in response.output — find the item whose
# type == "reasoning", then read item.summary[0].text (if present).
#
# NOTE: "summary": "auto" requires a VERIFIED OpenAI organization
# (https://platform.openai.com/settings/organization/general). If yours isn't
# verified you'll get a 400 BadRequestError on reasoning.summary — just drop the
# "summary" key (reasoning={"effort": effort}). The solution wraps the call in
# try / except BadRequestError to fall back automatically.
#
# TODO: Add GET /gpt/reasoning?prompt=...&effort=...   (default effort "medium")
#   Return jsonify(model=REASONING_MODEL, effort=effort, prompt=prompt,
#                  response=reply, reasoning_summary=reasoning_summary)
#
# Test:  /gpt/reasoning?prompt=I+speak+without+a+mouth,+hear+without+ears.+I+have+no+body+but+come+alive+with+the+wind.+What+am+I&effort=low   (then effort=high)


# ── EXERCISE 5 — First Claude call (GET /claude) ──────────────────────────────
# Anthropic differs from OpenAI:
#   - method is claude.messages.create()
#   - max_tokens is REQUIRED
#   - the reply lives at message.content[0].text
#
#   message = claude.messages.create(
#       model=CLAUDE_MODEL,
#       max_tokens=1024,
#       messages=[{"role": "user", "content": prompt}],
#   )
#   reply = message.content[0].text
#
# TODO: Add GET /claude?prompt=...  returning the reply as plain text.
# Test:  /claude?prompt=What+is+the+capital+of+France


# ── EXERCISE 6 — GPT HTML page (GET /gpt/html) ────────────────────────────────
# A system message sets the model's role. Ask GPT-4o to act as a web developer
# and output a full HTML page. Always run the output through strip_code_fences().
#
#   user_prompt = (f"Create a complete, self-contained, single-page HTML website "
#                  f"about: {subject}. Output only raw HTML starting with <!DOCTYPE html>. "
#                  "Include inline CSS. Make it visually appealing.")
#   completion = openai.chat.completions.create(
#       model=GPT_MODEL,
#       messages=[
#           {"role": "system", "content": "You are an expert web developer. Output only valid, complete HTML — nothing else."},
#           {"role": "user", "content": user_prompt},
#       ],
#   )
#   html = strip_code_fences(completion.choices[0].message.content)
#   return Response(html, mimetype="text/html")
#
# TODO: Add GET /gpt/html?subject=...
# Test:  /gpt/html?subject=the+solar+system   -> a rendered HTML page


# ── EXERCISE 7 — Claude HTML page (GET /claude/html) ──────────────────────────
# Same task as Exercise 6, but with Claude. With Anthropic, `system` is a
# TOP-LEVEL field (not inside messages[]):
#
#   message = claude.messages.create(
#       model=CLAUDE_MODEL,
#       max_tokens=10000,
#       system="You are an expert web developer. Output only valid, complete HTML — nothing else.",
#       messages=[{"role": "user", "content": user_prompt}],
#   )
#   html = strip_code_fences(message.content[0].text)
#
# TODO: Add GET /claude/html?subject=...   then compare with /gpt/html.


# ── EXERCISE 8 — Image generation (GET /gpt/image) ────────────────────────────
# openai.images.generate() with gpt-image-2 returns the image as base64 data
# (data[0].b64_json), NOT a URL:
#
#   result = openai.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024")
#   b64 = result.data[0].b64_json
#
# Embed that base64 directly in an <img> tag as a data URI and return it as HTML:
#   <img src="data:image/png;base64,{b64}" />
# (see solution for a ready-made template).
#
# TODO: Add GET /gpt/image?prompt=...
# Test:  /gpt/image?prompt=A+cat+astronaut+on+the+moon


# ── EXERCISE 9 — POST route (POST /gpt/ask) ───────────────────────────────────
# Real APIs use POST: the prompt travels in the JSON BODY, not the URL.
# Flask reads it with request.get_json(). Read prompt from body["prompt"].
#
# TODO: Add POST /gpt/ask  -> jsonify(prompt=prompt, response=reply)
# Test (Postman / curl): POST http://localhost:3000/gpt/ask  with JSON body
#   {"prompt": "What is the capital of France?"}


# ── EXERCISE 10 — Multi-turn chat (POST /gpt/chat) ────────────────────────────
# The model is stateless — it has no memory. The client keeps the full messages
# array and re-sends it every turn. You pass it straight to GPT-4o, append the
# reply, and return the updated array.
#
#   messages = request.get_json()["messages"]          # validate it is a non-empty list
#   completion = openai.chat.completions.create(model=GPT_MODEL, messages=messages)
#   reply = completion.choices[0].message.content
#   updated = messages + [{"role": "assistant", "content": reply}]
#   return jsonify(response=reply, messages=updated)
#
# TODO: Add POST /gpt/chat
# Test: send turn 1, then paste the returned `messages` into turn 2 with a follow-up.


# ── START SERVER ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Server running on http://localhost:3000")
    app.run(port=3000, debug=True)
