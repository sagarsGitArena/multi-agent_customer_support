# Multi-Agent Music Store Customer Support

A multi-agent customer support system for a music store (built on the Chinook
sample database), powered by a LangGraph agentic graph and served through a
Gradio chat UI. It answers catalog questions (albums, artists, tracks),
looks up a verified customer's own orders/invoices, remembers a customer's
stated music preferences across conversations, and rejects off-topic
requests.

## Overview

A router classifies each incoming message into `catalog` and/or `invoice`
intent(s). Catalog questions are answered by a hand-built tool-calling loop
against the music catalog. Invoice/order questions require identity
verification first (a human-in-the-loop interrupt) and are then answered by
a second tool-calling loop that's hard-scoped to the verified customer's own
data. A preference-memory node extracts and persists explicit taste
statements ("I love jazz") so later catalog answers can reference them.
Off-topic messages ("what's the weather?") are rejected directly, without
invoking either agent.

## Architecture

```
START -> router -> load_memory -> dispatch
                                     |
                 +-------------------+-------------------+
                 |                   |                   |
           catalog_agent       hitl_verify           (queue empty)
           (+ catalog_tools     (interrupt,               |
            tool loop)          waits for ID)         create_memory
                 |                   |                    |
                 +--> advance_intent <--- invoice_agent    |
                      (pop queue,     (+ invoice_tools     |
                       loop back to    tool loop)          |
                       dispatch)           |                |
                                           +---> advance_intent
                                                     |
                                              create_memory -> END
```

- **router** (`agents/router.py`) — classifies intent(s) via structured LLM
  output. Mixed queries ("do you have jazz albums, and what's my last
  order?") are handled invoice-first, then catalog. Off-topic messages get
  an empty intent list and a direct rejection reply, without calling any
  sub-agent.
- **load_memory / create_memory** (`agents/memory.py`) — read a verified
  customer's saved preferences into the catalog agent's prompt context at
  the start of a turn; extract any *new* explicit preference statements
  from the recent conversation and merge them (never overwrite) at the end
  of a turn, via LangGraph's `InMemoryStore`, keyed by `customer_id`.
- **catalog_agent** (`agents/catalog_agent.py`) — a hand-built tool-calling
  loop over 5 catalog tools (search by artist/genre/title, track details).
  Deliberately out of scope for order/invoice questions.
- **hitl_verify** (`graph/build.py`) — pauses the graph via LangGraph's
  `interrupt()` and asks for a customer ID + last name before any invoice
  data is touched.
- **invoice_agent** (`agents/invoice_agent.py`) — a second tool-calling loop
  over 4 invoice tools. Ownership is enforced in code (not just prompted):
  the verified `customer_id` is force-substituted into tool calls, and
  results that don't belong to that customer are rejected before being
  shown.

State persistence uses two independent in-memory stores: a `MemorySaver`
checkpointer scopes **conversation history** per `thread_id` (one per
browser session), and an `InMemoryStore` scopes **preference memory** per
real `customer_id` (shared across sessions for the same customer, by
design — that's what makes "remember me next time" work). Both are
process-local and reset on restart; there is no persistent database backing
either.

## Setup

Requires Python 3.12+ and an OpenAI API key.

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Using [uv](https://github.com/astral-sh/uv) (recommended — this is how the
project's own lockfile is maintained):

```bash
uv sync
```

Or plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
uv run python app.py
# or, with a plain venv:
python app.py
```

Opens a Gradio chat UI at `http://localhost:7860` (configurable via `PORT`
in `.env`).

### Docker

```bash
docker build -t multi-agent-music-store-customer-support .
docker run -p 7860:7860 --env-file .env multi-agent-music-store-customer-support
```

## How to test

```bash
uv run pytest tests/ -v
# or: pytest tests/ -v
```

72 tests covering the database layer, catalog/invoice tools, JSON response
validity, and utility functions.

## Sample usage

```
You: do you have any Beatles albums? also what's the status of my last order?
Bot: I couldn't find any albums by The Beatles in our catalog. As for the
     status of your last order, that's handled separately -- I can't
     provide that here. Please verify your identity first: can you share
     your customer ID and last name?

You: my customer id is 43, last name Mercier
Bot: Your last order, placed on June 6, 2025, included tracks by Led
     Zeppelin, totaling $8.91.

You: I love jazz, any recommendations?
Bot: Here are some jazz tracks you might enjoy: ...

You: [new message, later] what do you have in the catalog for me?
Bot: Based on your love of jazz, here are a few picks: ...
```

Use the "New Conversation" button to reset the chat and start a fresh
session (a new `thread_id` — conversation history is isolated per session;
a saved preference profile follows the *customer*, not the session, so it
reappears once the same customer verifies again in a new session).

## Known limitations (in-memory-only scope)

- All state (conversation checkpoints, preference profiles) lives in
  process memory and is lost on restart — there is no database-backed
  persistence layer.
- `hitl_verify`'s identity check is a stub (`bool(customer_id)`) — it
  confirms *a* customer ID was given, not that it belongs to the person
  typing. Not a real auth system.
- Abandoning a conversation mid-verification (e.g. hitting "New
  Conversation" instead of replying) leaves that thread's paused
  checkpoint in memory indefinitely — harmless for a demo, but would need
  explicit cleanup/expiry in a long-running production deployment.
