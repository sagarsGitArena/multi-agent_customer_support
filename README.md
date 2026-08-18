# Multi-Agent Music Store Customer Support

A multi-agent customer support system for a music store (built on the Chinook
sample database), powered by a LangGraph agentic graph and served through a
Gradio chat UI. It answers catalog questions (albums, artists, tracks),
looks up a verified customer's own orders/invoices, remembers a customer's
stated music preferences across conversations, and rejects off-topic
requests.

## Overview

A router classifies each incoming message into `catalog` and/or `invoice`
intent(s). Catalog questions are answered by a hand-built ReAct loop
(LLM + tools) compiled as its own subgraph. Invoice/order questions require
identity verification first (a human-in-the-loop interrupt) and are then
answered by a second subgraph, hard-scoped to the verified customer's own
data. A preference-memory node extracts and persists explicit taste
statements ("I love jazz") so later catalog answers can reference them.
Off-topic messages ("what's the weather?") are rejected directly, without
invoking either agent.

## Architecture

```
START -> router -> load_memory
                       |
              dispatch_next_intent   (one conditional-edge function,
                       |              reused below by advance_intent)
     +---------+-------+-------+---------+
     |         |               |         |
 catalog   invoice,        invoice,    queue
 intent   unverified       verified    empty
     |         |               |         |
     v         v               v         v
catalog_agent  hitl_verify --> invoice_agent   create_memory
[subgraph]     (interrupt,     [subgraph]           |
    |          waits for ID;        |                |
    |          loops on itself      |                |
    |          until verified)      |                |
    |               |               |                |
    +---> advance_intent <----------+                |
          (pop queue, loop back                       |
           to dispatch_next_intent)                    v
                                                       END
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
- **catalog_agent** (`agents/catalog_agent.py`) — a hand-built ReAct loop
  (LLM bound to 5 catalog tools: search by artist/genre/title, track
  details) compiled as its own `StateGraph` and added to the parent graph
  as a single node. It's fully self-contained: its state is just
  `messages` + `preferences_context`, nothing about customer identity or
  verification. Deliberately out of scope for order/invoice questions.
- **hitl_verify** (`graph/build.py`) — pauses the graph via LangGraph's
  `interrupt()` and asks for a customer ID, email, or phone number before
  any invoice data is touched. The three aren't checked with equal rigor:
  a customer ID is trusted as given (a stub — confirms *a* value was
  supplied, not that it belongs to the person typing), while email and
  phone are actually looked up against the `Customer` table —
  case-insensitively for email, formatting-insensitively for phone
  (`db/database.py::find_customer_id_by_email` /
  `find_customer_id_by_phone`) — so an unmatched email or phone leaves
  the customer unverified and re-prompts, rather than silently passing.
  Stays a flat node (not a subgraph) — it's a single interrupt-and-branch
  step, not a multi-node loop worth encapsulating.
- **invoice_agent** (`agents/invoice_agent.py`) — a second ReAct loop, also
  compiled as its own subgraph, over 4 invoice tools. Unlike catalog, its
  state also takes `customer_id` as an input, because ownership enforcement
  needs it: the verified `customer_id` is force-substituted into tool
  calls, and results that don't belong to that customer are rejected
  before being shown.

Both agent subgraphs hide their internal `agent <-> tools` loop behind one
`done` exit — from the parent graph's perspective, `catalog_agent` and
`invoice_agent` are each a single opaque node, tested independently of the
rest of the graph (`tests/test_catalog_agent.py`,
`tests/test_invoice_agent.py`).

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
pip install -e .   # makes the customer_support package importable
```

## How to run

```bash
uv run python -m customer_support.ui.app
# or, with a plain venv:
python -m customer_support.ui.app
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

119 tests covering the database layer, catalog/invoice tools, JSON response
validity, and utility functions, plus:

- `test_catalog_agent.py` / `test_invoice_agent.py` — the two agent
  subgraphs invoked standalone (no parent graph, no checkpointer), covering
  the tool-call loop, the no-results path, and (invoice only) cross-customer
  ownership rejection.
- `test_graph_flow.py` — the full compiled graph end-to-end: a catalog-only
  turn that never touches the identity gate, invoice turns that interrupt
  for verification and resume via `Command(resume=...)` across all three
  identifiers, and the identity-capture/preference-persistence scenarios
  described below.
- `test_app.py` — `_parse_verification_reply`'s free-text parsing, and
  `send_message` surfacing every sub-agent's answer in a mixed-intent turn.
- `test_router.py` — intent classification with recent conversation
  context, including the off-topic-mid-conversation regression check.
- `test_database.py::TestConcurrentQueries` — parallel tool calls against
  the shared SQLite connection.

## Sample usage

```
You: do you have any Beatles albums? also what's the status of my last order?
Bot: I couldn't find any albums by The Beatles in our catalog. As for the
     status of your last order, that's handled separately -- I can't
     provide that here. Please verify your identity first: can you share
     your customer ID, the email, or the phone number on your account?

You: my number is +33-03-80-73-66-99
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
- `hitl_verify`'s customer-ID path is still a stub (`bool(customer_id)`)
  — it confirms *a* ID was given, not that it belongs to the person
  typing. The email and phone paths are real lookups (a matching account
  is required to verify), but none of the three checks a password or
  second factor — not a real auth system.
- Abandoning a conversation mid-verification (e.g. hitting "New
  Conversation" instead of replying) leaves that thread's paused
  checkpoint in memory indefinitely — harmless for a demo, but would need
  explicit cleanup/expiry in a long-running production deployment.

## Bugs found and fixed during testing

The automated test suite catches regressions in individual tools and
nodes, but several real bugs only surfaced through actual multi-turn
conversation testing — most involved cross-turn state or concurrency,
which single-call unit tests don't exercise. Each has a regression test
in `tests/` guarding against recurrence.

- **Mixed-intent turns silently dropped one agent's answer.** A question
  touching both catalog and invoice runs both subgraphs in sequence, each
  producing its own final answer — but `send_message` (`ui/app.py`) only
  ever showed `messages[-1]`, the last one processed (catalog), dropping
  the first (invoice) without any error. Fixed by collecting every new
  `AIMessage` produced during the turn, not just the last.
- **Concurrent tool calls corrupted the database connection.**
  LangGraph's `ToolNode` runs multiple tool calls from one message in
  real parallel threads (e.g. checking two saved-preference genres at
  once) — but the DB is a single shared SQLite `:memory:` connection
  (`StaticPool`, required to keep the in-memory data alive at all across
  threads), which isn't safe for simultaneous access. Two threads
  querying at once corrupted each other's cursor state, raising
  `IndexError`. Fixed with a `threading.Lock()` around every function
  that touches the shared connection (`db/database.py`).
- **Identity stated in the very first message was ignored.** "my
  customer id is 43, where's my order?" always triggered a redundant
  verification prompt, because `hitl_verify_node` interrupted
  unconditionally without checking whether the triggering message
  already contained an identifier. Fixed with a pre-check against the
  triggering message before ever asking (`graph/build.py`).
- **A bare preference reply was rejected as off-topic.** "I love jazz,"
  replying to "what are you interested in?", was misclassified because
  the router only ever saw the single latest message, with no way to
  know it was answering a question. Fixed by including recent
  conversation history in classification (`agents/router.py`) — the
  system prompt is explicit that history is for interpreting the latest
  message only, so an unrelated new message stays off-topic regardless
  of what came before.
- **Identity was never captured outside the invoice flow.** `customer_id`
  was only ever set by `hitl_verify_node`, which only runs for unverified
  invoice questions — so a catalog-only conversation had no way to ever
  establish who the customer was, even if they volunteered their ID.
  Stated preferences could never be saved, and previously-saved ones
  could never be fetched, in a conversation that never asked an invoice
  question. Fixed by moving opportunistic identity capture into
  `load_memory_node` (`agents/memory.py`), which runs on every turn
  regardless of intent — factored into a shared `identity.py` module to
  avoid a circular import between `graph/build.py` and `agents/memory.py`.
- **Catalog recommendations inferred an unstated "taste" from purchase
  history.** Because `catalog_agent` receives the full shared
  conversation, it could read an earlier invoice answer's purchase list
  off the transcript and assert the customer's "diverse taste" as fact —
  never something they actually said, and a different mechanism entirely
  from the explicit-statement-only preference system. Fixed with an
  explicit grounding rule in `CATALOG_SYSTEM_PROMPT`
  (`agents/catalog_agent.py`) restricting personalization to
  `preferences_context` only.
- **A mixed-intent answer contradicted itself.** `invoice_agent`'s prompt
  told it to proactively decline any catalog part of the question ("I
  can't help with albums..."), which made sense when its answer was
  shown alone — but once mixed-intent answers are joined into one reply
  (see the first bug above), that disclaimer sat right next to
  `catalog_agent`'s own correct answer to the exact question it just
  said it couldn't help with. Fixed by telling `invoice_agent` to
  silently ignore the catalog part rather than comment on it
  (`agents/invoice_agent.py`), since a separate answer to it is always
  generated anyway.
