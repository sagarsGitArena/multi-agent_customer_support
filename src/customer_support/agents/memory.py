"""
Memory node for the customer support agentic graph.

Location: src/customer_support/agents/memory.py

load_memory_node reads a verified customer's saved music preferences
from the store and formats them for prompt injection. create_memory_node
runs after each turn, extracts any NEW explicit preference statements
from the recent conversation via structured output, and merges them
into the stored profile via set union -- existing preferences are never
removed or overwritten.

Both nodes use get_store() (langgraph.config) rather than a store
parameter on the node signature -- the documented, idiomatic way to
reach the store bound via graph.compile(store=...) from inside any
node function without changing its signature. (InjectedStore is a
different mechanism, for @tool functions run through ToolNode -- it
doesn't apply to plain StateGraph nodes like these.)
"""

import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_store
from pydantic import BaseModel, Field

from customer_support.config import get_llm
from customer_support.graph.state import GraphState, format_state

logger = logging.getLogger(__name__)

MEMORY_NAMESPACE = "memory_profile"
MEMORY_KEY = "user_memory"
HISTORY_WINDOW = 10


class UserProfile(BaseModel):
    """Storage schema. Always constructed in code from the verified
    customer_id -- never trust the model to fill this field itself,
    same principle as invoice_tools_node's handling of customer_id."""

    customer_id: str
    music_preferences: list[str] = Field(default_factory=list)


class ExtractedPreferences(BaseModel):
    """Narrower extraction-only schema. Only asks the model for what
    it can actually observe -- new preferences in this window -- not
    an identity field it has no business asserting."""

    music_preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit NEW music preferences the customer stated in this "
            "excerpt. Empty list if none. See system prompt for the "
            "explicit-statement-vs-question distinction."
        ),
    )


EXTRACTION_SYSTEM_PROMPT = """You are analyzing a short, recent excerpt \
of a conversation between a customer and a music store assistant. Your \
only job is to identify explicit, lasting music preference statements \
made by the CUSTOMER -- ignore the assistant's messages except as \
context.

Extract a preference ONLY when the customer explicitly states that they \
like, love, prefer, or are a fan of a genre, artist, or style, using \
language such as "I love...", "I like...", "I'm a fan of...", "I \
prefer...", "my favorite is...".

Do NOT extract a preference when the customer:
- asks a question ("Do you have any jazz albums?", "Is AC/DC in stock?")
- makes a one-off request ("play me some rock", "find me a jazz \
  playlist", "recommend something in the rock genre")
- merely mentions a genre or artist without a like/love/prefer statement

If the excerpt contains no explicit preference statements, return an \
empty list. When in doubt, do NOT extract -- it is better to miss a \
preference than to record something the customer never actually said \
they like.

Return each preference as a short string (a genre or artist name, e.g. \
"rock", "AC/DC", "jazz"), not a full sentence."""

extraction_llm = get_llm().with_structured_output(ExtractedPreferences)


def _format_preferences(music_preferences: list[str]) -> Optional[str]:
    if not music_preferences:
        return None
    return "Music Preferences: " + ", ".join(music_preferences)


def load_memory_node(state: GraphState) -> dict:
    """Reads the verified customer's saved preferences from the store
    and formats them for injection into the music agent's prompt. If
    the customer isn't verified yet this turn, there's nothing to look
    up -- return None rather than crash on a missing customer_id."""

    logger.info("load_memory_node: state=\n%s", format_state(state))

    customer_id = state.get("customer_id")
    if not customer_id:
        logger.info("load_memory_node: no verified customer_id yet, skipping")
        return {"preferences_context": None}

    store = get_store()
    item = store.get((MEMORY_NAMESPACE, customer_id), MEMORY_KEY)
    music_preferences = item.value.get("music_preferences", []) if item else []

    formatted = _format_preferences(music_preferences)
    logger.info(
        "load_memory_node: customer_id=%s loaded preferences=%s",
        customer_id,
        music_preferences,
    )
    return {"preferences_context": formatted}


def _recent_text_messages(messages: list, limit: int = HISTORY_WINDOW) -> list[dict]:
    """Filters to Human/AI messages with non-empty content FIRST, then
    takes the last `limit` -- not the other way around. Slicing the raw
    message list first risks pushing the very human statement we care
    about out of the window on tool-call-heavy turns (a multi-step
    catalog/invoice loop can inject many ToolMessages between two real
    conversational turns).

    Reconstructs plain role/content dicts rather than reusing the
    LangChain message objects: an AIMessage that still carries
    tool_calls (even alongside real content) would make this an
    invalid message sequence for the extraction call once its paired
    ToolMessages have been filtered out of the window. Stripping to
    content-only dicts sidesteps that entirely.
    """

    filtered = []
    for m in messages:
        content = getattr(m, "content", None)
        if not content or not isinstance(content, str) or not content.strip():
            continue
        if isinstance(m, HumanMessage):
            filtered.append({"role": "user", "content": content})
        elif isinstance(m, AIMessage):
            filtered.append({"role": "assistant", "content": content})
    return filtered[-limit:]


def create_memory_node(state: GraphState) -> dict:
    """Runs after a turn completes. Extracts any new explicit
    preferences from the recent conversation and merges them into the
    stored profile via set union. Never removes existing preferences.
    If nothing new was said, skips the write entirely -- both to avoid
    a pointless store call and, per spec, to guarantee an empty
    extraction can never wipe out a previously saved profile."""

    logger.info("create_memory_node: state=\n%s", format_state(state))

    customer_id = state.get("customer_id")
    if not customer_id:
        logger.info("create_memory_node: no verified customer_id, skipping")
        return {}

    recent = _recent_text_messages(state["messages"])
    if not recent:
        logger.info("create_memory_node: no text messages in window, skipping")
        return {}

    result: ExtractedPreferences = extraction_llm.invoke(
        [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}, *recent]
    )
    new_preferences = result.music_preferences
    logger.info("create_memory_node: extracted new preferences=%s", new_preferences)

    if not new_preferences:
        logger.info("create_memory_node: nothing new, skipping write")
        return {}

    store = get_store()
    namespace = (MEMORY_NAMESPACE, customer_id)
    existing_item = store.get(namespace, MEMORY_KEY)
    existing = existing_item.value.get("music_preferences", []) if existing_item else []

    # NOTE: exact-string dedup, so e.g. "AC/DC" and "ac/dc" would be
    # stored as two separate entries -- not handled here.
    merged = sorted(set(existing) | set(new_preferences))
    profile = UserProfile(customer_id=customer_id, music_preferences=merged)

    store.put(namespace, MEMORY_KEY, profile.model_dump())
    logger.info(
        "create_memory_node: customer_id=%s merged preferences=%s",
        customer_id,
        merged,
    )
    return {}
