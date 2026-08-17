"""
Shared graph state for the customer support agentic graph.

Location: src/customer_support/graph/state.py

`intents` is a queue, not a single value — a message can ask about
catalog AND invoice in the same turn ("do you have Beatles albums,
and what's the status of my last order?"). The graph processes one
intent at a time, popping the front of the list as each is handled.
"""

import pprint
from typing import Optional
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    ## List of  LangChain messages objects(HumanMessage, AIMessage, ToolMessage , etc)
    messages: Annotated[list, add_messages]

    session_id: str
    customer_id: Optional[str]
    customer_verified: bool

    intents: list  # queue of Literal["catalog", "invoice"], processed front to back
    preferences_context: Optional[str]  # formatted "Music Preferences: ..." string, or None


def format_state(state: "GraphState") -> str:
    """Pretty-formats graph state for logging. Messages render as
    compact `Type(content=...)` lines instead of pprint's default
    multi-field message repr, which is mostly noise in logs."""

    printable = dict(state)
    messages = printable.get("messages")
    if messages is not None:
        printable["messages"] = [_format_message(m) for m in messages]
    return pprint.pformat(printable, indent=2, width=100, sort_dicts=False)


def _format_message(m) -> str:
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        calls = ", ".join(
            f"{call.get('name')}({call.get('args')})" for call in tool_calls
        )
        return f"{type(m).__name__}(content={getattr(m, 'content', m)!r}, tool_calls=[{calls}])"
    return f"{type(m).__name__}(content={getattr(m, 'content', m)!r})"


def recent_text_messages(messages: list, limit: int) -> list[dict]:
    """Filters to Human/AI messages with non-empty content FIRST, then
    takes the last `limit` -- not the other way around. Slicing the raw
    message list first risks pushing the very human statement callers
    care about out of the window on tool-call-heavy turns (a
    multi-step catalog/invoice loop can inject many ToolMessages
    between two real conversational turns). Shared by router_node
    (recent context for intent classification) and create_memory_node
    (recent context for preference extraction).

    Reconstructs plain role/content dicts rather than reusing the
    LangChain message objects: an AIMessage that still carries
    tool_calls (even alongside real content) would make this an
    invalid message sequence for a plain-text LLM call once its paired
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