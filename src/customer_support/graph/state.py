"""
Shared graph state for the customer support agentic graph.

Location: src/customer_support/graph/state.py

`intents` is a queue, not a single value — a message can ask about
catalog AND invoice in the same turn ("do you have Beatles albums,
and what's the status of my last order?"). The graph processes one
intent at a time, popping the front of the list as each is handled.
"""

from typing import Literal, Optional
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

    session_id: str
    customer_id: Optional[str]
    customer_verified: bool

    intents: list  # queue of Literal["catalog", "invoice"], processed front to back
    pending_preference_signal: Optional[str]
    preferences: dict