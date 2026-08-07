"""
Router node for the catalog / invoice agentic graph.

Location: src/customer_support/agents/router.py

Extracts ALL intents present in the message, not just one — a query
can ask about catalog and invoice in the same turn. Also flags an
explicit preference statement, if any (not a genre mention in a
request — see preference_signal description).
"""

import logging
from typing import Literal, Optional
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from customer_support.config import get_llm
from customer_support.graph.state import GraphState, format_state

logger = logging.getLogger(__name__)

OFF_TOPIC_REJECTION = (
    "I can only help with questions about our music catalog or your orders "
    "and invoices. Could you ask something related to one of those?"
)


class IntentClassification(BaseModel):
    intents: list[Literal["catalog", "invoice"]] = Field(
        description=(
            "All intents present in the message, in the order they should "
            "be handled. Most messages have exactly one. Include both only "
            "if the customer genuinely asked about two different things, "
            "e.g. catalog availability AND an order/invoice question."
        )
    )
    preference_signal: Optional[str] = Field(
        default=None,
        description=(
            "ONLY set this if the customer explicitly states a lasting "
            "preference using language like 'I like', 'I love', 'I prefer', "
            "'my favorite is', or similar. Do NOT set this just because a "
            "genre, artist, or product was mentioned in a request — "
            "'play me some rock' or 'find rock albums' is a one-off request, "
            "not a preference, and must be left null."
        ),
    )
    reasoning: str = Field(description="One sentence explaining the classification.")


ROUTER_SYSTEM_PROMPT = """You are the routing layer for a music store \
assistant. Read the customer's latest message and identify every intent \
present in it.

- "catalog": browsing music, artists, albums, tracks, availability, or \
  asking for recommendations.
- "invoice": anything about a specific purchase, order, receipt, refund, \
  or billing history.

A message can contain both — list every intent that applies, in the \
order the customer raised them. If the message is unrelated to both \
(e.g. small talk, weather, or anything outside a music store's scope), \
leave intents empty.

For preference_signal: only capture it when the customer explicitly \
states a lasting like/love/preference (e.g. "I love jazz", "I prefer \
email receipts"). A request that merely mentions a genre or product \
(e.g. "play me some rock", "do you have any jazz albums") is NOT a \
preference — leave preference_signal null in that case, even though \
the intent classification itself is unaffected. Do not answer the \
customer's question yourself — only classify."""

router_llm = get_llm().with_structured_output(IntentClassification)


def router_node(state: GraphState) -> dict:
    """Classify all intents in the latest customer message and return
    a partial state update. LangGraph merges this into the full state."""

    logger.info("router_node: state=\n%s", format_state(state))

    last_message = state["messages"][-1]
    content = getattr(last_message, "content", last_message)

    logger.info("router_node: classifying message: %r", content)

    result: IntentClassification = router_llm.invoke(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
    )

    # Mixed queries are handled invoice-first, then catalog.
    ordered_intents = sorted(result.intents, key=lambda i: 0 if i == "invoice" else 1)

    logger.info(
        "router_node: intents=%s preference_signal=%s reasoning=%s",
        ordered_intents,
        result.preference_signal,
        result.reasoning,
    )

    update: dict = {
        "intents": ordered_intents,
        "pending_preference_signal": result.preference_signal,
    }

    if not ordered_intents:
        logger.info("router_node: no in-scope intent detected, rejecting off-topic query")
        update["messages"] = [AIMessage(content=OFF_TOPIC_REJECTION)]

    return update