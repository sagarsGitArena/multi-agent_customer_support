"""
Router node for the catalog / invoice agentic graph.

Location: src/customer_support/agents/router.py

Extracts ALL intents present in the message, not just one — a query
can ask about catalog and invoice in the same turn. Preference
extraction is handled separately, by create_memory_node
(agents/memory.py).

Classifies the LATEST message, but includes a little recent history so
the model can resolve what that message refers to -- e.g. a bare "I
love jazz" replying to "what are you interested in?" carries no
catalog/invoice keyword on its own and was previously misclassified as
off-topic. History is context for interpretation only: the system
prompt is explicit that an unrelated new message (small talk, weather)
stays off-topic regardless of what was discussed earlier.
"""

import logging
from typing import Literal
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from customer_support.config import get_llm
from customer_support.graph.state import GraphState, format_state, recent_text_messages

logger = logging.getLogger(__name__)

# Only needs enough turns to resolve a reference or a reply to the
# assistant's last question -- not a full taste profile (that's
# create_memory_node's much wider HISTORY_WINDOW). Kept small
# deliberately so older, less relevant turns don't bias classification
# of an unrelated new message toward whatever topic came before it.
ROUTER_HISTORY_WINDOW = 6

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
    reasoning: str = Field(description="One sentence explaining the classification.")


ROUTER_SYSTEM_PROMPT = """You are the routing layer for a music store \
assistant. Classify the intent(s) of the customer's LATEST message only. \
Earlier turns, if shown, are given purely as context to help you \
understand what the latest message refers to -- e.g. a bare preference \
statement like "I love jazz" replying to "what are you interested in?", \
or "the second one" referring to something named earlier. They are not \
themselves a source of intent.

- "catalog": browsing music, artists, albums, tracks, availability, \
  asking for recommendations, or stating a music preference/taste.
- "invoice": anything about a specific purchase, order, receipt, refund, \
  or billing history.

A message can contain both — list every intent that applies, in the \
order the customer raised them. If the LATEST message is unrelated to \
both, leave intents empty -- this includes small talk, weather, or \
anything outside a music store's scope, EVEN IF earlier turns were \
about music or invoices. Earlier turns being in-scope does not make an \
unrelated new message in-scope. Do not answer the customer's question \
yourself — only classify."""

router_llm = get_llm().with_structured_output(IntentClassification)


def router_node(state: GraphState) -> dict:
    """Classify all intents in the latest customer message and return
    a partial state update. LangGraph merges this into the full state."""

    logger.info("router_node: state=\n%s", format_state(state))

    last_message = state["messages"][-1]
    content = getattr(last_message, "content", last_message)

    logger.info("router_node: classifying message: %r", content)

    history = recent_text_messages(state["messages"], limit=ROUTER_HISTORY_WINDOW)
    if not history:
        # Only happens if the latest message has no usable text content
        # (shouldn't occur via the UI, which rejects empty input) --
        # fall back to classifying it directly rather than sending the
        # LLM a system prompt with no user turn at all.
        history = [{"role": "user", "content": content}]

    result: IntentClassification = router_llm.invoke(
        [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}, *history]
    )

    # Mixed queries are handled invoice-first, then catalog.
    ordered_intents = sorted(result.intents, key=lambda i: 0 if i == "invoice" else 1)

    logger.info(
        "router_node: intents=%s reasoning=%s",
        ordered_intents,
        result.reasoning,
    )

    update: dict = {
        "intents": ordered_intents,
    }

    if not ordered_intents:
        logger.info("router_node: no in-scope intent detected, rejecting off-topic query")
        update["messages"] = [AIMessage(content=OFF_TOPIC_REJECTION)]

    return update