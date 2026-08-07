"""
Invoice agent node for the music store agentic graph.

Location: src/customer_support/agents/invoice_agent.py

Uses your four real invoice tools directly, unmodified — same names,
docstrings, and schemas the LLM already sees. Ownership enforcement
happens entirely in invoice_tools_node, not by redefining the tools:

- get_customer_invoices / get_customer_purchased_tracks: take
  customer_id as an argument. Rather than trust whatever the model
  passes, this node overwrites it with the verified customer_id from
  state before calling.
- get_invoice_support_rep: its query already joins Customer and
  returns customer_id in the result, so ownership is checked AFTER
  calling it — if the returned customer_id doesn't match, the result
  is replaced with a rejection.
- get_invoice_line_items: its result has no customer info at all, so
  ownership is checked BEFORE calling it, against the customer's own
  invoice IDs (fetched via get_customer_invoices).
"""

import json
import logging
from typing import Literal
from langchain_core.messages import ToolMessage

from customer_support.config import get_llm
from customer_support.graph.state import GraphState, format_state
from customer_support.tools.invoice_tools import (
    get_customer_invoices,
    get_customer_purchased_tracks,
    get_invoice_support_rep,
    get_invoice_line_items,
)

logger = logging.getLogger(__name__)

INVOICE_TOOLS = [
    get_customer_invoices,
    get_customer_purchased_tracks,
    get_invoice_support_rep,
    get_invoice_line_items,
]

TOOLS_BY_NAME = {t.name: t for t in INVOICE_TOOLS}


def _invoice_system_prompt(customer_id: str) -> str:
    return f"""You are the invoice and order assistant for a music \
store. The customer's identity has already been verified — you do \
not need to ask for or confirm who they are again this turn. The \
verified customer's ID is {customer_id}; use this ID whenever a tool \
needs a customer_id. Call tools as needed before answering — don't \
guess at invoice details you haven't looked up. If a tool tells you \
information isn't available or doesn't belong to this customer, say \
you're unable to share that rather than guessing why. If asked about \
music, albums, or catalog availability, note that's handled \
separately and don't attempt to answer it here."""


invoice_llm = get_llm().bind_tools(INVOICE_TOOLS)


def invoice_agent_node(state: GraphState) -> dict:
    """Runs the invoice LLM with your real tools bound as-is. Assumes
    customer_verified is already True and customer_id is set — the
    graph's routing guarantees this node is never reached otherwise."""

    logger.info("invoice_agent_node: state=\n%s", format_state(state))

    system_message = {
        "role": "system",
        "content": _invoice_system_prompt(state["customer_id"]),
    }

    logger.info("invoice_agent_node: invoking invoice LLM, customer_id=%s", state["customer_id"])

    response = invoice_llm.invoke([system_message, *state["messages"]])

    tool_calls = getattr(response, "tool_calls", None) or []
    logger.info(
        "invoice_agent_node: response has %d tool call(s): %s",
        len(tool_calls),
        [call.get("name") for call in tool_calls],
    )

    return {"messages": [response]}


def _try_parse_json(raw: str):
    """Returns parsed JSON, or None if raw is a not_found()-style plain
    message rather than a JSON array."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _owned_invoice_ids(customer_id: str) -> set:
    raw = get_customer_invoices.invoke({"customer_id": customer_id})
    rows = _try_parse_json(raw)
    if not isinstance(rows, list):
        return set()
    return {str(row.get("invoice_id")) for row in rows if row.get("invoice_id") is not None}


def invoice_tools_node(state: GraphState) -> dict:
    """Executes whatever tool calls the agent just made, enforcing
    that the verified customer can only see their own data."""

    logger.info("invoice_tools_node: state=\n%s", format_state(state))

    customer_id = state["customer_id"]
    last_message = state["messages"][-1]
    tool_messages = []
    owned_invoice_ids = None  # computed lazily, only if a line-items call needs it

    logger.info(
        "invoice_tools_node: executing tool call(s): %s",
        [call.get("name") for call in last_message.tool_calls],
    )

    for call in last_message.tool_calls:
        name = call["name"]
        args = dict(call["args"])
        tool = TOOLS_BY_NAME.get(name)

        if tool is None:
            logger.warning("invoice_tools_node: unknown tool requested: %s", name)
            content = f"Unknown tool: {name}"

        elif name in ("get_customer_invoices", "get_customer_purchased_tracks"):
            # Never trust the model's customer_id — always use the verified one.
            args["customer_id"] = customer_id
            content = tool.invoke(args)

        elif name == "get_invoice_line_items":
            if owned_invoice_ids is None:
                owned_invoice_ids = _owned_invoice_ids(customer_id)
            if str(args.get("invoice_id")) not in owned_invoice_ids:
                logger.warning(
                    "invoice_tools_node: rejected get_invoice_line_items for invoice_id=%s, customer_id=%s (not owned)",
                    args.get("invoice_id"),
                    customer_id,
                )
                content = "That invoice does not belong to the verified customer."
            else:
                content = tool.invoke(args)

        elif name == "get_invoice_support_rep":
            content = tool.invoke(args)
            rows = _try_parse_json(content)
            row = rows[0] if isinstance(rows, list) and rows else rows
            if isinstance(row, dict) and str(row.get("customer_id")) != str(customer_id):
                logger.warning(
                    "invoice_tools_node: rejected get_invoice_support_rep result, owner customer_id=%s != verified customer_id=%s",
                    row.get("customer_id"),
                    customer_id,
                )
                content = "That invoice does not belong to the verified customer."

        else:
            logger.warning("invoice_tools_node: unhandled tool: %s", name)
            content = f"Unhandled tool: {name}"

        tool_messages.append(ToolMessage(content=str(content), tool_call_id=call["id"]))

    return {"messages": tool_messages}


def route_after_invoice_agent(state: GraphState) -> Literal["invoice_tools", "advance_intent"]:
    last_message = state["messages"][-1]
    decision = "invoice_tools" if getattr(last_message, "tool_calls", None) else "done"
    logger.info("route_after_invoice_agent: -> %s", decision)
    return decision


# --- Graph wiring (goes in graph/build.py) ---------------------------------
#
#   graph.add_node("invoice_agent", invoice_agent_node)
#   graph.add_node("invoice_tools", invoice_tools_node)
#
#   graph.add_conditional_edges(
#       "invoice_agent",
#       route_after_invoice_agent,
#       {"invoice_tools": "invoice_tools", "advance_intent": "advance_intent"},
#   )
#   graph.add_edge("invoice_tools", "invoice_agent")