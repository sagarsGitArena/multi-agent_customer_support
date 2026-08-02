"""
Graph assembly for the customer support agentic system.

Location: src/customer_support/graph/build.py

Handles mixed-intent turns: catalog is always answered; invoice is
gated on customer_verified. If unverified, hitl_verify calls
interrupt() — this pauses the graph and returns control to the
caller immediately, with everything generated so far (e.g. the
catalog answer, already appended to state["messages"]) intact. The
caller shows that plus the verification prompt in one turn, and
resumes the graph with the customer's next message via
Command(resume=...).
"""

import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from customer_support.graph.state import GraphState
from customer_support.agents.router import router_node
from customer_support.agents.catalog_agent import (
    catalog_agent_node,
    catalog_tools_node,
    route_after_catalog_agent,
)
from customer_support.agents.invoice_agent import (
    invoice_agent_node,
    invoice_tools_node,
    route_after_invoice_agent,
)

logger = logging.getLogger(__name__)


# --- Preferences: temporary inline stand-ins --------------------------------
# TODO: move to agents/memory.py once get_preferences/save_preference tools
# exist in tools/.

def load_preferences_node(state: GraphState) -> dict:
    logger.info(
        "load_preferences_node: intents=%s preferences=%s",
        state.get("intents"),
        state.get("preferences"),
    )
    # Replace with:
    #   preferences = get_preferences(state["session_id"], state.get("customer_id"))
    return {"preferences": state.get("preferences", {})}


def save_preferences_node(state: GraphState) -> dict:
    signal = state.get("pending_preference_signal")
    if signal:
        # Replace with:
        #   save_preference(state.get("customer_id") or state["session_id"], signal)
        logger.info("save_preferences_node: would save preference: %s", signal)
    else:
        logger.info("save_preferences_node: no pending preference signal")
    return {"pending_preference_signal": None}


# --- Identity gate: HITL verify --------------------------------------------

def hitl_verify_node(state: GraphState) -> dict:
    """Pauses the graph and asks the caller for verification info. On
    resume, `verification_input` is whatever was passed to
    Command(resume=...) — e.g. {"customer_id": "123", "last_name": "Diaz"}."""

    logger.info("hitl_verify_node: interrupting to request identity verification")

    verification_input = interrupt(
        {
            "reason": "identity_verification_required",
            "message": (
                "To help with your order or invoice, I need to verify "
                "your identity first — can you share your customer ID "
                "and last name?"
            ),
        }
    )

    # Replace with your real check, e.g. a DB lookup tool:
    #   is_verified = verify_customer(verification_input.get("customer_id"),
    #                                  verification_input.get("last_name"))
    is_verified = bool(verification_input.get("customer_id"))  # stub

    logger.info("hitl_verify_node: resumed with customer_verified=%s", is_verified)

    return {
        "customer_verified": is_verified,
        "customer_id": verification_input.get("customer_id") if is_verified else None,
    }


def route_after_hitl(state: GraphState) -> str:
    # Verified -> proceed to invoice. Still not verified -> ask again.
    decision = "invoice_agent" if state["customer_verified"] else "hitl_verify"
    logger.info(
        "route_after_hitl: customer_verified=%s -> %s",
        state["customer_verified"],
        decision,
    )
    return decision


# --- Intent queue: dispatch + advance ---------------------------------------

def dispatch_next_intent(state: GraphState) -> str:
    """Looks at the front of the intent queue and decides where to go.
    Used after load_preferences AND after advance_intent, so it's the
    single place that knows how to route any given intent."""

    intents = state["intents"]
    if not intents:
        logger.info("dispatch_next_intent: queue empty -> save_preferences")
        return "save_preferences"

    next_intent = intents[0]
    if next_intent == "catalog":
        logger.info("dispatch_next_intent: intents=%s -> catalog_agent", intents)
        return "catalog_agent"

    # invoice
    decision = "invoice_agent" if state["customer_verified"] else "hitl_verify"
    logger.info(
        "dispatch_next_intent: intents=%s customer_verified=%s -> %s",
        intents,
        state["customer_verified"],
        decision,
    )
    return decision


def advance_intent_node(state: GraphState) -> dict:
    """Pops the just-completed intent off the front of the queue."""
    remaining = state["intents"][1:]
    logger.info(
        "advance_intent_node: completed=%s remaining=%s",
        state["intents"][0] if state["intents"] else None,
        remaining,
    )
    return {"intents": remaining}


# --- Build ------------------------------------------------------------------

def build_graph():
    logger.info("build_graph: assembling graph")
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("load_preferences", load_preferences_node)
    graph.add_node("catalog_agent", catalog_agent_node)
    graph.add_node("catalog_tools", catalog_tools_node)
    graph.add_node("hitl_verify", hitl_verify_node)
    graph.add_node("invoice_agent", invoice_agent_node)
    graph.add_node("invoice_tools", invoice_tools_node)
    graph.add_node("advance_intent", advance_intent_node)
    graph.add_node("save_preferences", save_preferences_node)
    logger.debug(
        "build_graph: registered nodes: %s",
        [
            "router",
            "load_preferences",
            "catalog_agent",
            "catalog_tools",
            "hitl_verify",
            "invoice_agent",
            "invoice_tools",
            "advance_intent",
            "save_preferences",
        ],
    )

    graph.set_entry_point("router")
    logger.debug("build_graph: entry point -> router")

    graph.add_edge("router", "load_preferences")
    logger.debug("build_graph: edge router -> load_preferences")

    dispatch_map = {
        "catalog_agent": "catalog_agent",
        "invoice_agent": "invoice_agent",
        "hitl_verify": "hitl_verify",
        "save_preferences": "save_preferences",
    }

    graph.add_conditional_edges("load_preferences", dispatch_next_intent, dispatch_map)
    logger.debug(
        "build_graph: conditional edges load_preferences -[dispatch_next_intent]-> %s",
        dispatch_map,
    )

    # Catalog: tool loop, then advance to the next intent (if any)
    catalog_map = {
        "catalog_tools": "catalog_tools",
        "done": "advance_intent",
    }

    graph.add_conditional_edges("catalog_agent", route_after_catalog_agent, catalog_map)
    logger.debug(
        "build_graph: conditional edges catalog_agent -[route_after_catalog_agent]-> %s",
        catalog_map,
    )

    graph.add_edge("catalog_tools", "catalog_agent")
    logger.debug("build_graph: edge catalog_tools -> catalog_agent")

    # Invoice: identity gate, then advance to the next intent (if any)
    hitl_map = {
        "invoice_agent": "invoice_agent",
        "hitl_verify": "hitl_verify",
    }
    graph.add_conditional_edges("hitl_verify", route_after_hitl, hitl_map)
    logger.debug(
        "build_graph: conditional edges hitl_verify -[route_after_hitl]-> %s",
        hitl_map,
    )

    invoice_map = {
        "invoice_tools": "invoice_tools",
        "done": "advance_intent",
    }
    # NOTE: this was `add_conditional_edge` (singular) in the version you
    # pasted — that method doesn't exist on StateGraph and would raise
    # AttributeError at build time. Fixed to `add_conditional_edges`.
    graph.add_conditional_edges("invoice_agent", route_after_invoice_agent, invoice_map)
    logger.debug(
        "build_graph: conditional edges invoice_agent -[route_after_invoice_agent]-> %s",
        invoice_map,
    )

    graph.add_edge("invoice_tools", "invoice_agent")
    logger.debug("build_graph: edge invoice_tools -> invoice_agent")

    # advance_intent re-runs dispatch on whatever's left in the queue
    graph.add_conditional_edges("advance_intent", dispatch_next_intent, dispatch_map)
    logger.debug(
        "build_graph: conditional edges advance_intent -[dispatch_next_intent]-> %s",
        dispatch_map,
    )

    graph.add_edge("save_preferences", END)
    logger.debug("build_graph: edge save_preferences -> END")

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("build_graph: graph compiled")
    return compiled


compiled_graph = build_graph()


if __name__ == "__main__":
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "test-session-1"}}

    # Turn 1: mixed intent, not yet verified
    result = compiled_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "do you have any Beatles albums? also what's the status of my last order?",
                }
            ],
            "session_id": "test-session-1",
            "customer_id": None,
            "customer_verified": False,
            "intents": [],
            "pending_preference_signal": None,
            "preferences": {},
        },
        config=config,
    )

    if "__interrupt__" in result:
        # Catalog answer is already in result["messages"]; the interrupt
        # payload carries the verification prompt to show alongside it.
        print("Catalog answer so far:", result["messages"][-1].content)
        print("Verification needed:", result["__interrupt__"][0].value["message"])

        # Turn 2: customer replies with their ID -> resume the graph
        # result = compiled_graph.invoke(
        #     Command(resume={"customer_id": "123", "last_name": "Diaz"}),
        #     config=config,
        # )
        # Turn 3: customer replies with their ID -> resume the graph
        result = compiled_graph.invoke(
            Command(resume={"customer_id": "43", "last_name": "Mercier"}),
            config=config,
        )
        print("Final:", result["messages"][-1].content)
    else:
        print(result["messages"][-1].content)