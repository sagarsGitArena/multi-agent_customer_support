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

Invoice agent itself is still a stub (not built yet) — this wires the
identity gate and interrupt mechanics around where it will plug in.
"""

from langchain_core.messages import AIMessage
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


# --- Preferences: temporary inline stand-ins --------------------------------
# TODO: move to agents/memory.py once get_preferences/save_preference tools
# exist in tools/.

def load_preferences_node(state: GraphState) -> dict:
    # Replace with:
    #   preferences = get_preferences(state["session_id"], state.get("customer_id"))
    return {"preferences": state.get("preferences", {})}


def save_preferences_node(state: GraphState) -> dict:
    signal = state.get("pending_preference_signal")
    if signal:
        # Replace with:
        #   save_preference(state.get("customer_id") or state["session_id"], signal)
        print(f"[stub] would save preference: {signal}")
    return {"pending_preference_signal": None}


# --- Identity gate: HITL verify --------------------------------------------

def hitl_verify_node(state: GraphState) -> dict:
    """Pauses the graph and asks the caller for verification info. On
    resume, `verification_input` is whatever was passed to
    Command(resume=...) — e.g. {"customer_id": "123", "last_name": "Diaz"}."""

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

    return {
        "customer_verified": is_verified,
        "customer_id": verification_input.get("customer_id") if is_verified else None,
    }


def route_after_hitl(state: GraphState) -> str:
    # Verified -> proceed to invoice. Still not verified -> ask again.
    return "invoice_agent" if state["customer_verified"] else "hitl_verify"


# --- Invoice agent: not built yet -------------------------------------------

def invoice_agent_node(state: GraphState) -> dict:
    # TODO: real invoice agent + tool loop, mirroring catalog_agent_node.
    return {
        "messages": [
            AIMessage(content="[stub] Here's what I'd tell you about your invoice.")
        ]
    }


# --- Intent queue: dispatch + advance ---------------------------------------

def dispatch_next_intent(state: GraphState) -> str:
    """Looks at the front of the intent queue and decides where to go.
    Used after load_preferences AND after advance_intent, so it's the
    single place that knows how to route any given intent."""

    intents = state["intents"]
    if not intents:
        return "save_preferences"

    next_intent = intents[0]
    if next_intent == "catalog":
        return "catalog_agent"

    # invoice
    return "invoice_agent" if state["customer_verified"] else "hitl_verify"


def advance_intent_node(state: GraphState) -> dict:
    """Pops the just-completed intent off the front of the queue."""
    return {"intents": state["intents"][1:]}


# --- Build ------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("load_preferences", load_preferences_node)
    graph.add_node("catalog_agent", catalog_agent_node)
    graph.add_node("catalog_tools", catalog_tools_node)
    graph.add_node("hitl_verify", hitl_verify_node)
    graph.add_node("invoice_agent", invoice_agent_node)
    graph.add_node("advance_intent", advance_intent_node)
    graph.add_node("save_preferences", save_preferences_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "load_preferences")

    graph.add_conditional_edges(
        "load_preferences",
        dispatch_next_intent,
        {
            "catalog_agent": "catalog_agent",
            "invoice_agent": "invoice_agent",
            "hitl_verify": "hitl_verify",
            "save_preferences": "save_preferences",
        },
    )

    # Catalog: tool loop, then advance to the next intent (if any)
    graph.add_conditional_edges(
        "catalog_agent",
        route_after_catalog_agent,
        {"catalog_tools": "catalog_tools", "save_preferences": "advance_intent"},
    )
    graph.add_edge("catalog_tools", "catalog_agent")

    # Invoice: identity gate, then advance to the next intent (if any)
    graph.add_conditional_edges(
        "hitl_verify",
        route_after_hitl,
        {"invoice_agent": "invoice_agent", "hitl_verify": "hitl_verify"},
    )
    graph.add_edge("invoice_agent", "advance_intent")

    # advance_intent re-runs dispatch on whatever's left in the queue
    graph.add_conditional_edges(
        "advance_intent",
        dispatch_next_intent,
        {
            "catalog_agent": "catalog_agent",
            "invoice_agent": "invoice_agent",
            "hitl_verify": "hitl_verify",
            "save_preferences": "save_preferences",
        },
    )

    graph.add_edge("save_preferences", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


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
        result = compiled_graph.invoke(
            Command(resume={"customer_id": "123", "last_name": "Diaz"}),
            config=config,
        )
        print("Final:", result["messages"][-1].content)
    else:
        print(result["messages"][-1].content)