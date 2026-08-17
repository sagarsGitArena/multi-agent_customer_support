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
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt

from customer_support.identity import extract_identity_from_message, resolve_customer_id
from customer_support.graph.state import GraphState, format_state
from customer_support.agents.router import router_node
from customer_support.agents.catalog_agent import catalog_subgraph
from customer_support.agents.invoice_agent import invoice_subgraph
from customer_support.agents.memory import load_memory_node, create_memory_node

logger = logging.getLogger(__name__)


# --- Identity gate: HITL verify --------------------------------------------

def hitl_verify_node(state: GraphState) -> dict:
    """Pauses the graph and asks the caller for verification info --
    unless the message that triggered this turn already contains a
    usable identifier (e.g. "my customer id is 43, where's my
    order?"), in which case it verifies immediately without asking at
    all. In practice this is now mostly a fallback/safety net:
    load_memory_node (agents/memory.py) already attempts the same
    capture on every turn, before intent dispatch even happens, so by
    the time this node is reached customer_id is often already set.

    On resume, `verification_input` is whatever was passed to
    Command(resume=...) — e.g. {"customer_id": "123", "last_name": "Diaz"},
    {"email": "isabelle_mercier@apple.fr"}, or {"phone": "+33 3 80 73 66 99"}.

    The pre-check and the interrupt() call happen in the same pass
    deliberately: if the pre-check finds nothing (or finds something
    that doesn't verify), it falls straight through to interrupt()
    right here rather than returning unverified and relying on
    route_after_hitl to loop back -- looping back would just re-run
    this same pre-check against the same unchanged message forever
    and never actually pause to ask."""

    logger.info("hitl_verify_node: state=\n%s", format_state(state))

    triggering_message = state["messages"][-1]
    triggering_content = getattr(triggering_message, "content", "") or ""

    verification_input = extract_identity_from_message(triggering_content)
    customer_id = resolve_customer_id(verification_input)

    if customer_id:
        logger.info(
            "hitl_verify_node: found identifier already in the triggering message, skipping interrupt"
        )
    else:
        logger.info("hitl_verify_node: interrupting to request identity verification")
        verification_input = interrupt(
            {
                "reason": "identity_verification_required",
                "message": (
                    "To help with your order or invoice, I need to verify "
                    "your identity first — can you share your customer ID, "
                    "the email, or the phone number on your account?"
                ),
            }
        )
        customer_id = resolve_customer_id(verification_input)

    is_verified = bool(customer_id)

    logger.info("hitl_verify_node: resumed with customer_verified=%s", is_verified)

    return {
        "customer_verified": is_verified,
        "customer_id": customer_id if is_verified else None,
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
    Used after load_memory AND after advance_intent, so it's the
    single place that knows how to route any given intent."""

    intents = state["intents"]
    if not intents:
        logger.info("dispatch_next_intent: queue empty -> create_memory")
        return "create_memory"

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
    logger.info("advance_intent_node: state=\n%s", format_state(state))

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
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("catalog_agent", catalog_subgraph)
    graph.add_node("hitl_verify", hitl_verify_node)
    graph.add_node("invoice_agent", invoice_subgraph)
    graph.add_node("advance_intent", advance_intent_node)
    graph.add_node("create_memory", create_memory_node)
    logger.debug(
        "build_graph: registered nodes: %s",
        [
            "router",
            "load_memory",
            "catalog_agent",
            "hitl_verify",
            "invoice_agent",
            "advance_intent",
            "create_memory",
        ],
    )

    graph.set_entry_point("router")
    logger.debug("build_graph: entry point -> router")

    graph.add_edge("router", "load_memory")
    logger.debug("build_graph: edge router -> load_memory")

    dispatch_map = {
        "catalog_agent": "catalog_agent",
        "invoice_agent": "invoice_agent",
        "hitl_verify": "hitl_verify",
        "create_memory": "create_memory",
    }

    graph.add_conditional_edges("load_memory", dispatch_next_intent, dispatch_map)
    logger.debug(
        "build_graph: conditional edges load_memory -[dispatch_next_intent]-> %s",
        dispatch_map,
    )

    # Catalog: the tool loop is internal to catalog_subgraph; from the
    # parent's perspective it's a single node, so just advance to the
    # next intent (if any) once it returns.
    graph.add_edge("catalog_agent", "advance_intent")
    logger.debug("build_graph: edge catalog_agent -> advance_intent")

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

    # Invoice: the tool loop and ownership enforcement are internal to
    # invoice_subgraph; from the parent's perspective it's a single node,
    # so just advance to the next intent (if any) once it returns.
    graph.add_edge("invoice_agent", "advance_intent")
    logger.debug("build_graph: edge invoice_agent -> advance_intent")

    # advance_intent re-runs dispatch on whatever's left in the queue
    graph.add_conditional_edges("advance_intent", dispatch_next_intent, dispatch_map)
    logger.debug(
        "build_graph: conditional edges advance_intent -[dispatch_next_intent]-> %s",
        dispatch_map,
    )

    graph.add_edge("create_memory", END)
    logger.debug("build_graph: edge create_memory -> END")

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer, store=memory_store)
    logger.info("build_graph: graph compiled")
    return compiled


memory_store = InMemoryStore()
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
        },
        config=config,
    )

    if "__interrupt__" in result:
        # Mixed queries handle invoice first, so on an unverified customer
        # the graph interrupts here before catalog has run at all -- there's
        # no answer yet, just the verification prompt.
        print("Verification needed:", result["__interrupt__"][0].value["message"])

        # #Turn 2: customer replies with their ID -> resume the graph
        # result = compiled_graph.invoke(
        #     Command(resume={"customer_id": "123", "last_name": "Diaz"}),
        #     config=config,
        # )
        #Turn 3: customer replies with their ID -> resume the graph
        result = compiled_graph.invoke(
            Command(resume={"customer_id": "43", "last_name": "Mercier"}),
            config=config,
        )
        print("Final:", result["messages"][-1].content)
    else:
        print(result["messages"][-1].content)