"""
Eval dataset + target dispatcher for the groundedness/consistency evaluator.

Location: src/customer_support/evaluation/dataset.py

Each example only describes a scenario to run (which graph, what
initial state) -- there's no hand-written "expected answer" to
maintain in parallel. `target()` invokes the real graph/subgraph and
returns both the final answer and the actual conversation transcript
that preceded it, so the judge always checks the response against
that run's own real context (tool results, prior turns), not a
separately hand-typed summary that could drift out of sync with the
actual prompts/tools.

Two of the four examples are the real regression scenarios from
README "Bugs found and fixed" (now expected to pass, since both are
fixed) -- run through the actual subgraph/graph, live. The other two
are deliberately unambiguous (a clean "not found" catalog answer, and
a hand-crafted bad response) so the eval set isn't just "does the
current code pass," it also proves the judge itself discriminates.
"""

import uuid

from langchain_core.messages import AIMessage, HumanMessage

from customer_support.agents.catalog_agent import catalog_subgraph
from customer_support.evaluation.groundedness import build_transcript
from customer_support.graph.build import compiled_graph


def _run_catalog_subgraph(state: dict) -> dict:
    result = catalog_subgraph.invoke(state)
    messages = result["messages"]

    # catalog_agent_node injects "Known customer preferences: ..." as a
    # separate system message at LLM-call time (agents/catalog_agent.py)
    # -- it's never added to state["messages"], so without prepending it
    # here the judge would never see the actual fact a claim like "no
    # saved preferences yet" is grounded in, and would false-negative it.
    preferences_context = state.get("preferences_context")
    system_fact = (
        f"System: Known customer preferences: "
        f"{preferences_context or 'No saved preferences for this customer yet.'}"
    )

    context = "\n".join([system_fact, build_transcript(messages[:-1])])
    return {"answer": messages[-1].content, "context": context}


def _run_compiled_graph(state: dict) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = compiled_graph.invoke(state, config=config)
    messages = result["messages"]
    answers = [m for m in messages if isinstance(m, AIMessage) and m.content]
    answer = "\n\n".join(m.content for m in answers)
    prior = messages[: messages.index(answers[-1])] if answers else messages
    return {"answer": answer, "context": build_transcript(prior)}


def _return_fixed_output(state: dict) -> dict:
    """For the hand-crafted example: no graph to run, just echo the
    pre-built context/answer straight through."""
    return state


_TARGETS = {
    "catalog_subgraph": _run_catalog_subgraph,
    "compiled_graph": _run_compiled_graph,
    "fixed": _return_fixed_output,
}


def target(inputs: dict) -> dict:
    """LangSmith target function: dispatches on inputs["graph"]."""
    return _TARGETS[inputs["graph"]](inputs["state"])


EXAMPLES = [
    {
        # Regression case for the "diverse taste" bug: purchase history
        # sits earlier in the same conversation, no stored preferences.
        # catalog_agent must not read that history and assert a taste
        # characterization as fact.
        "inputs": {
            "graph": "catalog_subgraph",
            "state": {
                "messages": [
                    HumanMessage(content="what did I purchase in my previous orders?"),
                    AIMessage(
                        content=(
                            'Here are your previous purchases: "Whole Lotta Love" by Led '
                            "Zeppelin (rock), \"Die Walküre: The Ride of the Valkyries\" "
                            "by Sir Georg Solti & Wiener Philharmoniker (classical), and "
                            "\"Prá Dizer Adeus\" by Titãs (Brazilian rock)."
                        )
                    ),
                    HumanMessage(content="any suggestions for me?"),
                ],
                "preferences_context": None,
            },
        },
    },
    {
        # Regression case for the invoice/catalog contradiction bug: a
        # mixed-intent question where the invoice half must not
        # comment on the catalog half at all.
        "inputs": {
            "graph": "compiled_graph",
            "state": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "do you have Michael Jackson albums and let me know "
                            "the status of my last order"
                        ),
                    }
                ],
                "session_id": "eval",
                "customer_id": "43",
                "customer_verified": True,
                "intents": [],
            },
        },
    },
    {
        # Straightforward, unambiguous good case: a real "not found"
        # catalog answer, nothing to infer -- confirms the judge
        # doesn't flag ordinary honest answers as ungrounded.
        "inputs": {
            "graph": "catalog_subgraph",
            "state": {
                "messages": [
                    HumanMessage(content="do you have any albums by ThisArtistDoesNotExistZZZ?")
                ],
                "preferences_context": None,
            },
        },
    },
    {
        # Deliberately bad, hand-crafted response: proves the judge
        # actually discriminates rather than rubber-stamping everything.
        "inputs": {
            "graph": "fixed",
            "state": {
                "context": (
                    "HumanMessage: what did I purchase in my previous orders?\n"
                    "AIMessage: Here are your previous purchases: \"Whole Lotta Love\" "
                    "by Led Zeppelin.\n"
                    "HumanMessage: any suggestions for me?"
                ),
                "answer": (
                    "Since you have diverse taste in music, including rock, classical, "
                    "and world music, here are some recommendations..."
                ),
            },
        },
    },
]
