"""
Catalog agent node for the music store agentic graph.
 
Location: src/customer_support/agents/catalog_agent.py
 
This is a small agent loop, not a single LLM call: the model can call
catalog tools, see the results, and call more tools before answering
(e.g. look up an artist, then check whether a specific track is in
stock). The loop exits once the model responds without any tool calls,
and the graph then moves on to save_preferences.
"""
 
import logging
from typing import Literal
from langgraph.prebuilt import ToolNode

from customer_support.config import get_llm
from customer_support.graph.state import GraphState, format_state
from customer_support.tools.music_catalog_tools import (
    search_albums_by_artist,
    search_tracks_by_artist,
    browse_songs_by_genre,
    search_songs_by_title,
    get_track_details,
)  # adjust names to match your actual exports from this module

logger = logging.getLogger(__name__)

CATALOG_TOOLS = [search_albums_by_artist, search_tracks_by_artist, browse_songs_by_genre, search_songs_by_title, get_track_details]



# --- Catalog agent node ---------------------------------------------------

CATALOG_SYSTEM_PROMPT = """You are the catalog assistant for a music \
store. You can search the catalog, look up tracks and artists, and \
recommend music. Use the customer's known preferences to personalize \
recommendations when relevant, but don't force them in if they're not \
relevant to the question. Call tools as needed before answering — don't \
guess at catalog details you haven't looked up. Stay in the catalog scope and do not attempt to answer outside of catalog scope."""
# If asked about orders, \
# invoices, or billing, note that's handled separately and don't attempt \
# to answer it here."""

catalog_llm = get_llm().bind_tools(CATALOG_TOOLS)

def catalog_agent_node(state: GraphState) -> dict:
    """Runs the catalog LLM with preferences in context. May return a
    message containing tool calls, which routes to catalog_tools next,
    or a plain answer, which routes on to save_preferences."""
    
    logger.info("catalog_agent_node: state=\n%s", format_state(state))

    preferences = state.get("preferences", {})
    system_message = {
        "role": "system",
        "content": (
            f"{CATALOG_SYSTEM_PROMPT}\n\n"
            f"Known customer preferences: {preferences or 'none recorded yet'}"
        ),
    }

    logger.info("catalog_agent_node: invoking catalog LLM, preferences=%s", preferences)

    response = catalog_llm.invoke([system_message, *state["messages"]])

    tool_calls = getattr(response, "tool_calls", None) or []
    logger.info(
        "catalog_agent_node: response has %d tool call(s): %s",
        len(tool_calls),
        [call.get("name") for call in tool_calls],
    )

    return {"messages": [response]}

def route_after_catalog_agent(state: GraphState) -> Literal["catalog_tools", "save_preferences"]:
    """Loop back to tools if the model made tool calls, otherwise the
    turn is done and we head to save_preferences.

    NOTE: once graph/routing.py exists, move this function there
    alongside the other conditional-edge functions (identity check,
    invoice loop, etc.) so all branching logic lives in one place.
    """

    last_message = state["messages"][-1]
    decision = "catalog_tools" if getattr(last_message, "tool_calls", None) else "done"
    logger.info("route_after_catalog_agent: -> %s", decision)
    return decision


_catalog_tool_runner = ToolNode(CATALOG_TOOLS)


def catalog_tools_node(state: GraphState) -> dict:
    """Executes the tool call(s) requested by catalog_agent_node."""
    logger.info("catalog_tools_node: state=\n%s", format_state(state))

    tool_calls = getattr(state["messages"][-1], "tool_calls", None) or []
    logger.info(
        "catalog_tools_node: executing tool call(s): %s",
        [call.get("name") for call in tool_calls],
    )
    return _catalog_tool_runner.invoke(state)
 
 
# --- Graph wiring (goes in graph/build.py) ---------------------------------
#
#   from customer_support.agents.catalog_agent import (
#       catalog_agent_node, catalog_tools_node, route_after_catalog_agent,
#   )
#
#   graph.add_node("catalog_agent", catalog_agent_node)
#   graph.add_node("catalog_tools", catalog_tools_node)
#
#   graph.add_conditional_edges(
#       "catalog_agent",
#       route_after_catalog_agent,
#       {"catalog_tools": "catalog_tools", "save_preferences": "save_preferences"},
#   )
#   graph.add_edge("catalog_tools", "catalog_agent")  # loop back after tool results
 