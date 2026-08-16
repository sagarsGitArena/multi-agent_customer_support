from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from customer_support.agents.catalog_agent import catalog_subgraph


class TestCatalogSubgraph:
    def test_answers_with_tool_call_loop(self):
        result = catalog_subgraph.invoke(
            {
                "messages": [HumanMessage(content="do you have any AC/DC albums?")],
                "preferences_context": None,
            }
        )

        messages = result["messages"]
        last_message = messages[-1]

        assert isinstance(last_message, AIMessage)
        assert last_message.content
        assert not getattr(last_message, "tool_calls", None)

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert tool_messages
        assert any("Let There Be Rock" in m.content for m in tool_messages)

    def test_answers_when_nothing_found(self):
        result = catalog_subgraph.invoke(
            {
                "messages": [
                    HumanMessage(content="do you have any albums by ThisArtistDoesNotExistZZZ?")
                ],
                "preferences_context": None,
            }
        )

        messages = result["messages"]
        last_message = messages[-1]

        assert isinstance(last_message, AIMessage)
        assert last_message.content
        assert not getattr(last_message, "tool_calls", None)

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert tool_messages

    def test_off_topic_scope_is_declined(self):
        result = catalog_subgraph.invoke(
            {
                "messages": [HumanMessage(content="what's the weather like today?")],
                "preferences_context": None,
            }
        )

        last_message = result["messages"][-1]

        assert isinstance(last_message, AIMessage)
        assert not getattr(last_message, "tool_calls", None)