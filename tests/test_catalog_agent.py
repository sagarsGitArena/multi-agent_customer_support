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

    def test_does_not_infer_taste_from_purchase_history_in_context(self):
        # Regression test: catalog_agent_node sees the FULL shared
        # message history, including whatever invoice_agent said
        # earlier in the same conversation. It must not read a
        # customer's purchase history off that and assert a taste
        # characterization ("you have diverse taste...") as fact --
        # personalization is only allowed from preferences_context.
        purchase_summary = AIMessage(
            content=(
                "Here are your previous purchases: \"Whole Lotta Love\" by Led "
                "Zeppelin (rock), \"Die Walküre: The Ride of the Valkyries\" "
                "by Sir Georg Solti & Wiener Philharmoniker (classical), and "
                "\"Prá Dizer Adeus\" by Titãs (Brazilian rock)."
            )
        )

        result = catalog_subgraph.invoke(
            {
                "messages": [
                    HumanMessage(content="what did I purchase in my previous orders?"),
                    purchase_summary,
                    HumanMessage(content="any suggestions for me?"),
                ],
                "preferences_context": None,
            }
        )

        last_message = result["messages"][-1]
        content_lower = last_message.content.lower()

        assert isinstance(last_message, AIMessage)
        assert "diverse taste" not in content_lower
        assert "your taste" not in content_lower