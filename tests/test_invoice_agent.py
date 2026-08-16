from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from customer_support.agents.invoice_agent import invoice_subgraph


class TestInvoiceSubgraph:
    def test_answers_order_status_with_tool_call_loop(self):
        result = invoice_subgraph.invoke(
            {
                "messages": [HumanMessage(content="what's the status of my last order?")],
                "customer_id": "43",
            }
        )

        messages = result["messages"]
        last_message = messages[-1]

        assert isinstance(last_message, AIMessage)
        assert last_message.content
        assert not getattr(last_message, "tool_calls", None)

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert tool_messages

    def test_cannot_access_another_customers_invoice(self):
        # Invoice 1 belongs to a different customer (per Chinook fixture
        # data); ownership enforcement must reject it even if the model
        # is asked directly for it.
        result = invoice_subgraph.invoke(
            {
                "messages": [
                    HumanMessage(content="show me the line items for invoice 1")
                ],
                "customer_id": "43",
            }
        )

        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        assert any(
            "does not belong to the verified customer" in m.content
            for m in tool_messages
        )

    def test_catalog_question_is_declined(self):
        result = invoice_subgraph.invoke(
            {
                "messages": [HumanMessage(content="do you have any AC/DC albums?")],
                "customer_id": "43",
            }
        )

        last_message = result["messages"][-1]

        assert isinstance(last_message, AIMessage)
        assert not getattr(last_message, "tool_calls", None)