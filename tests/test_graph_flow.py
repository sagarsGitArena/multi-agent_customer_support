import uuid

from langgraph.types import Command

from customer_support.graph.build import compiled_graph


class TestCatalogOnlyFlow:
    def test_catalog_only_question_answers_without_verification(self):
        config = {"configurable": {"thread_id": f"test-catalog-only-{uuid.uuid4()}"}}

        result = compiled_graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "do you have any AC/DC albums?"}
                ],
                "session_id": "test-catalog-only",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=config,
        )

        # A catalog-only question never touches the invoice/identity gate,
        # so the graph should run straight through to a final answer --
        # no interrupt, no leftover intents in the queue.
        assert "__interrupt__" not in result
        assert result["intents"] == []
        assert result["customer_verified"] is False

        last_message = result["messages"][-1]
        assert last_message.content
        assert not getattr(last_message, "tool_calls", None)


class TestInvoiceOnlyFlow:
    def test_invoice_question_interrupts_then_resumes_to_answer(self):
        config = {"configurable": {"thread_id": f"test-invoice-only-{uuid.uuid4()}"}}

        result = compiled_graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "what's the status of my last order?"}
                ],
                "session_id": "test-invoice-only",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=config,
        )

        # Unverified customer: the identity gate must interrupt before
        # invoice_subgraph ever runs.
        assert "__interrupt__" in result
        assert (
            result["__interrupt__"][0].value["reason"]
            == "identity_verification_required"
        )

        result = compiled_graph.invoke(
            Command(resume={"customer_id": "43", "last_name": "Mercier"}),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["intents"] == []
        assert result["customer_verified"] is True
        assert result["customer_id"] == "43"

        last_message = result["messages"][-1]
        assert last_message.content
        assert not getattr(last_message, "tool_calls", None)