from customer_support.agents.router import router_llm, ROUTER_SYSTEM_PROMPT


def _classify(history):
    return router_llm.invoke([{"role": "system", "content": ROUTER_SYSTEM_PROMPT}, *history])


class TestRouterContextAwareness:
    def test_bare_preference_statement_needs_context_to_classify(self):
        # Regression test: "i love jazz" alone carries no catalog
        # keyword and was previously misclassified as off-topic, even
        # though it's an unambiguous reply to the assistant's own
        # question.
        history = [
            {"role": "user", "content": "can you suggest me some albums?"},
            {
                "role": "assistant",
                "content": (
                    "Since there are no saved preferences for you yet, could "
                    "you let me know what type of music you're interested in?"
                ),
            },
            {"role": "user", "content": "i love jazz"},
        ]

        result = _classify(history)

        assert "catalog" in result.intents

    def test_reference_to_earlier_turn_resolves_with_context(self):
        history = [
            {"role": "user", "content": "what did I purchase in my previous orders?"},
            {
                "role": "assistant",
                "content": "You bought tracks from Led Zeppelin, Titas, and Yo-Yo Ma.",
            },
            {"role": "user", "content": "what about the second one?"},
        ]

        result = _classify(history)

        assert "invoice" in result.intents

    def test_off_topic_message_still_rejected_mid_conversation(self):
        # The critical regression check: more context must not make
        # the router LESS strict about genuinely unrelated messages --
        # prior turns being about music doesn't make a weather
        # question in-scope.
        history = [
            {"role": "user", "content": "do you have any AC/DC albums?"},
            {
                "role": "assistant",
                "content": "Yes, we have For Those About To Rock and Let There Be Rock.",
            },
            {"role": "user", "content": "what's the weather like today?"},
        ]

        result = _classify(history)

        assert result.intents == []

    def test_self_contained_message_classifies_without_history(self):
        # A message with no prior conversation must still classify
        # correctly on its own -- the fix adds context, it doesn't
        # require it.
        result = _classify([{"role": "user", "content": "do you have any AC/DC albums?"}])

        assert result.intents == ["catalog"]

    def test_mixed_intent_still_detected(self):
        # Ordering (invoice-first) is applied afterward by router_node
        # via sorted(), not by the raw classification -- covered
        # end-to-end in test_graph_flow.py. This just checks both
        # intents are still found in one message.
        result = _classify(
            [
                {
                    "role": "user",
                    "content": (
                        "do you have any Beatles albums? also what's the "
                        "status of my last order?"
                    ),
                }
            ]
        )

        assert set(result.intents) == {"invoice", "catalog"}