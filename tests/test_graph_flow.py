import uuid

from langchain_core.messages import AIMessage
from langgraph.types import Command

from customer_support.graph.build import compiled_graph, memory_store


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
    def _start_unverified_invoice_turn(self):
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
        return config

    def test_invoice_question_interrupts_then_resumes_to_answer(self):
        config = self._start_unverified_invoice_turn()

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

    def test_resumes_via_email_lookup(self):
        config = self._start_unverified_invoice_turn()

        result = compiled_graph.invoke(
            Command(resume={"email": "ISABELLE_MERCIER@apple.fr"}),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["customer_verified"] is True
        assert result["customer_id"] == "43"

        last_message = result["messages"][-1]
        assert last_message.content
        assert not getattr(last_message, "tool_calls", None)

    def test_unknown_email_stays_unverified_and_asks_again(self):
        config = self._start_unverified_invoice_turn()

        result = compiled_graph.invoke(
            Command(resume={"email": "nobody@nowhere.example"}),
            config=config,
        )

        # No matching account -> still unverified, interrupted again
        # asking for identity, not routed on to invoice_agent.
        assert "__interrupt__" in result
        assert result["customer_verified"] is False
        assert result["customer_id"] is None

    def test_resumes_via_phone_lookup(self):
        config = self._start_unverified_invoice_turn()

        # Deliberately differently formatted from the stored
        # "+33 03 80 73 66 99" -- proves the lookup normalizes both
        # sides rather than requiring an exact string match.
        result = compiled_graph.invoke(
            Command(resume={"phone": "+33-03-80-73-66-99"}),
            config=config,
        )

        assert "__interrupt__" not in result
        assert result["customer_verified"] is True
        assert result["customer_id"] == "43"

        last_message = result["messages"][-1]
        assert last_message.content
        assert not getattr(last_message, "tool_calls", None)

    def test_unknown_phone_stays_unverified_and_asks_again(self):
        config = self._start_unverified_invoice_turn()

        result = compiled_graph.invoke(
            Command(resume={"phone": "+1 555 000 0000"}),
            config=config,
        )

        assert "__interrupt__" in result
        assert result["customer_verified"] is False
        assert result["customer_id"] is None


class TestIdentityInFirstMessage:
    """Regression tests: a customer who volunteers their identity in
    the very message that asks the question ("my customer id is 43,
    where's my order?") must be verified immediately, not re-asked for
    info they already gave."""

    def _config(self):
        return {"configurable": {"thread_id": f"test-identity-in-msg-{uuid.uuid4()}"}}

    def test_customer_id_in_triggering_message_skips_interrupt(self):
        result = compiled_graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "my customer id is 43. where is my order?",
                    }
                ],
                "session_id": "test-identity-in-msg",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=self._config(),
        )

        assert "__interrupt__" not in result
        assert result["customer_verified"] is True
        assert result["customer_id"] == "43"

        last_message = result["messages"][-1]
        assert last_message.content
        assert not getattr(last_message, "tool_calls", None)

    def test_unrelated_number_in_message_does_not_falsely_verify(self):
        # "order 43" is not a customer ID -- the pre-check must require
        # an explicit "customer id" label, not just any digit run,
        # or an invoice/order number would be misread as an identity.
        result = compiled_graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "where's the status of order 43?"}
                ],
                "session_id": "test-identity-in-msg",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=self._config(),
        )

        assert "__interrupt__" in result
        assert result["customer_verified"] is False

    def test_unresolvable_identifier_in_message_still_asks_cleanly(self):
        # An email-shaped string that doesn't match any real account
        # must fall through to a normal interrupt in the same pass,
        # not loop silently.
        config = self._config()

        result = compiled_graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "my email is nobody@nowhere.example, where's my order?",
                    }
                ],
                "session_id": "test-identity-in-msg",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=config,
        )

        assert "__interrupt__" in result
        assert result["customer_verified"] is False

        # And a real reply afterward still resolves normally.
        result = compiled_graph.invoke(
            Command(resume={"customer_id": "43"}), config=config
        )
        assert "__interrupt__" not in result
        assert result["customer_verified"] is True
        assert result["customer_id"] == "43"


class TestBarePreferenceStatementFollowUp:
    def test_preference_reply_is_not_rejected_as_off_topic(self):
        # Regression test for the reported bug: a bare "i love jazz"
        # replying to "what are you interested in?" was previously
        # misclassified as off-topic by the router (no history, no
        # catalog keyword in the message itself) and got the canned
        # rejection instead of a real recommendation.
        config = {"configurable": {"thread_id": f"test-bare-pref-{uuid.uuid4()}"}}

        compiled_graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "can you suggest me some albums?"}
                ],
                "session_id": "test-bare-pref",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=config,
        )

        result = compiled_graph.invoke(
            {
                "messages": [{"role": "user", "content": "i love jazz"}],
                "session_id": "test-bare-pref",
                "intents": [],
            },
            config=config,
        )

        last_message = result["messages"][-1]
        assert "I can only help with questions about our music catalog" not in last_message.content


class TestIdentityCaptureInCatalogOnlyConversation:
    """Regression tests: identity must be captured on ANY turn, not
    only when an invoice question happens to trigger hitl_verify_node
    -- otherwise a catalog-only conversation can never attach saved
    preferences to a real customer, or fetch previously-saved ones."""

    def test_identity_volunteered_mid_conversation_is_captured(self):
        config = {"configurable": {"thread_id": f"test-id-capture-{uuid.uuid4()}"}}

        compiled_graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "do you have any suggestions for me?"}
                ],
                "session_id": "test-id-capture",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=config,
        )

        # This message alone isn't a catalog/invoice request, so the
        # off-topic rejection is expected -- but customer_id must
        # still get captured in the background.
        result = compiled_graph.invoke(
            {
                "messages": [{"role": "user", "content": "my customer id is 43"}],
                "session_id": "test-id-capture",
                "intents": [],
            },
            config=config,
        )

        assert result["customer_id"] == "43"
        assert result["customer_verified"] is True

    def test_preference_stated_before_identity_is_retroactively_saved(self):
        # Reproduces the exact reported scenario: a preference stated
        # BEFORE identity is known must still get saved once identity
        # IS established later in the same conversation, as long as
        # it's still within the recent-history window.
        config = {"configurable": {"thread_id": f"test-retro-save-{uuid.uuid4()}"}}

        compiled_graph.invoke(
            {
                "messages": [{"role": "user", "content": "i love pop music"}],
                "session_id": "test-retro-save",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=config,
        )
        compiled_graph.invoke(
            {
                "messages": [{"role": "user", "content": "my customer id: 43"}],
                "session_id": "test-retro-save",
                "intents": [],
            },
            config=config,
        )

        item = memory_store.get(("memory_profile", "43"), "user_memory")
        assert item is not None
        assert "pop" in item.value.get("music_preferences", [])

    def test_previously_saved_preferences_are_fetched_in_a_new_thread(self):
        # The full end-to-end scenario a user actually experiences:
        # preferences saved under customer 43 in one thread must be
        # fetchable from a completely separate, brand-new thread once
        # that thread also establishes the same identity.
        memory_store.put(
            ("memory_profile", "43"),
            "user_memory",
            {"customer_id": "43", "music_preferences": ["pop"]},
        )

        config = {"configurable": {"thread_id": f"test-cross-thread-{uuid.uuid4()}"}}

        compiled_graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "do you have any suggestions for me?"}
                ],
                "session_id": "test-cross-thread",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=config,
        )
        result = compiled_graph.invoke(
            {
                "messages": [{"role": "user", "content": "my customer id is 43"}],
                "session_id": "test-cross-thread",
                "intents": [],
            },
            config=config,
        )

        # Fetched in the very same turn the identity was established.
        assert result["preferences_context"] == "Music Preferences: pop"


class TestInvoiceAgentDoesNotCommentOnCatalogScope:
    def test_invoice_answer_has_no_contradictory_catalog_disclaimer(self):
        # Regression test: invoice_agent used to proactively decline
        # the catalog part of a mixed question ("I can't help with
        # albums..."), which read as a flat contradiction once joined
        # with catalog_agent's own, correct answer to that exact
        # question in the same reply.
        config = {"configurable": {"thread_id": f"test-no-contradiction-{uuid.uuid4()}"}}

        compiled_graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "do you have Michael Jackson albums and let me know "
                            "the status of my last order"
                        ),
                    }
                ],
                "session_id": "test-no-contradiction",
                "customer_id": None,
                "customer_verified": False,
                "intents": [],
            },
            config=config,
        )
        result = compiled_graph.invoke(
            Command(resume={"customer_id": "43"}), config=config
        )

        answers = [
            m.content for m in result["messages"] if isinstance(m, AIMessage) and m.content
        ]
        invoice_answer = answers[0].lower()

        assert "unable" not in invoice_answer
        assert "can't help" not in invoice_answer
        assert "cannot help" not in invoice_answer
        assert "handled separately" not in invoice_answer