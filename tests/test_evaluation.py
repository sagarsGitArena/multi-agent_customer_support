from customer_support.evaluation.groundedness import FALLBACK_MESSAGE, guarded_response, score_response


class TestScoreResponse:
    def test_flags_unstated_taste_characterization_as_ungrounded(self):
        context = (
            "HumanMessage: what did I purchase in my previous orders?\n"
            'AIMessage: Here are your previous purchases: "Whole Lotta Love" '
            "by Led Zeppelin.\n"
            "HumanMessage: any suggestions for me?"
        )
        response = (
            "Since you have diverse taste in music, including rock, "
            "classical, and world music, here are some recommendations..."
        )

        verdict = score_response(context, response)

        assert verdict.grounded is False

    def test_passes_response_that_asks_instead_of_inferring(self):
        # Mirrors what catalog_agent's real system prompt actually
        # injects (agents/catalog_agent.py) -- "no saved preferences" is
        # a grounded claim only because the agent is always told that
        # fact explicitly, not because it's inferred from silence.
        context = (
            "System: Known customer preferences: No saved preferences for "
            "this customer yet.\n"
            "HumanMessage: what did I purchase in my previous orders?\n"
            'AIMessage: Here are your previous purchases: "Whole Lotta Love" '
            "by Led Zeppelin.\n"
            "HumanMessage: any suggestions for me?"
        )
        response = (
            "Since there are no saved preferences for you yet, could you "
            "let me know what kind of music you're interested in?"
        )

        verdict = score_response(context, response)

        assert verdict.grounded is True

    def test_flags_decline_then_answer_contradiction(self):
        context = (
            "HumanMessage: do you have Michael Jackson albums and what's "
            "the status of my last order?\n"
            'ToolMessage: [{"invoice_id": 368, "invoice_date": "2025-06-06", "total": 8.91}]'
        )
        response = (
            "I'm unable to provide information about music or album "
            "availability. However, regarding your order, it was placed on "
            "June 6, 2025, with a total of $8.91. As for Michael Jackson "
            "albums, we don't currently have any in our catalog."
        )

        verdict = score_response(context, response)

        assert verdict.grounded is False

    def test_passes_response_that_answers_only_its_own_part(self):
        context = (
            "HumanMessage: do you have Michael Jackson albums and what's "
            "the status of my last order?\n"
            'ToolMessage: [{"invoice_id": 368, "invoice_date": "2025-06-06", "total": 8.91}]'
        )
        response = (
            "Your last order, placed on June 6, 2025, had a total of $8.91."
        )

        verdict = score_response(context, response)

        assert verdict.grounded is True

    def test_passes_a_grounded_not_found_answer(self):
        context = (
            "HumanMessage: do you have any albums by ThisArtistDoesNotExistZZZ?\n"
            'ToolMessage: {"message": "No albums found for artist: '
            'ThisArtistDoesNotExistZZZ"}'
        )
        response = (
            "It looks like we don't have any albums by that artist in our "
            "catalog right now."
        )

        verdict = score_response(context, response)

        assert verdict.grounded is True

    def test_does_not_treat_purchase_history_as_a_stated_preference(self):
        # Regression test: the judge briefly conflated "the customer has
        # purchase history" with "the customer has a saved preference",
        # which would have defeated the entire point of this guardrail
        # (explicit statements only, never inferred from behavior).
        context = (
            "System: Known customer preferences: No saved preferences for "
            "this customer yet.\n"
            "HumanMessage: what did I purchase in my previous orders?\n"
            'AIMessage: Here are your previous purchases: "Whole Lotta Love" '
            "by Led Zeppelin.\n"
            "HumanMessage: any suggestions for me?"
        )
        response = (
            "There are no saved preferences for you yet, could you tell me "
            "what you like?"
        )

        verdict = score_response(context, response)

        assert verdict.grounded is True


class TestGuardedResponse:
    def test_blocks_and_replaces_an_ungrounded_response(self):
        context = (
            "HumanMessage: what did I purchase in my previous orders?\n"
            'AIMessage: Here are your previous purchases: "Whole Lotta Love" '
            "by Led Zeppelin.\n"
            "HumanMessage: any suggestions for me?"
        )
        bad_response = (
            "Since you have diverse taste in music, including rock, "
            "classical, and world music, here are some recommendations..."
        )

        content, verdict = guarded_response(context, bad_response)

        assert content == FALLBACK_MESSAGE
        assert verdict is not None
        assert verdict.grounded is False

    def test_passes_a_grounded_response_through_unchanged(self):
        context = (
            "System: Known customer preferences: No saved preferences for "
            "this customer yet.\n"
            "HumanMessage: any suggestions for me?"
        )
        good_response = "There are no saved preferences for you yet, could you tell me what you like?"

        content, verdict = guarded_response(context, good_response)

        assert content == good_response
        assert verdict is not None
        assert verdict.grounded is True

    def test_empty_response_passes_through_without_calling_the_judge(self):
        content, verdict = guarded_response("some context", "")

        assert content == ""
        assert verdict is None
