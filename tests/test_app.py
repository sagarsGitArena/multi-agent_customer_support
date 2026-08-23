from customer_support.ui.app import _parse_verification_reply, new_session, send_message


def _drain(generator):
    """Runs a send_message generator to completion, Gradio-style, and
    returns its final yielded (history, msgbox, thread_id, status)."""

    result = None
    for result in generator:
        pass
    return result


class TestSendMessageMixedIntent:
    def test_mixed_intent_turn_surfaces_every_sub_agents_answer(self):
        # Regression test: a mixed-intent turn ("Beatles albums" +
        # "status of my last order") runs both invoice_agent and
        # catalog_agent in sequence, each producing its own final
        # AIMessage. send_message must surface both, not just the
        # last one processed (catalog) while silently dropping the
        # first (invoice).
        thread_id, history, _ = new_session()

        history, _, thread_id, status = _drain(
            send_message(
                "do you have any beatles albums? also what's the status of my last order?",
                history,
                thread_id,
            )
        )
        assert "Waiting for your input" in status

        history, _, thread_id, status = _drain(
            send_message("my customer id is 43", history, thread_id)
        )

        assert status.startswith("✅")
        final_answer = history[-1]["content"].lower()
        # "beatles" is the direct subject of the catalog question, so
        # catalog_agent's answer virtually always names it. "8.91" is
        # the invoice total straight from tool data -- checked instead
        # of a band name like "Led Zeppelin", which the model could
        # occasionally paraphrase away ("the artist from that order")
        # without changing the answer's correctness, causing a flaky
        # false failure on a prose-wording detail this test doesn't
        # actually care about.
        assert "beatles" in final_answer  # catalog_agent's answer present
        assert "8.91" in final_answer  # invoice_agent's answer present too


class TestParseVerificationReply:
    def test_extracts_email(self):
        result = _parse_verification_reply("my email is isabelle_mercier@apple.fr")

        assert result == {"email": "isabelle_mercier@apple.fr"}

    def test_email_takes_priority_over_digits(self):
        # A message can contain both a number and an email (e.g. an
        # address) -- the email is the stronger identifier, so it wins.
        result = _parse_verification_reply(
            "I'm at 123 Main St, email me at jane43@example.com"
        )

        assert result == {"email": "jane43@example.com"}

    def test_falls_back_to_customer_id_when_no_email_present(self):
        result = _parse_verification_reply("my customer id is 43, last name Mercier")

        assert result["customer_id"] == "43"

    def test_no_customer_id_or_email_found(self):
        result = _parse_verification_reply("I don't remember")

        assert result["customer_id"] is None
        assert "email" not in result

    def test_extracts_phone(self):
        result = _parse_verification_reply("my phone is +33 03 80 73 66 99")

        assert result == {"phone": "+33 03 80 73 66 99"}

    def test_extracts_phone_with_parens_and_dashes(self):
        result = _parse_verification_reply("call me at (302) 555-1234")

        assert result == {"phone": "(302) 555-1234"}

    def test_short_digit_run_is_customer_id_not_phone(self):
        # This project's customer IDs are 1-2 digits, well under the
        # phone heuristic's 6-digit floor, so "43" alone must not be
        # swept up as a phone number.
        result = _parse_verification_reply("my customer id is 43")

        assert result["customer_id"] == "43"
        assert "phone" not in result

    def test_email_takes_priority_over_phone(self):
        result = _parse_verification_reply(
            "you can reach me at +33 03 80 73 66 99 or jane@example.com"
        )

        assert result == {"email": "jane@example.com"}