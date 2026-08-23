import concurrent.futures
import json

import pytest

from customer_support.db import (
    execute_query,
    run_query_safe,
    verify_database,
    find_customer_id_by_email,
    find_customer_id_by_phone,
)
import logging

logger = logging.getLogger(__name__)

class TestExecuteQuery:

    def test_basic_select(self):
        result = execute_query("SELECT COUNT(*) AS count FROM Customer")
        data = json.loads(result)

        assert isinstance(data, list)
        assert data[0]["count"] == 59

    def test_parameterized_query(self):
        result = execute_query(
            "SELECT CustomerId FROM Customer WHERE CustomerId = :cid",
            {"cid": 1},
        )
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["CustomerId"] == 1

    def test_empty_results(self):
        result = execute_query(
            "SELECT * FROM Customer WHERE CustomerId = :cid",
            {"cid": -1},
        )

        assert result == "[]"

    def test_valid_json_output(self):
        result = execute_query("SELECT * FROM Genre LIMIT 3")
        data = json.loads(result)  # raises if not valid JSON

        assert isinstance(data, list)
        assert len(data) == 3
        assert "Name" in data[0]


class TestRunQuerySafe:

    def test_valid_query_returns_json(self):
        result = run_query_safe("SELECT COUNT(*) AS count FROM Customer")
        data = json.loads(result)

        assert data[0]["count"] == 59

    def test_invalid_query_returns_error_json_not_exception(self):
        result = run_query_safe("SELECT * FROM NotARealTable")
        data = json.loads(result)  # should not raise

        assert "error" in data


class TestVerifyDatabase:

    def test_returns_healthy_status(self):
        health = verify_database()

        assert health["status"] == "healthy"

    def test_table_count(self):
        health = verify_database()

        assert health["table_count"] == 11
        assert "Customer" in health["tables"]

    def test_customer_count(self):
        health = verify_database()

        assert health["customer_count"] == 59

class TestFindCustomerIdByEmail:

    def test_match(self):
        customer_id = find_customer_id_by_email("isabelle_mercier@apple.fr")

        assert customer_id == "43"

    def test_match_is_case_insensitive(self):
        customer_id = find_customer_id_by_email("ISABELLE_MERCIER@APPLE.FR")

        assert customer_id == "43"

    def test_match_strips_surrounding_whitespace(self):
        customer_id = find_customer_id_by_email("  isabelle_mercier@apple.fr  ")

        assert customer_id == "43"

    def test_no_match_returns_none(self):
        customer_id = find_customer_id_by_email("nobody@nowhere.example")

        assert customer_id is None


class TestFindCustomerIdByPhone:
    # Customer 43's stored number is "+33 03 80 73 66 99".

    def test_match_exact_format(self):
        customer_id = find_customer_id_by_phone("+33 03 80 73 66 99")

        assert customer_id == "43"

    def test_match_ignores_punctuation_differences(self):
        customer_id = find_customer_id_by_phone("+33-03-80-73-66-99")

        assert customer_id == "43"

    def test_match_without_plus_prefix(self):
        customer_id = find_customer_id_by_phone("33 03 80 73 66 99")

        assert customer_id == "43"

    def test_match_with_surrounding_whitespace(self):
        customer_id = find_customer_id_by_phone("  +330380736699  ")

        assert customer_id == "43"

    def test_no_match_returns_none(self):
        customer_id = find_customer_id_by_phone("+1 555 000 0000")

        assert customer_id is None

    def test_empty_input_returns_none(self):
        assert find_customer_id_by_phone("") is None
        assert find_customer_id_by_phone(None) is None


class TestConcurrentQueries:
    """Regression test for a real crash: LangGraph's ToolNode runs
    multiple tool calls from one AIMessage in real parallel threads
    (e.g. checking two saved-preference genres at once), and the DB
    is a single shared SQLite connection (StaticPool -- required to
    keep the in-memory data alive at all) that isn't safe for two
    threads to call .execute() on simultaneously. Without a lock
    serializing access, concurrent queries corrupted each other's
    cursor state and raised IndexError mid-read."""

    def test_concurrent_queries_return_correct_uncorrupted_results(self):
        def run():
            result = execute_query(
                "SELECT TrackId, Name FROM Track ORDER BY TrackId LIMIT 50"
            )
            return json.loads(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(run) for _ in range(40)]
            results = [f.result() for f in futures]

        assert all(len(r) == 50 for r in results)
        assert all(r == results[0] for r in results)

    def test_concurrent_mixed_query_shapes_do_not_interfere(self):
        # Mirrors the real trigger: different queries (not just
        # identical ones) firing at the same time.
        def run_tracks():
            return json.loads(
                execute_query("SELECT TrackId FROM Track ORDER BY TrackId LIMIT 20")
            )

        def run_email_lookup():
            return find_customer_id_by_email("isabelle_mercier@apple.fr")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = (
                [executor.submit(run_tracks) for _ in range(10)]
                + [executor.submit(run_email_lookup) for _ in range(10)]
            )
            results = [f.result() for f in futures]

        track_results = [r for r in results if isinstance(r, list)]
        email_results = [r for r in results if not isinstance(r, list)]

        assert all(len(r) == 20 for r in track_results)
        assert all(r == "43" for r in email_results)


class TestHelloworldPyTest:
    def test_print_helloworld(self):
        print('HelloWorld PyTest')
        logger.info("HelloWorld PyTest")