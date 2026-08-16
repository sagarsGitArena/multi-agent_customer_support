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


class TestHelloworldPyTest:
    def test_print_helloworld(self):
        print('HelloWorld PyTest')
        logger.info("HelloWorld PyTest")