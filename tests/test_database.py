import json

import pytest

from customer_support.db import execute_query, run_query_safe, verify_database


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