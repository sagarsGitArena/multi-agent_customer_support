import pytest

from customer_support.db import load_database


@pytest.fixture(scope="session", autouse=True)
def _ensure_database_loaded():
    load_database()