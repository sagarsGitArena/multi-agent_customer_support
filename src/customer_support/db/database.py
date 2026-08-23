from pathlib import Path
import json
import logging
import threading
import requests

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from customer_support.db.utils import normalize_phone

logger = logging.getLogger(__name__)


##########################################################################
# Configuration
##########################################################################

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SQL_FILE = CACHE_DIR / "Chinook_Sqlite.sql"

CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)


##########################################################################
# Create Engine
##########################################################################

engine = create_engine(
    "sqlite+pysqlite:///:memory:",
    poolclass=StaticPool,
    connect_args={"check_same_thread": False},
    future=True,
)

# StaticPool means every caller shares ONE underlying sqlite3 connection
# (required to keep the in-memory data alive at all -- a second real
# connection would see an empty database). check_same_thread=False only
# lifts Python's same-thread guard; it doesn't make the connection safe
# for two threads to call .execute() on at once. LangGraph's ToolNode
# runs multiple tool calls in real parallel threads, and Gradio's queue
# can serve multiple conversations concurrently too, so every function
# below that touches `engine` must hold this lock for its query.
_engine_lock = threading.Lock()

_database_loaded = False


##########################################################################
# Download SQL Script
##########################################################################

def download_sql_script() -> Path:
    """
    Downloads the Chinook SQL script only once.
    """

    if not SQL_FILE.exists():
        logger.info("Downloading Chinook SQL script from %s", CHINOOK_URL)

        response = requests.get(CHINOOK_URL, timeout=30)
        response.raise_for_status()

        SQL_FILE.write_text(response.text, encoding="utf-8")
        logger.info("Download complete: %s", SQL_FILE)

    else:
        logger.info("Using cached SQL script: %s", SQL_FILE)

    return SQL_FILE


##########################################################################
# Load Database
##########################################################################

def load_database():
    """
    Loads the SQL script into an in-memory SQLite database.
    """

    global _database_loaded

    if _database_loaded:
        logger.debug("Database already loaded, skipping.")
        return

    sql_file = download_sql_script()

    sql_script = sql_file.read_text(encoding="utf-8")

    with engine.begin() as conn:
        raw_conn = conn.connection.driver_connection  # underlying sqlite3.Connection
        raw_conn.executescript(sql_script)

    _database_loaded = True

    logger.info("Chinook database loaded into memory.")

##########################################################################
# Safe Query Execution
##########################################################################

def run_query_safe(sql: str, params: dict | None = None) -> str:
    """
    Executes a query, catching any exception and returning
    a JSON-encoded error payload instead of raising.
    """
    try:
        return execute_query(sql, params)
    except Exception as e:
        logger.exception("Query failed, returning error payload instead.")
        return json.dumps({"error": str(e)})


##########################################################################
# Execute Query
##########################################################################

def execute_query(sql: str, params: dict | None = None) -> str:
    """
    Executes a SQL query using SQLAlchemy parameter binding.

    Returns:
        JSON string
    """

    params = params or {}

    with _engine_lock, engine.connect() as conn:
        result = conn.execute(text(sql), params)
        rows = [dict(row) for row in result.mappings().all()]
        logger.debug("Query returned %d row(s).", len(rows))
        return json.dumps(rows, indent=4)

##########################################################################
# Identity Lookup
##########################################################################

def find_customer_id_by_email(email: str) -> str | None:
    """
    Looks up a customer by email, case-insensitively.

    Returns:
        The customer's ID as a string, or None if no customer has
        that email.
    """

    with _engine_lock, engine.connect() as conn:
        row = conn.execute(
            text("SELECT CustomerId FROM Customer WHERE LOWER(Email) = LOWER(:email)"),
            {"email": email.strip()},
        ).mappings().first()

    return str(row["CustomerId"]) if row else None


def find_customer_id_by_phone(phone: str) -> str | None:
    """
    Looks up a customer by phone number, ignoring formatting
    differences (spaces, dashes, parens, a leading '+') on both sides.

    Stored numbers are heavily punctuated (e.g. "+33 03 80 73 66 99"),
    and there's no reliable way to normalize that SQL-side in SQLite,
    so this fetches the (small, demo-scale) customer list and compares
    normalized digits in Python via the same normalize_phone() used
    elsewhere -- one definition of "same number", not two.
    """

    target = normalize_phone(phone).lstrip("+")
    if not target:
        return None

    with _engine_lock, engine.connect() as conn:
        rows = conn.execute(
            text("SELECT CustomerId, Phone FROM Customer WHERE Phone IS NOT NULL")
        ).mappings().all()

    for row in rows:
        if normalize_phone(row["Phone"]).lstrip("+") == target:
            return str(row["CustomerId"])

    return None


##########################################################################
# Health Check
##########################################################################

def verify_database() -> dict:

    with _engine_lock, engine.connect() as conn:
        tables = conn.execute(text("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """)).scalars().all()
        customer_count = conn.execute(
            text("SELECT COUNT(*) FROM Customer")
        ).scalar_one()

    return {
        "status": "healthy",
        "table_count": len(tables),
        "tables": tables,
        "customer_count": customer_count
    }

# Backwards-compatible alias
database_health_check = verify_database
load_database()