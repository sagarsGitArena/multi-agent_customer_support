from pathlib import Path
import json
import requests

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool


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

_database_loaded = False


##########################################################################
# Download SQL Script
##########################################################################

def download_sql_script() -> Path:
    """
    Downloads the Chinook SQL script only once.
    """

    if not SQL_FILE.exists():
        print("Downloading Chinook SQL script...")

        response = requests.get(CHINOOK_URL, timeout=30)
        response.raise_for_status()

        SQL_FILE.write_text(response.text, encoding="utf-8")
        print("Download complete.")

    else:
        print("Using cached SQL script.")

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
        return

    sql_file = download_sql_script()

    sql_script = sql_file.read_text(encoding="utf-8")

    with engine.begin() as conn:
        print(type(conn.connection))
        raw_conn = conn.connection.driver_connection  # underlying sqlite3.Connection
        raw_conn.executescript(sql_script)


    _database_loaded = True

    print("Chinook database loaded into memory.")

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

    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        rows = [dict(row) for row in result.mappings().all()]
        return json.dumps(rows, indent=4)

##########################################################################
# Health Check
##########################################################################

def verify_database() -> dict:

    with engine.connect() as conn:
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