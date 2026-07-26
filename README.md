# multi-agent_customer_support
https://github.com/sagarsGitArena/multi-agent_customer_support

commands:

Running the App:

    uv run python -u -m customer_support.main 2>&1 | tee output.log

Running the Test Cases:
    uv run pytest -v

Running the functions command line:
uv run python -c "
from customer_support.db.database import load_database, execute_query
load_database()
print(execute_query(\"SELECT name FROM sqlite_master WHERE type = 'table'\"))
"
Using cached SQL script.
<class 'sqlalchemy.pool.base._ConnectionFairy'>
Chinook database loaded into memory.
[
    {
        "name": "Album"
    },
    {
        "name": "Artist"
    },
    {
        "name": "Customer"
    },
    {
        "name": "Employee"
    },
    {
        "name": "Genre"
    },
    {
        "name": "Invoice"
    },
    {
        "name": "InvoiceLine"
    },
    {
        "name": "MediaType"
    },
    {
        "name": "Playlist"
    },
    {
        "name": "PlaylistTrack"
    },
    {
        "name": "Track"
    }
]


uv add python-dotenv
uv add --dev pytest


cat > /tmp/test_query.py << 'EOF'
from customer_support.db.database import load_database, execute_query

load_database()
print(execute_query("SELECT name FROM sqlite_master WHERE type = 'table'"))
EOF

uv run python /tmp/test_query.py