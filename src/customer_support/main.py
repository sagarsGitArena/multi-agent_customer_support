from customer_support.db.database import (
    load_database,
    execute_query,
    database_health_check,
)

from customer_support.db.utils import normalize_phone


def main():

    ###########################################################
    # Load Database
    ###########################################################

    load_database()

    ###########################################################
    # Health Check
    ###########################################################

    print("\nDatabase Health Check")
    print("=" * 60)

    health = database_health_check()

    print(health)

    ###########################################################
    # Query Example
    ###########################################################

    print("\nQuery Example")
    print("=" * 60)

    sql = """
    SELECT
        CustomerId,
        FirstName,
        LastName,
        Phone
    FROM Customer
    WHERE Country = :country
    LIMIT 5
    """

    result = execute_query(
        sql,
        {
            "country": "USA"
        }
    )

    print(result)

    ###########################################################
    # Phone Normalization
    ###########################################################

    print("\nPhone Normalization")
    print("=" * 60)

    numbers = [
        "(302) 555-1234",
        "+1 (302) 555-9999",
        "+44 (20) 7946-0958",
        "555.123.4567",
        None,
    ]

    for n in numbers:
        print(f"{n}  ->  {normalize_phone(n)}")


if __name__ == "__main__":
    main()
