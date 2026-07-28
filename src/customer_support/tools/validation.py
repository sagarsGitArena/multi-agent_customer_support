import json


def parse_id(value: str, field_name: str) -> tuple[int | None, str | None]:
    """
    Parses a numeric ID passed in as a string.

    Returns:
        (parsed_id, None) on success, or (None, error_json) if the
        value is not a valid integer.
    """

    try:
        return int(str(value).strip()), None
    except (TypeError, ValueError):
        error = json.dumps(
            {"error": f"Invalid {field_name}: '{value}' is not a valid number."}
        )
        return None, error
