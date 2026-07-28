import json


def not_found(message: str) -> str:
    """
    Builds a JSON string carrying a human-readable not-found message.
    """

    return json.dumps({"message": message})
