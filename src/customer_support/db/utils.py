import re


def normalize_phone(phone: str | None) -> str | None:
    """
    Removes all non-digit characters while preserving
    a leading '+' if present.
    """

    if not phone:
        return ""

    phone = phone.strip()

    keep_plus = phone.startswith("+")

    digits = re.sub(r"\D", "", phone)

    if keep_plus:
        return "+" + digits

    return digits
