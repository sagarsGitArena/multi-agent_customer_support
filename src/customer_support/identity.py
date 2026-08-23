"""
Shared identity extraction/resolution helpers.

Location: src/customer_support/identity.py

Used by hitl_verify_node (graph/build.py) to opportunistically resolve
identity from the message that triggers the invoice flow, and by
load_memory_node (agents/memory.py) to do the same on EVERY turn --
that's what lets a customer who volunteers "my customer id is 43" in a
catalog-only conversation (which never reaches hitl_verify_node at
all) still get identified, so their preferences can be saved to and
fetched from the right profile.

Lives here, in neither of those modules, specifically to avoid a
circular import: graph/build.py already imports from agents/memory.py,
so agents/memory.py can't import identity helpers back from
graph/build.py.
"""

import logging
import re

from customer_support.db import find_customer_id_by_email, find_customer_id_by_phone

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"[+(]?\d[\d\s\-().]{5,}\d")
# Deliberately stricter than a bare digit match: this runs against
# arbitrary messages (e.g. "where's invoice 43?"), not a dedicated
# reply to "what's your ID?", so it only fires on an explicitly
# labeled ID -- "my customer id is 43" -- not any number in the text.
EXPLICIT_CUSTOMER_ID_PATTERN = re.compile(r"customer[\s']*s?\s*id\D{0,10}(\d+)", re.IGNORECASE)


def extract_email(text: str) -> str | None:
    match = EMAIL_PATTERN.search(text)
    return match.group() if match else None


def extract_phone(text: str) -> str | None:
    match = PHONE_PATTERN.search(text)
    if match and len(re.sub(r"\D", "", match.group())) >= 6:
        return match.group()
    return None


def extract_identity_from_message(text: str) -> dict:
    """Opportunistically pulls a usable identifier straight out of an
    arbitrary message. Empty dict if nothing usable is found."""

    email = extract_email(text)
    if email:
        return {"email": email}

    phone = extract_phone(text)
    if phone:
        return {"phone": phone}

    match = EXPLICIT_CUSTOMER_ID_PATTERN.search(text)
    if match:
        return {"customer_id": match.group(1)}

    return {}


def resolve_customer_id(verification_input: dict) -> str | None:
    """Turns a {"customer_id"|"email"|"phone": ...} payload into an
    actual customer_id. The three aren't checked with equal rigor: a
    customer_id is trusted as-is (stub -- confirms *a* value was
    given, not that it belongs to the person typing), while email and
    phone are actually looked up against the Customer table --
    case-insensitively for email, formatting-insensitively for phone
    -- so either only resolves if it matches a real account."""

    customer_id = verification_input.get("customer_id")
    email = verification_input.get("email")
    phone = verification_input.get("phone")

    if not customer_id and email:
        customer_id = find_customer_id_by_email(email)
        logger.info(
            "resolve_customer_id: looked up email=%r -> customer_id=%s", email, customer_id
        )

    if not customer_id and phone:
        customer_id = find_customer_id_by_phone(phone)
        logger.info(
            "resolve_customer_id: looked up phone=%r -> customer_id=%s", phone, customer_id
        )

    return customer_id