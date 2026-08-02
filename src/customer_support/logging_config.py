"""
Central logging configuration for the application.

Location: src/customer_support/logging_config.py

Every log record carries its logger name (dotted module path),
function, and line number via LOG_FORMAT, so individual log calls
only need to state what happened -- not where. Configured once, on
first import of the `customer_support` package.
"""

import logging
import os

from dotenv import load_dotenv

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | "
    "%(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: str | None = None) -> None:
    """
    Configures the root logger's format and handlers for the whole
    application. Safe to call multiple times -- only the first call
    takes effect.

    Reads LOG_LEVEL (default "INFO") and, if set, LOG_FILE to also
    write logs to a file alongside the console.
    """

    global _configured
    if _configured:
        return

    load_dotenv()

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_file = os.getenv("LOG_FILE")
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=resolved_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    _configured = True
