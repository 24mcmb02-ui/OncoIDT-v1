"""
OncoIDT structured JSON logging configuration.

Usage:
    from shared.logging_config import configure_logging
    configure_logging(settings.service_name, settings.log_level)

Falls back to stdlib logging with a JSON formatter if structlog is not installed.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Stdlib JSON formatter (fallback)
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """Minimal JSON log formatter used when structlog is unavailable."""

    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self._service_name,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

def configure_logging(service_name: str, log_level: str = "INFO") -> None:
    """
    Configure structured JSON logging for the service.

    Attempts to use structlog with JSON renderer. Falls back to stdlib
    logging with a JSON formatter if structlog is not installed.

    Args:
        service_name: Bound to every log entry as the "service" field.
        log_level: Logging level string (e.g. "INFO", "DEBUG", "WARNING").
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )

        # Bind service_name to all log entries
        structlog.contextvars.bind_contextvars(service=service_name)

        # Also configure stdlib root logger so third-party libs emit JSON
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=numeric_level,
        )

        logging.getLogger().setLevel(numeric_level)

    except ImportError:
        # Fallback: stdlib logging with JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter(service_name))
        handler.setLevel(numeric_level)

        root = logging.getLogger()
        root.setLevel(numeric_level)
        # Remove any existing handlers to avoid duplicate output
        root.handlers.clear()
        root.addHandler(handler)

    logging.getLogger(__name__).debug(
        "Logging configured: service=%s level=%s", service_name, log_level
    )
