"""
SourceAdapter Protocol and AdapterRegistry for the OncoIDT ingestion service.

All source adapters must implement the SourceAdapter Protocol.
Register adapters via AdapterRegistry.register() before use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from shared.models import CanonicalRecord


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(valid=True, errors=[])

    @classmethod
    def fail(cls, *errors: str) -> "ValidationResult":
        return cls(valid=False, errors=list(errors))


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol that all ingestion source adapters must satisfy."""

    source_type: str

    def parse(self, raw: bytes) -> list[CanonicalRecord]:
        """Parse raw bytes into a list of CanonicalRecord objects."""
        ...

    def validate(self, record: CanonicalRecord) -> ValidationResult:
        """Validate a single CanonicalRecord; return ValidationResult."""
        ...


class AdapterRegistry:
    """Registry mapping source_type strings to SourceAdapter instances."""

    def __init__(self) -> None:
        self._adapters: dict[str, SourceAdapter] = {}

    def register(self, adapter: SourceAdapter) -> None:
        """Register an adapter. Raises ValueError on duplicate source_type."""
        if adapter.source_type in self._adapters:
            raise ValueError(
                f"Adapter for source_type '{adapter.source_type}' is already registered."
            )
        self._adapters[adapter.source_type] = adapter

    def get_adapter(self, source_type: str) -> SourceAdapter:
        """Return the adapter for the given source_type.

        Raises KeyError if no adapter is registered for that type.
        """
        try:
            return self._adapters[source_type]
        except KeyError:
            available = ", ".join(self._adapters) or "<none>"
            raise KeyError(
                f"No adapter registered for source_type '{source_type}'. "
                f"Available: {available}"
            )

    @property
    def registered_types(self) -> list[str]:
        return list(self._adapters.keys())


# Module-level default registry — services import and register into this
registry = AdapterRegistry()
