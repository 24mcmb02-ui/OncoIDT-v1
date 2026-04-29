"""
Property test — Audit log hash chain tamper evidence.

Property 1 (Requirement 17.5):
    Inserting N entries then mutating any entry's `details` field causes
    verify_audit_chain() to return False for all entries after the mutation point.

This test runs entirely in-memory using a fake async session backed by a plain
Python list, so it requires no database connection and runs fast under Hypothesis.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from shared.audit import (
    GENESIS_HASH,
    _compute_hash,
    append_audit_entry,
    verify_audit_chain,
)


# ---------------------------------------------------------------------------
# In-memory fake session
# ---------------------------------------------------------------------------

class _FakeRow:
    """Mimics a SQLAlchemy Row for single-column access."""

    def __init__(self, *values: Any) -> None:
        self._values = values

    def __getitem__(self, idx: int) -> Any:
        return self._values[idx]


class _FakeResult:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchone(self) -> _FakeRow | None:
        return _FakeRow(*self._rows[0]) if self._rows else None

    def fetchall(self) -> list[_FakeRow]:
        return [_FakeRow(*r) for r in self._rows]

    def scalar_one(self) -> Any:
        return self._rows[0][0]


class InMemoryAuditSession:
    """
    Fake AsyncSession that stores audit_log rows in a plain Python list.
    Supports the exact SQL patterns used by shared/audit.py.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._next_id: int = 1

    async def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        sql = str(stmt).strip()

        # _get_last_hash query
        if "SELECT entry_hash FROM audit_log ORDER BY entry_id DESC LIMIT 1" in sql:
            if self._rows:
                return _FakeResult([(self._rows[-1]["entry_hash"],)])
            return _FakeResult([])

        # verify_audit_chain query
        if "SELECT entry_id, timestamp, user_id, action, resource_id" in sql:
            rows = [
                (
                    r["entry_id"],
                    r["timestamp"],
                    r["user_id"],
                    r["action"],
                    r["resource_id"],
                    r["details"],
                    r["prev_hash"],
                    r["entry_hash"],
                )
                for r in self._rows
            ]
            return _FakeResult(rows)

        # INSERT INTO audit_log ... RETURNING entry_id
        if "INSERT INTO audit_log" in sql and params is not None:
            entry_id = self._next_id
            self._next_id += 1
            self._rows.append(
                {
                    "entry_id": entry_id,
                    "timestamp": params["timestamp"],
                    "user_id": params["user_id"],
                    "action": params["action"],
                    "resource_type": params["resource_type"],
                    "resource_id": params["resource_id"],
                    "details": params.get("details"),
                    "prev_hash": params["prev_hash"],
                    "entry_hash": params["entry_hash"],
                }
            )
            return _FakeResult([(entry_id,)])

        raise NotImplementedError(f"Unhandled SQL in fake session:\n{sql}")

    # Alembic / SQLAlchemy compatibility stubs (not used by audit.py)
    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


def _insert_n_entries(session: InMemoryAuditSession, n: int) -> None:
    """Insert N audit entries with deterministic but varied content."""
    ts_base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(n):
        _run(
            append_audit_entry(
                session,  # type: ignore[arg-type]
                user_id=f"user_{i % 5}",
                action="read" if i % 2 == 0 else "write",
                resource_type="patient",
                resource_id=f"patient_{i}",
                details={"index": i, "note": f"entry {i}"},
                timestamp=ts_base.replace(second=i % 60, microsecond=i),
            )
        )


# ---------------------------------------------------------------------------
# Property 1: Hash chain tamper evidence
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=2, max_value=30),
    mutation_idx=st.integers(min_value=0, max_value=29),
    new_details=st.fixed_dictionaries(
        {"tampered": st.booleans(), "value": st.integers(min_value=0, max_value=9999)}
    ),
)
@h_settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_tamper_invalidates_chain_from_mutation_point(
    n: int,
    mutation_idx: int,
    new_details: dict[str, Any],
) -> None:
    """
    Property 1: Inserting N entries then mutating any entry's `details` field
    causes verify_audit_chain() to return False.

    The mutation_idx is clamped to [0, n-1] so it always targets a real row.
    """
    session = InMemoryAuditSession()
    _insert_n_entries(session, n)

    # Sanity: chain must be valid before mutation
    assert _run(verify_audit_chain(session)) is True  # type: ignore[arg-type]

    # Mutate the details of a row at the clamped index
    target_idx = mutation_idx % n
    session._rows[target_idx]["details"] = json.dumps(new_details)

    # Chain must now be invalid
    result = _run(verify_audit_chain(session))  # type: ignore[arg-type]
    assert result is False, (
        f"Expected chain to be invalid after mutating row {target_idx} "
        f"(n={n}), but verify_audit_chain() returned True"
    )


@given(n=st.integers(min_value=1, max_value=30))
@h_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_untampered_chain_is_valid(n: int) -> None:
    """Baseline: an untampered chain of N entries always verifies as True."""
    session = InMemoryAuditSession()
    _insert_n_entries(session, n)
    assert _run(verify_audit_chain(session)) is True  # type: ignore[arg-type]


@given(
    n=st.integers(min_value=2, max_value=20),
    mutation_idx=st.integers(min_value=0, max_value=19),
)
@h_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_entry_hash_mutation_invalidates_chain(n: int, mutation_idx: int) -> None:
    """Directly overwriting entry_hash also breaks the chain."""
    session = InMemoryAuditSession()
    _insert_n_entries(session, n)

    target_idx = mutation_idx % n
    session._rows[target_idx]["entry_hash"] = "deadbeef" * 8  # 64 hex chars

    assert _run(verify_audit_chain(session)) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unit tests for _compute_hash determinism
# ---------------------------------------------------------------------------

def test_compute_hash_is_deterministic() -> None:
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    h1 = _compute_hash(GENESIS_HASH, ts, "user1", "read", "res1", {"k": "v"})
    h2 = _compute_hash(GENESIS_HASH, ts, "user1", "read", "res1", {"k": "v"})
    assert h1 == h2


def test_compute_hash_changes_on_any_field() -> None:
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    base = _compute_hash(GENESIS_HASH, ts, "user1", "read", "res1", None)

    assert _compute_hash("a" * 64, ts, "user1", "read", "res1", None) != base
    assert _compute_hash(GENESIS_HASH, ts, "user2", "read", "res1", None) != base
    assert _compute_hash(GENESIS_HASH, ts, "user1", "write", "res1", None) != base
    assert _compute_hash(GENESIS_HASH, ts, "user1", "read", "res2", None) != base
    assert _compute_hash(GENESIS_HASH, ts, "user1", "read", "res1", {"x": 1}) != base


def test_empty_chain_is_valid() -> None:
    session = InMemoryAuditSession()
    assert _run(verify_audit_chain(session)) is True  # type: ignore[arg-type]
