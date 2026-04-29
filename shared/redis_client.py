"""
OncoIDT async Redis connection pool and Redis Streams helpers.

Stream naming convention: onco:{domain}:{event_type}
  e.g. onco:ingestion:patient_event
       onco:inference:score_update
       onco:alert:generated
       onco:graph:edge_update
       onco:ward:state_change
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool

from shared.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool singleton
# ---------------------------------------------------------------------------

_pool: ConnectionPool | None = None


def get_redis_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=50,
            decode_responses=True,
        )
    return _pool


def get_redis() -> Redis:
    """Return a Redis client backed by the shared connection pool."""
    return aioredis.Redis(connection_pool=get_redis_pool())


async def close_redis_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

async def publish_event(
    stream: str,
    payload: dict[str, Any],
    *,
    redis: Redis | None = None,
    maxlen: int | None = None,
) -> str:
    """
    Publish a JSON-serialisable payload to a Redis Stream.
    Returns the message ID assigned by Redis.
    """
    client = redis or get_redis()
    settings = get_settings()
    trim = maxlen if maxlen is not None else settings.redis_stream_max_len
    message_id: str = await client.xadd(
        stream,
        {"data": json.dumps(payload, default=str)},
        maxlen=trim,
        approximate=True,
    )
    return message_id


async def consume_events(
    stream: str,
    group: str,
    consumer: str,
    *,
    count: int = 10,
    block_ms: int = 2000,
    redis: Redis | None = None,
) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
    """
    Async generator that yields (message_id, payload_dict) tuples from a
    Redis Stream consumer group.  Creates the group (from '$') if absent.
    Acknowledges each message after yielding.
    """
    client = redis or get_redis()

    # Ensure consumer group exists
    try:
        await client.xgroup_create(stream, group, id="$", mkstream=True)
    except aioredis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise

    while True:
        results = await client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count,
            block=block_ms,
        )
        if not results:
            continue
        for _stream_name, messages in results:
            for msg_id, fields in messages:
                try:
                    data = json.loads(fields.get("data", "{}"))
                    yield msg_id, data
                    await client.xack(stream, group, msg_id)
                except Exception as exc:
                    logger.error("Failed to process message %s: %s", msg_id, exc)
                    # Do not ack — message will be redelivered
