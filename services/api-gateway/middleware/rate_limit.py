"""
OncoIDT API Gateway — Redis-backed sliding window rate limiter.

Limits: 1000 requests per minute per authenticated client (keyed by user_id).
Returns HTTP 429 with Retry-After header when the limit is exceeded.

Algorithm: sliding window log using a Redis sorted set.
  - Key: rate_limit:{user_id}
  - Members: request timestamps (epoch seconds, float)
  - Score: same timestamp (for range queries)
  - On each request:
      1. Remove members older than now - 60s
      2. Count remaining members
      3. If count >= limit → 429
      4. Else add current timestamp, set TTL 60s

Requirements: 13.4
"""
from __future__ import annotations

import logging
import time

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from shared.redis_client import get_redis

logger = logging.getLogger(__name__)

_RATE_LIMIT = 1000          # requests per window
_WINDOW_SECONDS = 60        # sliding window size


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis sliding-window rate limiter.
    Skips unauthenticated requests (JWT middleware handles those).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        user = getattr(request.state, "user", None)
        if user is None:
            # Not authenticated — let JWT middleware handle it
            return await call_next(request)

        client_id: str = user.user_id
        now = time.time()
        window_start = now - _WINDOW_SECONDS
        redis_key = f"rate_limit:{client_id}"

        try:
            redis = get_redis()
            pipe = redis.pipeline()
            # Remove timestamps outside the window
            pipe.zremrangebyscore(redis_key, "-inf", window_start)
            # Count requests in window
            pipe.zcard(redis_key)
            # Add current request
            pipe.zadd(redis_key, {str(now): now})
            # Reset TTL
            pipe.expire(redis_key, _WINDOW_SECONDS + 1)
            results = await pipe.execute()

            request_count: int = results[1]  # count before adding current

            if request_count >= _RATE_LIMIT:
                retry_after = int(_WINDOW_SECONDS - (now - window_start)) + 1
                logger.warning(
                    "Rate limit exceeded for client=%s count=%d", client_id, request_count
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded: {_RATE_LIMIT} requests per {_WINDOW_SECONDS}s"
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        except Exception as exc:
            # Redis failure — fail open (do not block requests)
            logger.error("Rate limit check failed for client=%s: %s", client_id, exc)

        return await call_next(request)
