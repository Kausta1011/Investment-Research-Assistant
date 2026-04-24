"""Disk-backed cache wrapper.

Thin wrapper around diskcache so every tool can decorate its fetchers
with `@cached(namespace='news')` and get TTL + keying for free.

The cache handle is initialized lazily so that importing this module
never fails — useful on filesystems where SQLite can't run (certain
FUSE mounts, read-only filesystems). If the configured CACHE_DIR
doesn't support SQLite, we fall back to a tempdir.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

import diskcache

from config import CACHE_DIR, settings

logger = logging.getLogger(__name__)

_cache: diskcache.Cache | None = None


def _get_cache() -> diskcache.Cache:
    """Lazily initialize the cache. Falls back to tempdir on filesystem errors."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        _cache = diskcache.Cache(str(CACHE_DIR))
    except Exception as e:
        fallback = Path(tempfile.gettempdir()) / "ira_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning("Primary cache dir unusable (%s); falling back to %s", e, fallback)
        _cache = diskcache.Cache(str(fallback))
    return _cache


def _make_key(namespace: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps(
        {"ns": namespace, "args": args, "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"{namespace}:{digest}"


def cached(namespace: str, ttl: int | None = None) -> Callable:
    """Cache the decorated function's return value with a TTL.

    Usage:
        @cached("news", ttl=3600)
        def fetch_news(ticker: str) -> list[dict]:
            ...
    """
    effective_ttl = ttl if ttl is not None else settings.cache_ttl_seconds

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if kwargs.pop("_skip_cache", False):
                return fn(*args, **kwargs)
            key = _make_key(namespace, args, kwargs)
            cache = _get_cache()
            try:
                hit = cache.get(key)
                if hit is not None:
                    logger.debug("Cache HIT %s", key)
                    return hit
            except Exception as e:
                logger.debug("Cache read failed (%s); bypassing", e)
            logger.debug("Cache MISS %s", key)
            result = fn(*args, **kwargs)
            try:
                cache.set(key, result, expire=effective_ttl)
            except Exception as e:
                logger.debug("Cache write failed (%s); continuing", e)
            return result

        return wrapped

    return decorator


def clear_cache() -> None:
    """Wipe the entire cache. Used by --no-cache flag."""
    try:
        _get_cache().clear()
    except Exception as e:
        logger.warning("Could not clear cache: %s", e)
