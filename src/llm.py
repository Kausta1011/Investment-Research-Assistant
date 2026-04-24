"""LLM factory — supports Anthropic (default), OpenAI, and Google Gemini.

Centralizing this makes it trivial to swap providers in one place.
Also provides:
  * Per-task model routing (LLM_MODEL_NEWS / _ANALYST / _SENTIMENT / _REPORT)
  * Global token-bucket rate limiter via `invoke_llm()` — prevents the
    parallel-fanout burst from tripping free-tier 10 RPM limits
  * Fail-fast 429 handling for Gemini (max_retries=0) so a single transient
    quota error doesn't become 5 compounded quota hits.
"""
from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache

from config import settings

logger = logging.getLogger(__name__)


# ─── Global rate limiter ────────────────────────────────────────────────
# Thread-safe sliding-window: we record the timestamp of each call and
# block any new call that would push us past LLM_RPM_LIMIT in the trailing
# 60 seconds. This covers ALL LLM calls across all agents in a run.
_rate_lock = threading.Lock()
_request_times: list[float] = []


def _wait_for_slot() -> None:
    limit = settings.llm_rpm_limit
    if limit <= 0:
        return  # disabled
    while True:
        with _rate_lock:
            now = time.time()
            # drop timestamps that have aged out of the 60s window
            _request_times[:] = [t for t in _request_times if now - t < 60]
            if len(_request_times) < limit:
                _request_times.append(now)
                return
            oldest = _request_times[0]
            wait = 60 - (now - oldest) + 0.2
        logger.info("Rate limiter: sleeping %.1fs (at %d RPM cap)", wait, limit)
        time.sleep(max(wait, 0.5))


def invoke_llm(llm, messages):
    """Rate-limited wrapper around llm.invoke(). Use this everywhere
    instead of calling llm.invoke() directly so the global cap is honored.
    """
    _wait_for_slot()
    return llm.invoke(messages)


# ─── Model factory ──────────────────────────────────────────────────────
@lru_cache(maxsize=32)
def get_llm(task: str = "default", temperature: float = 0.2):
    """Return a LangChain chat model instance based on settings.

    Args:
        task: One of "news", "analyst", "sentiment", "report", or "default".
              Picks a per-task model override if configured via
              LLM_MODEL_NEWS / LLM_MODEL_ANALYST / etc. Falls back to LLM_MODEL.
        temperature: Sampling temperature.

    Cached on (task, temperature) so we reuse clients across agent calls.
    """
    settings.validate()
    model = settings.model_for(task)

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=4096,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": model,
            "api_key": settings.openai_api_key,
            "temperature": temperature,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
            if "openrouter" in settings.openai_base_url:
                kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/investment-research-assistant",
                    "X-Title": "Investment Research Assistant",
                }
        return ChatOpenAI(**kwargs)

    if settings.llm_provider == "google":
        # Google AI Studio free tier. max_retries=0 is CRITICAL: the google-genai
        # SDK's default tenacity retry on 429 burns 5 more quota units per
        # failed call, which is exactly what you DON'T want on a free tier.
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            max_output_tokens=4096,
            max_retries=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
