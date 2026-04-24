"""Shared helpers for agent nodes."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict | list | None:
    """Robustly pull a JSON object or array from an LLM response.

    LLMs wrap JSON in ```json fences, add prose, and often emit literal
    newlines/tabs inside string values (especially non-OpenAI/Anthropic
    models like llama via Groq). We try several strategies in order:
      1. Strict parse of the matched block.
      2. strict=False — tolerates raw control chars inside JSON strings.
      3. Manual control-char escape inside string literals, then parse.
    """
    if not text:
        return None
    # Try fenced block first
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    # Try to locate the outermost {...} or [...]
    match = re.search(r"(\{.*\}|\[.*\])", candidate, re.DOTALL)
    if not match:
        return None
    raw = match.group(1)

    # Strategy 1: strict
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: relaxed — allow raw control chars inside strings
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass

    # Strategy 3: aggressively escape control chars inside string literals
    try:
        cleaned = _escape_unescaped_control_chars(raw)
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError as e:
        logger.warning("JSON decode failed: %s", e)
        return None


def _escape_unescaped_control_chars(raw: str) -> str:
    """Walk the JSON text, when inside a "..." string replace raw newlines,
    tabs, and carriage returns with their escaped forms. Backslashes keep
    their normal JSON escape semantics.
    """
    out: list[str] = []
    in_string = False
    escape_next = False
    for ch in raw:
        if escape_next:
            out.append(ch)
            escape_next = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == "\n":
            out.append("\\n")
        elif in_string and ch == "\r":
            out.append("\\r")
        elif in_string and ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return "".join(out)


def record_error(state: dict, agent: str, exc: Exception) -> dict:
    """Return a single-error update. The state's reducer concatenates these
    across parallel agents without raising INVALID_CONCURRENT_GRAPH_UPDATE.
    """
    logger.exception("Agent %s errored", agent)
    return {"errors": [f"{agent}: {type(exc).__name__}: {exc}"]}


# ─── Non-LLM summary builders (used in LLM_MODE=single) ─────────────────
# When the 3 data agents skip their own LLM call, they still need to
# populate *_summary keys so the Report Writer has consistent inputs.
# These heuristic fillers carry the raw signal (headlines, post titles)
# but defer all interpretation to the single Report Writer LLM call.


def heuristic_news_summary(news_items: list[dict]) -> dict:
    return {
        "themes": [],
        "catalysts": [],
        "risk_flags": [],
        "headline_tone": "unknown",
        "top_headlines": [n.get("title", "") for n in (news_items or [])[:5]],
        "note": "Raw headlines only — synthesis deferred to Report Writer (LLM_MODE=single).",
    }


def heuristic_analyst_summary(fundamentals: dict, filings: list[dict]) -> dict:
    return {
        "key_metrics": {
            "valuation": "deferred to Report Writer",
            "growth": "deferred to Report Writer",
            "profitability": "deferred to Report Writer",
            "balance_sheet": "deferred to Report Writer",
        },
        "strengths": [],
        "weaknesses": [],
        "valuation_view": "unclear",
        "filings_of_note": [
            {"form": f.get("form"), "date": f.get("filing_date"), "why": "recent filing"}
            for f in (filings or [])[:3]
        ],
        "note": "Raw data only — synthesis deferred to Report Writer (LLM_MODE=single).",
    }


def heuristic_sentiment_summary(
    reddit_posts: list[dict], analyst_ratings: list[dict]
) -> dict:
    return {
        "score": 0.0,
        "label": "unknown",
        "wsb_tone": "deferred to Report Writer",
        "analyst_tone": "see analyst_ratings array",
        "notable_posts": [p.get("title", "") for p in (reddit_posts or [])[:3]],
        "summary": "Raw data only — synthesis deferred to Report Writer (LLM_MODE=single).",
    }
