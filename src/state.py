"""Shared LangGraph state.

Using a TypedDict keeps LangGraph's type inference happy. Each agent node
writes its own sub-key so there are no write conflicts when parallel
branches converge at the report_writer node — EXCEPT for `errors`, which
any agent may append to. That field uses an Annotated reducer so
concurrent appends are merged instead of raising INVALID_CONCURRENT_GRAPH_UPDATE.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def _merge_errors(left: list[str], right: list[str]) -> list[str]:
    """Concat-dedup reducer for the errors list."""
    combined = list(left or []) + list(right or [])
    # Preserve order while removing exact duplicates
    seen: set[str] = set()
    out: list[str] = []
    for e in combined:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


class ResearchState(TypedDict, total=False):
    # Inputs
    ticker: str
    company_name: str

    # Raw data fetched by tools (populated by agents via their tool calls)
    news_items: list[dict]
    fundamentals: dict
    price_history: list[dict]
    analyst_ratings: list[dict]
    sec_filings: list[dict]
    reddit_posts: list[dict]

    # Agent outputs (each agent summarizes its findings)
    news_summary: dict  # {headlines, themes, risk_flags, catalysts}
    analyst_summary: dict  # {key_metrics, strengths, weaknesses, valuation_view}
    sentiment_summary: dict  # {score: -1..1, wsb_tone, analyst_tone, summary}

    # Final
    report: dict  # {verdict, confidence, thesis, risks, price_target_view, markdown}

    # Meta — both need reducers because parallel agents may write to them
    errors: Annotated[list[str], _merge_errors]
    messages: Annotated[list, add_messages]  # Optional LLM trace
