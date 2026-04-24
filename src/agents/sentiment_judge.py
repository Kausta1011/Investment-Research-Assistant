"""Agent 3 — Sentiment Judge.

Scans Reddit (WSB, stocks, investing) + analyst ratings to gauge public sentiment.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.agents._common import (
    extract_json,
    heuristic_sentiment_summary,
    record_error,
)
from src.llm import get_llm, invoke_llm
from src.tools.reddit import fetch_reddit_posts

logger = logging.getLogger(__name__)

_SYSTEM = """You are a social sentiment analyst. You will receive Reddit posts
mentioning a ticker along with summary analyst rating data. Gauge the overall
public sentiment.

Return ONLY a JSON object with this exact shape:
{
  "score": a number from -1.0 (very bearish) to +1.0 (very bullish),
  "label": "bearish" | "neutral" | "bullish",
  "wsb_tone": short sentence describing WSB / retail tone,
  "analyst_tone": short sentence describing professional analyst consensus,
  "notable_posts": [list of up to 3 representative post titles],
  "summary": "2-3 sentence synthesis"
}

Be calibrated. A small number of low-score posts ≠ strong sentiment.
"""


def _format_reddit(posts: list[dict]) -> str:
    if not posts:
        return "No Reddit posts found mentioning the ticker."
    lines = []
    for p in posts[:20]:
        snippet = (p.get("selftext") or "")[:160].replace("\n", " ")
        lines.append(
            f"- [r/{p['subreddit']}] {p['title']} "
            f"(score {p['score']}, {p['num_comments']} comments) — {snippet}"
        )
    return "\n".join(lines)


def _format_analyst_ratings(ratings: list[dict]) -> str:
    if not ratings:
        return "No analyst rating breakdown available."
    lines = []
    for r in ratings[:4]:
        lines.append(
            f"- {r.get('period', '?')}: "
            f"strongBuy={r.get('strong_buy')}, buy={r.get('buy')}, "
            f"hold={r.get('hold')}, sell={r.get('sell')}, "
            f"strongSell={r.get('strong_sell')}"
        )
    return "\n".join(lines)


def sentiment_judge_node(state: dict) -> dict:
    ticker = state["ticker"]
    logger.info("[Sentiment Judge] scanning Reddit for %s", ticker)

    try:
        reddit_posts = fetch_reddit_posts(ticker)
    except Exception as e:
        return record_error(state, "sentiment_judge", e)

    analyst_ratings = state.get("analyst_ratings", []) or []

    # In single-call mode, skip the LLM — Report Writer will read raw
    # reddit_posts + analyst_ratings directly from state.
    if settings.llm_mode == "single":
        logger.info("[Sentiment Judge] LLM_MODE=single — skipping LLM summary")
        return {
            "reddit_posts": reddit_posts,
            "sentiment_summary": heuristic_sentiment_summary(
                reddit_posts, analyst_ratings
            ),
        }

    user_msg = f"""Ticker: {ticker}

REDDIT POSTS (recent, filtered for ticker mention)
{_format_reddit(reddit_posts)}

ANALYST RATING BREAKDOWN
{_format_analyst_ratings(analyst_ratings)}

Produce the sentiment object."""

    updates: dict = {"reddit_posts": reddit_posts}

    try:
        llm = get_llm(task="sentiment", temperature=0.2)
        resp = invoke_llm(llm, [SystemMessage(_SYSTEM), HumanMessage(user_msg)])
        summary = extract_json(resp.content) or {}
    except Exception as e:
        updates.update(record_error(state, "sentiment_judge", e))
        return updates

    updates["sentiment_summary"] = summary
    return updates
