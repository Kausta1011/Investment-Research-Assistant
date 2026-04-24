"""Agent 1 — News Scout.

Fetches recent financial news for the ticker and summarizes themes,
catalysts, and risk flags.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.agents._common import (
    extract_json,
    heuristic_news_summary,
    record_error,
)
from src.llm import get_llm, invoke_llm
from src.tools.news import fetch_news

logger = logging.getLogger(__name__)

_SYSTEM = """You are a financial news analyst. Given a batch of recent news
headlines and snippets about a company, produce a concise structured summary.

Return ONLY a JSON object with this exact shape:
{
  "themes": [list of 3-5 short strings describing the dominant news themes],
  "catalysts": [list of upcoming or recent events that could move the stock],
  "risk_flags": [list of concerning items — lawsuits, probes, guidance cuts, etc.],
  "headline_tone": "positive" | "neutral" | "negative" | "mixed",
  "top_headlines": [list of up to 5 of the most material headlines, verbatim]
}
"""


def news_scout_node(state: dict) -> dict:
    ticker = state["ticker"]
    company = state.get("company_name") or ticker
    logger.info("[News Scout] fetching news for %s", ticker)

    try:
        news_items = fetch_news(ticker, company)
    except Exception as e:
        return {"news_items": [], **record_error(state, "news_scout", e)}

    if not news_items:
        logger.info("[News Scout] no news items found")
        return {
            "news_items": [],
            "news_summary": {
                "themes": [],
                "catalysts": [],
                "risk_flags": [],
                "headline_tone": "neutral",
                "top_headlines": [],
                "note": "No recent news found.",
            },
        }

    # In single-call mode, skip the LLM summary — the Report Writer will
    # synthesize directly from the raw news_items list.
    if settings.llm_mode == "single":
        logger.info("[News Scout] LLM_MODE=single — skipping LLM summary")
        return {
            "news_items": news_items,
            "news_summary": heuristic_news_summary(news_items),
        }

    # Build compact context for the LLM
    bullets = "\n".join(
        f"- [{n['published'][:10]}] {n['title']} — {n.get('source', '')}"
        for n in news_items[:20]
    )
    user_msg = f"""Company: {company} (ticker: {ticker})

Recent news items (most recent first):
{bullets}

Summarize per the schema."""

    try:
        llm = get_llm(task="news", temperature=0.1)
        resp = invoke_llm(llm, [SystemMessage(_SYSTEM), HumanMessage(user_msg)])
        summary = extract_json(resp.content) or {}
    except Exception as e:
        return {"news_items": news_items, **record_error(state, "news_scout", e)}

    return {"news_items": news_items, "news_summary": summary}
