"""Agent 4 — Report Writer.

Takes news + analyst + sentiment summaries and produces a polished
investment brief with a Buy/Hold/Sell recommendation.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._common import extract_json, record_error
from src.llm import get_llm, invoke_llm

logger = logging.getLogger(__name__)

_SYSTEM = """You are a senior portfolio analyst writing a concise investment
brief. You synthesize fundamentals, recent news, and public sentiment into a
clear, defensible recommendation. You are NOT a licensed financial advisor.

Return ONLY a JSON object with this exact shape:
{
  "verdict": "BUY" | "HOLD" | "SELL",
  "confidence": "low" | "medium" | "high",
  "thesis": "3-5 sentence investment thesis explaining the recommendation",
  "bull_case": [list of 2-4 bullets],
  "bear_case": [list of 2-4 bullets],
  "key_risks": [list of 2-4 bullets],
  "price_target_view": "one sentence on whether price looks attractive vs analyst targets",
  "time_horizon": "short-term (< 3mo)" | "medium (3-12mo)" | "long (1y+)",
  "markdown": "a polished ~400-word markdown brief with headers (## Thesis, ## Bull Case, ## Bear Case, ## Risks, ## Recommendation). Include a disclaimer at the end."
}

Rules:
- Be decisive but intellectually honest — weigh evidence on both sides.
- If data is sparse or conflicting, set confidence="low" and say so in the thesis.
- Never recommend BUY on purely speculative/meme momentum.
- Always include a disclaimer that this is not financial advice.
"""


def _fmt(obj) -> str:
    """JSON-format a dict for the prompt, stripping Nones."""
    if obj is None:
        return "{}"
    return json.dumps(obj, indent=2, default=str)


def report_writer_node(state: dict) -> dict:
    ticker = state["ticker"]
    logger.info("[Report Writer] synthesizing brief for %s", ticker)

    fund = state.get("fundamentals", {}) or {}
    compact_fund = {
        k: fund.get(k)
        for k in (
            "company_name",
            "sector",
            "industry",
            "market_cap",
            "current_price",
            "pe_ratio",
            "52w_high",
            "52w_low",
            "target_mean_price",
            "recommendation",
            "revenue_growth",
            "earnings_growth",
            "profit_margin",
            "debt_to_equity",
        )
    }

    # Also surface raw data so this prompt works in LLM_MODE=single
    # (where the agent summaries are just placeholders).
    raw_headlines = [
        {"t": n.get("title"), "d": (n.get("published") or "")[:10], "s": n.get("source")}
        for n in (state.get("news_items") or [])[:10]
    ]
    raw_reddit = [
        {"sub": p.get("subreddit"), "t": p.get("title"), "score": p.get("score")}
        for p in (state.get("reddit_posts") or [])[:10]
    ]
    raw_filings = [
        {"form": f.get("form"), "date": f.get("filing_date")}
        for f in (state.get("sec_filings") or [])[:6]
    ]
    raw_ratings = state.get("analyst_ratings") or []

    user_msg = f"""Ticker: {ticker}
Company: {state.get('company_name', ticker)}

FUNDAMENTALS SNAPSHOT
{_fmt(compact_fund)}

NEWS SUMMARY
{_fmt(state.get('news_summary', {}))}

ANALYST SUMMARY
{_fmt(state.get('analyst_summary', {}))}

SENTIMENT SUMMARY
{_fmt(state.get('sentiment_summary', {}))}

RAW HEADLINES (most recent)
{_fmt(raw_headlines)}

RAW REDDIT POSTS
{_fmt(raw_reddit)}

RECENT SEC FILINGS
{_fmt(raw_filings)}

ANALYST RATINGS BREAKDOWN
{_fmt(raw_ratings)}

ERRORS (if any — lowers confidence)
{_fmt(state.get('errors', []))}

Write the investment brief. If the *_SUMMARY sections are placeholders
(they'll say "deferred to Report Writer"), work directly from the raw
data sections above."""

    try:
        llm = get_llm(task="report", temperature=0.3)
        resp = invoke_llm(llm, [SystemMessage(_SYSTEM), HumanMessage(user_msg)])
        report = extract_json(resp.content) or {
            "verdict": "HOLD",
            "confidence": "low",
            "thesis": "The report writer could not produce structured output.",
            "bull_case": [],
            "bear_case": [],
            "key_risks": [],
            "price_target_view": "N/A",
            "time_horizon": "medium (3-12mo)",
            "markdown": resp.content if hasattr(resp, "content") else "",
        }
    except Exception as e:
        return record_error(state, "report_writer", e)

    return {"report": report}
