"""Agent 2 — Analyst.

Reads SEC filings (titles + dates) and fundamentals, then summarizes
financial health, strengths, and weaknesses.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.agents._common import (
    extract_json,
    heuristic_analyst_summary,
    record_error,
)
from src.llm import get_llm, invoke_llm
from src.tools.financials import (
    fetch_analyst_ratings,
    fetch_fundamentals,
    fetch_price_history,
)
from src.tools.sec import fetch_recent_filings

logger = logging.getLogger(__name__)

_SYSTEM = """You are an equity research analyst. You will receive a company's
fundamentals and a list of its recent SEC filings. Produce a structured view.

Return ONLY a JSON object with this exact shape:
{
  "key_metrics": {
    "valuation": "one sentence on P/E, market cap, price relative to 52w range",
    "growth": "one sentence on revenue/earnings growth",
    "profitability": "one sentence on margins, cash flow",
    "balance_sheet": "one sentence on leverage / debt-to-equity"
  },
  "strengths": [list of 2-4 short bullets],
  "weaknesses": [list of 2-4 short bullets],
  "valuation_view": "undervalued" | "fair" | "overvalued" | "unclear",
  "filings_of_note": [list of up to 3 filings that look material, each: {form, date, why}]
}

Be honest — if data is missing, say so. Do not invent numbers.
"""


def _format_fundamentals(fund: dict) -> str:
    def fmt(v, suffix=""):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:,.2f}{suffix}"
        if isinstance(v, int):
            return f"{v:,}{suffix}"
        return str(v)

    return f"""- Company: {fund.get('company_name')} ({fund.get('ticker')})
- Sector / Industry: {fund.get('sector')} / {fund.get('industry')}
- Market cap: {fmt(fund.get('market_cap'))}
- Current price: {fmt(fund.get('current_price'))}
- 52w range: {fmt(fund.get('52w_low'))} — {fmt(fund.get('52w_high'))}
- P/E (trailing or forward): {fmt(fund.get('pe_ratio'))}
- EPS: {fmt(fund.get('eps'))}
- Beta: {fmt(fund.get('beta'))}
- Dividend yield: {fmt(fund.get('dividend_yield'))}
- Revenue: {fmt(fund.get('revenue'))}
- Revenue growth (YoY): {fmt(fund.get('revenue_growth'))}
- Earnings growth (YoY): {fmt(fund.get('earnings_growth'))}
- Profit margin: {fmt(fund.get('profit_margin'))}
- Debt/Equity: {fmt(fund.get('debt_to_equity'))}
- Free cash flow: {fmt(fund.get('free_cash_flow'))}
- Analyst rec (Yahoo): {fund.get('recommendation')} ({fund.get('number_of_analysts')} analysts)
- Mean analyst target: {fmt(fund.get('target_mean_price'))}
- Business summary: {(fund.get('summary') or '')[:600]}"""


def _format_filings(filings: list[dict]) -> str:
    if not filings:
        return "No recent SEC filings available."
    return "\n".join(
        f"- {f['form']} filed {f['filing_date']} ({f['primary_doc']})"
        for f in filings[:10]
    )


def analyst_node(state: dict) -> dict:
    ticker = state["ticker"]
    logger.info("[Analyst] fetching fundamentals + filings for %s", ticker)

    try:
        fundamentals = fetch_fundamentals(ticker)
        price_history = fetch_price_history(ticker)
        analyst_ratings = fetch_analyst_ratings(ticker)
        filings = fetch_recent_filings(ticker)
    except Exception as e:
        return record_error(state, "analyst", e)

    # If we got a company_name, propagate it (helps news_scout on 2nd run)
    updates: dict = {
        "fundamentals": fundamentals,
        "price_history": price_history,
        "analyst_ratings": analyst_ratings,
        "sec_filings": filings,
    }
    if fundamentals.get("company_name") and not state.get("company_name"):
        updates["company_name"] = fundamentals["company_name"]

    # In single-call mode, skip the LLM — Report Writer will read raw
    # fundamentals + filings directly from state.
    if settings.llm_mode == "single":
        logger.info("[Analyst] LLM_MODE=single — skipping LLM summary")
        updates["analyst_summary"] = heuristic_analyst_summary(fundamentals, filings)
        return updates

    user_msg = f"""FUNDAMENTALS
{_format_fundamentals(fundamentals)}

RECENT SEC FILINGS
{_format_filings(filings)}

Produce the structured analyst view."""

    try:
        llm = get_llm(task="analyst", temperature=0.1)
        resp = invoke_llm(llm, [SystemMessage(_SYSTEM), HumanMessage(user_msg)])
        summary = extract_json(resp.content) or {}
    except Exception as e:
        updates.update(record_error(state, "analyst", e))
        return updates

    updates["analyst_summary"] = summary
    return updates
