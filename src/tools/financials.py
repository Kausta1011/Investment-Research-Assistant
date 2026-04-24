"""yfinance wrapper — prices, fundamentals, analyst ratings.

yfinance scrapes Yahoo Finance and is fully free. It can be flaky, so every
call is wrapped in try/except and cached.
"""
from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from src.tools.cache import cached

logger = logging.getLogger(__name__)


def _safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Try each key in order; return first non-None value or default."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


@cached("fundamentals")
def fetch_fundamentals(ticker: str) -> dict:
    """Return fundamentals + identity info for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        logger.warning("yfinance info fetch failed for %s: %s", ticker, e)
        return {"ticker": ticker, "error": str(e)}

    return {
        "ticker": ticker,
        "company_name": _safe_get(info, "longName", "shortName", default=ticker),
        "sector": _safe_get(info, "sector", default="Unknown"),
        "industry": _safe_get(info, "industry", default="Unknown"),
        "summary": _safe_get(info, "longBusinessSummary", default=""),
        "market_cap": _safe_get(info, "marketCap"),
        "pe_ratio": _safe_get(info, "trailingPE", "forwardPE"),
        "eps": _safe_get(info, "trailingEps"),
        "dividend_yield": _safe_get(info, "dividendYield"),
        "beta": _safe_get(info, "beta"),
        "52w_high": _safe_get(info, "fiftyTwoWeekHigh"),
        "52w_low": _safe_get(info, "fiftyTwoWeekLow"),
        "current_price": _safe_get(info, "currentPrice", "regularMarketPrice"),
        "target_mean_price": _safe_get(info, "targetMeanPrice"),
        "recommendation": _safe_get(info, "recommendationKey", default="none"),
        "number_of_analysts": _safe_get(info, "numberOfAnalystOpinions"),
        "revenue": _safe_get(info, "totalRevenue"),
        "profit_margin": _safe_get(info, "profitMargins"),
        "revenue_growth": _safe_get(info, "revenueGrowth"),
        "earnings_growth": _safe_get(info, "earningsGrowth"),
        "debt_to_equity": _safe_get(info, "debtToEquity"),
        "free_cash_flow": _safe_get(info, "freeCashflow"),
    }


@cached("price_history")
def fetch_price_history(ticker: str, period: str = "1y") -> list[dict]:
    """Return daily OHLCV bars. Each item: {date, open, high, low, close, volume}."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
    except Exception as e:
        logger.warning("yfinance history failed for %s: %s", ticker, e)
        return []

    if hist.empty:
        return []

    records = []
    for idx, row in hist.iterrows():
        records.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )
    return records


@cached("analyst_ratings")
def fetch_analyst_ratings(ticker: str) -> list[dict]:
    """Return recent analyst recommendations if available."""
    try:
        t = yf.Ticker(ticker)
        recs = t.recommendations
    except Exception as e:
        logger.warning("yfinance recommendations failed for %s: %s", ticker, e)
        return []

    if recs is None or recs.empty:
        return []

    records = []
    # Recent yfinance versions return a summary df with period/strongBuy/buy/hold/sell/strongSell
    cols = set(c.lower() for c in recs.columns)
    if "strongbuy" in cols or "strong_buy" in cols:
        for _, row in recs.iterrows():
            records.append(
                {
                    "period": str(row.get("period", "")),
                    "strong_buy": int(row.get("strongBuy", 0) or 0),
                    "buy": int(row.get("buy", 0) or 0),
                    "hold": int(row.get("hold", 0) or 0),
                    "sell": int(row.get("sell", 0) or 0),
                    "strong_sell": int(row.get("strongSell", 0) or 0),
                }
            )
    return records
