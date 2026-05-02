"""Test fixtures shared across test modules."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `src` importable from tests
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

@pytest.fixture(autouse=True)
def _force_multi_mode():
    """Reset settings.llm_mode to 'multi' for every test so the LLM-stubbed
    paths are exercised regardless of whatever the developer has in their
    local .env file. Individual tests can override by re-patching.
    settings is a frozen dataclass, so we use object.__setattr__.
    """
    from config import settings

    original = settings.llm_mode
    object.__setattr__(settings, "llm_mode", "multi")
    try:
        yield
    finally:
        object.__setattr__(settings, "llm_mode", original)


@pytest.fixture
def sample_news():
    return [
        {
            "title": "Apple beats earnings estimates",
            "link": "https://example.com/a",
            "published": "2026-04-18T10:00:00",
            "source": "Reuters",
            "summary": "Apple reported record services revenue.",
        },
        {
            "title": "Apple launches new iPhone",
            "link": "https://example.com/b",
            "published": "2026-04-17T09:00:00",
            "source": "Bloomberg",
            "summary": "",
        },
    ]


@pytest.fixture
def sample_fundamentals():
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "summary": "Apple designs and sells consumer electronics.",
        "market_cap": 3_000_000_000_000,
        "pe_ratio": 30.5,
        "eps": 6.5,
        "dividend_yield": 0.005,
        "beta": 1.2,
        "52w_high": 210.0,
        "52w_low": 150.0,
        "current_price": 195.0,
        "target_mean_price": 210.0,
        "recommendation": "buy",
        "number_of_analysts": 35,
        "revenue": 400_000_000_000,
        "profit_margin": 0.25,
        "revenue_growth": 0.05,
        "earnings_growth": 0.07,
        "debt_to_equity": 140.0,
        "free_cash_flow": 100_000_000_000,
    }


@pytest.fixture
def sample_price_history():
    return [
        {"date": "2026-04-15", "open": 190, "high": 196, "low": 189, "close": 195, "volume": 1000000},
        {"date": "2026-04-16", "open": 195, "high": 198, "low": 194, "close": 197, "volume": 1100000},
        {"date": "2026-04-17", "open": 197, "high": 199, "low": 195, "close": 196, "volume": 900000},
    ]


@pytest.fixture
def sample_state(sample_news, sample_fundamentals, sample_price_history):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "news_items": sample_news,
        "fundamentals": sample_fundamentals,
        "price_history": sample_price_history,
        "analyst_ratings": [
            {"period": "0m", "strong_buy": 10, "buy": 15, "hold": 8, "sell": 1, "strong_sell": 0},
        ],
        "sec_filings": [
            {
                "form": "10-Q",
                "filing_date": "2026-02-01",
                "accession": "0000320193-26-000001",
                "primary_doc": "aapl-20260101.htm",
                "url": "https://sec.gov/example",
            }
        ],
        "reddit_posts": [
            {
                "subreddit": "wallstreetbets",
                "title": "AAPL to the moon",
                "selftext": "",
                "score": 1000,
                "num_comments": 200,
                "url": "https://reddit.com/r/wsb/x",
                "created_utc": 1713398400,
            }
            
        ],
        "news_summary": {
            "themes": ["earnings beat", "product launch"],
            "catalysts": ["new iPhone launch"],
            "risk_flags": [],
            "headline_tone": "positive",
            "top_headlines": ["Apple beats earnings estimates"],
        },
        "analyst_summary": {
            "key_metrics": {"valuation": "Premium P/E but justified by margins"},
            "strengths": ["Strong cash flow"],
            "weaknesses": ["Slow revenue growth"],
            "valuation_view": "fair",
            "filings_of_note": [],
        },
        "sentiment_summary": {
            "score": 0.6,
            "label": "bullish",
            "wsb_tone": "Retail is excited",
            "analyst_tone": "Consensus buy",
            "notable_posts": ["AAPL to the moon"],
            "summary": "Broadly bullish across retail and professional analysts.",
        },
        "report": {
            "verdict": "BUY",
            "confidence": "medium",
            "thesis": "Apple remains a cash-flow fortress with reasonable valuation.",
            "bull_case": ["Strong services growth", "Premium brand"],
            "bear_case": ["iPhone saturation"],
            "key_risks": ["China exposure"],
            "price_target_view": "Modest upside to analyst targets.",
            "time_horizon": "medium (3-12mo)",
            "markdown": "## Thesis\nApple is a cash-flow fortress.\n\n*Not financial advice.*",
        },
        "errors": [],
    }
