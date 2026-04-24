"""Tests for the data-source tools. Network calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_cache_roundtrip(tmp_path, monkeypatch):
    """The @cached decorator should cache and invalidate properly."""
    import diskcache

    from src.tools import cache as cache_mod

    # Point the lazy loader at a clean temp-dir cache
    monkeypatch.setattr(cache_mod, "_cache", diskcache.Cache(str(tmp_path)))

    calls = {"n": 0}

    @cache_mod.cached("test_ns", ttl=60)
    def slow(x):
        calls["n"] += 1
        return x * 2

    assert slow(3) == 6
    assert slow(3) == 6  # cached
    assert calls["n"] == 1
    assert slow(4) == 8  # different args
    assert calls["n"] == 2

    # Skip cache flag
    assert slow(3, _skip_cache=True) == 6
    assert calls["n"] == 3


def test_news_parses_feedparser_output():
    from src.tools import news as news_mod

    fake_feed = MagicMock()
    fake_feed.entries = [
        {
            "title": "Foo Corp beats",
            "link": "https://news.example/1",
            "published_parsed": (2026, 4, 18, 10, 0, 0, 0, 0, 0),
            "published": "Fri, 18 Apr 2026",
            "source": {"title": "Reuters"},
            "summary": "blurb",
        },
        {
            "title": "Foo Corp launches",
            "link": "https://news.example/2",
            "published_parsed": (2026, 4, 17, 9, 0, 0, 0, 0, 0),
            "published": "Thu, 17 Apr 2026",
            "source": {"title": "Bloomberg"},
            "summary": "",
        },
    ]

    with patch.object(news_mod.feedparser, "parse", return_value=fake_feed):
        items = news_mod.fetch_news("FOO", "Foo Corp", _skip_cache=True)

    assert len(items) >= 2
    assert items[0]["title"].startswith("Foo Corp")
    assert items[0]["published"].startswith("2026-04-18")
    assert {i["link"] for i in items}.issuperset(
        {"https://news.example/1", "https://news.example/2"}
    )


def test_financials_handles_missing_keys():
    """fetch_fundamentals should return a well-shaped dict even if yfinance fails."""
    from src.tools import financials as f

    class FakeTicker:
        @property
        def info(self):
            return {"longName": "Bar Inc.", "marketCap": 1_000_000}

    with patch.object(f.yf, "Ticker", return_value=FakeTicker()):
        out = f.fetch_fundamentals("BAR", _skip_cache=True)

    assert out["ticker"] == "BAR"
    assert out["company_name"] == "Bar Inc."
    assert out["market_cap"] == 1_000_000
    assert out["pe_ratio"] is None  # missing


def test_financials_handles_exception_gracefully():
    from src.tools import financials as f

    def boom(*a, **k):
        raise RuntimeError("boom")

    with patch.object(f.yf, "Ticker", side_effect=boom):
        out = f.fetch_fundamentals("BAR", _skip_cache=True)

    assert out["ticker"] == "BAR"
    assert "error" in out


def test_sec_returns_empty_when_no_cik():
    from src.tools import sec

    with patch.object(sec, "_ticker_to_cik", return_value={}):
        out = sec.fetch_recent_filings("NONEXISTENT", _skip_cache=True)
    assert out == []


def test_reddit_filters_to_mentioning_posts():
    from src.tools import reddit as r

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": {
            "children": [
                {"data": {"title": "AAPL is great", "selftext": "x", "score": 10,
                          "num_comments": 5, "permalink": "/r/x/1", "created_utc": 1}},
                {"data": {"title": "unrelated post", "selftext": "no ticker", "score": 5,
                          "num_comments": 1, "permalink": "/r/x/2", "created_utc": 2}},
            ]
        }
    }

    with patch.object(r.requests, "get", return_value=fake_resp):
        out = r.fetch_reddit_posts("AAPL", _skip_cache=True)

    # Only the AAPL-mentioning post should pass the filter
    assert any("AAPL" in p["title"] for p in out)
    assert not any("unrelated" in p["title"] for p in out)
