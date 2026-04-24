"""End-to-end graph test with stubbed agents."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_graph_compiles():
    from src.graph import build_graph
    g = build_graph()
    assert g is not None


def test_graph_end_to_end_with_stubs(sample_state):
    """Stub every agent node to return a fragment, confirm state converges."""
    from src import graph as g_mod

    def stub_news(state):
        return {"news_items": [{"title": "x", "link": "y", "published": "2026-01-01", "source": "z"}],
                "news_summary": {"themes": ["a"], "catalysts": [], "risk_flags": [],
                                 "headline_tone": "neutral", "top_headlines": []}}

    def stub_analyst(state):
        return {"fundamentals": {"ticker": state["ticker"], "company_name": "X Corp"},
                "price_history": [], "analyst_ratings": [], "sec_filings": [],
                "analyst_summary": {"valuation_view": "fair"},
                "company_name": "X Corp"}

    def stub_sentiment(state):
        return {"reddit_posts": [],
                "sentiment_summary": {"score": 0, "label": "neutral", "summary": "meh",
                                      "wsb_tone": "q", "analyst_tone": "q"}}

    def stub_report(state):
        return {"report": {"verdict": "HOLD", "confidence": "medium",
                           "thesis": "unclear", "bull_case": [], "bear_case": [],
                           "key_risks": [], "price_target_view": "—",
                           "time_horizon": "medium (3-12mo)", "markdown": "# H"}}

    with patch.object(g_mod, "news_scout_node", side_effect=stub_news), \
         patch.object(g_mod, "analyst_node", side_effect=stub_analyst), \
         patch.object(g_mod, "sentiment_judge_node", side_effect=stub_sentiment), \
         patch.object(g_mod, "report_writer_node", side_effect=stub_report):
        final = g_mod.run_research("TEST")

    assert final["ticker"] == "TEST"
    assert final["report"]["verdict"] == "HOLD"
    assert final["fundamentals"]["ticker"] == "TEST"
    assert final["sentiment_summary"]["label"] == "neutral"


def test_dashboard_renders(tmp_path, sample_state):
    from src.dashboard import render_dashboard

    out = render_dashboard(sample_state, tmp_path / "out.html")
    assert out.exists()
    html = out.read_text()
    # Core assertions
    assert "AAPL" in html
    assert "BUY" in html
    assert "verdict-BUY" in html
    assert "Apple beats earnings" in html
    # Plotly chart JSON should be present
    assert "priceChart" in html
    assert "price-chart" in html


def test_dashboard_handles_empty_price_history(tmp_path, sample_state):
    from src.dashboard import render_dashboard

    state = {**sample_state, "price_history": [], "analyst_ratings": []}
    out = render_dashboard(state, tmp_path / "empty.html")
    assert out.exists()
    html = out.read_text()
    # Both charts should resolve to JS null — no crash
    assert "null" in html


def test_graph_tolerates_all_agents_erroring_in_parallel():
    """Regression: when all three parallel agents error, their concurrent
    writes to 'errors' must not raise INVALID_CONCURRENT_GRAPH_UPDATE."""
    from src import graph as g_mod
    from src.agents._common import record_error

    def stub_news(state):
        return record_error(state, "news_scout", RuntimeError("news down"))

    def stub_analyst(state):
        return record_error(state, "analyst", RuntimeError("yf down"))

    def stub_sentiment(state):
        return record_error(state, "sentiment_judge", RuntimeError("reddit 429"))

    def stub_report(state):
        return {"report": {"verdict": "HOLD", "confidence": "low",
                           "thesis": "All upstream failed.", "bull_case": [],
                           "bear_case": [], "key_risks": [],
                           "price_target_view": "—",
                           "time_horizon": "medium (3-12mo)", "markdown": "# H"}}

    from unittest.mock import patch
    with patch.object(g_mod, "news_scout_node", side_effect=stub_news), \
         patch.object(g_mod, "analyst_node", side_effect=stub_analyst), \
         patch.object(g_mod, "sentiment_judge_node", side_effect=stub_sentiment), \
         patch.object(g_mod, "report_writer_node", side_effect=stub_report):
        final = g_mod.run_research("FAIL")

    # All three errors should be present, merged by the reducer
    assert len(final["errors"]) == 3
    assert any("news_scout" in e for e in final["errors"])
    assert any("analyst" in e for e in final["errors"])
    assert any("sentiment_judge" in e for e in final["errors"])
    # Report still got written
    assert final["report"]["verdict"] == "HOLD"
