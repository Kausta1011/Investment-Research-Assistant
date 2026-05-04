"""Tests for agent nodes. LLM + HTTP are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _fake_llm_response(json_text: str):
    resp = MagicMock()
    resp.content = json_text
    llm = MagicMock()
    llm.invoke.return_value = resp
    return llm



def test_extract_json_from_fenced_block():
    from src.agents._common import extract_json

    text = '```json\n{"verdict": "BUY", "confidence": "high"}\n```'
    assert extract_json(text) == {"verdict": "BUY", "confidence": "high"}



def test_extract_json_from_prose():
    from src.agents._common import extract_json

    text = 'Here is the object: {"a": 1, "b": [1,2,3]} — let me know.'
    assert extract_json(text) == {"a": 1, "b": [1, 2, 3]}


def test_extract_json_handles_none():
    from src.agents._common import extract_json

    assert extract_json("") is None
    assert extract_json("nothing here") is None


def test_news_scout_no_news_returns_empty_summary():
    from src.agents import news_scout as ns

    with patch.object(ns, "fetch_news", return_value=[]):
        out = ns.news_scout_node({"ticker": "FOO"})

    assert out["news_items"] == []
    assert "news_summary" in out
    assert out["news_summary"]["top_headlines"] == []


def test_news_scout_happy_path(sample_news):
    from src.agents import news_scout as ns

    llm = _fake_llm_response('{"themes":["earnings"],"catalysts":[],"risk_flags":[],"headline_tone":"positive","top_headlines":["Apple beats"]}')
    with patch.object(ns, "fetch_news", return_value=sample_news), \
         patch.object(ns, "get_llm", return_value=llm):
        out = ns.news_scout_node({"ticker": "AAPL", "company_name": "Apple Inc."})

    assert out["news_items"] == sample_news
    assert out["news_summary"]["headline_tone"] == "positive"


def test_analyst_happy_path(sample_fundamentals, sample_price_history):
    from src.agents import analyst as a

    llm = _fake_llm_response('{"key_metrics":{"valuation":"fair"},"strengths":["x"],"weaknesses":["y"],"valuation_view":"fair","filings_of_note":[]}')
    with patch.object(a, "fetch_fundamentals", return_value=sample_fundamentals), \
         patch.object(a, "fetch_price_history", return_value=sample_price_history), \
         patch.object(a, "fetch_analyst_ratings", return_value=[]), \
         patch.object(a, "fetch_recent_filings", return_value=[]), \
         patch.object(a, "get_llm", return_value=llm):
        out = a.analyst_node({"ticker": "AAPL"}) #Why hard coded



    assert out["fundamentals"]["ticker"] == "AAPL"
    assert out["analyst_summary"]["valuation_view"] == "fair"
    # Should propagate company_name when state doesn't have one
    assert out["company_name"] == "Apple Inc."


def test_sentiment_judge_happy_path():
    from src.agents import sentiment_judge as s

    reddit_data = [
        {"subreddit": "wallstreetbets", "title": "AAPL rocks", "selftext": "",
         "score": 100, "num_comments": 20, "url": "x", "created_utc": 1}
    ]
    llm = _fake_llm_response('{"score":0.5,"label":"bullish","wsb_tone":"hyped","analyst_tone":"buy","notable_posts":[],"summary":"positive"}')
    with patch.object(s, "fetch_reddit_posts", return_value=reddit_data), \
         patch.object(s, "get_llm", return_value=llm):
        out = s.sentiment_judge_node({"ticker": "AAPL", "analyst_ratings": []})

    assert out["reddit_posts"] == reddit_data
    assert out["sentiment_summary"]["label"] == "bullish"


def test_report_writer_happy_path(sample_state):
    from src.agents import report_writer as rw

    llm = _fake_llm_response("""
```json
{
  "verdict": "BUY",
  "confidence": "high",
  "thesis": "Strong fundamentals.",
  "bull_case": ["margins"],
  "bear_case": ["china"],
  "key_risks": ["regulation"],
  "price_target_view": "upside",
  "time_horizon": "medium (3-12mo)",
  "markdown": "## Thesis\\nStrong.\\n\\n*Not financial advice.*"
}
```
""")
    with patch.object(rw, "get_llm", return_value=llm):
        out = rw.report_writer_node(sample_state)

    assert out["report"]["verdict"] == "BUY"
    assert out["report"]["confidence"] == "high"


def test_report_writer_fallback_on_bad_json(sample_state):
    from src.agents import report_writer as rw

    llm = _fake_llm_response("this is not JSON at all")
    with patch.object(rw, "get_llm", return_value=llm):
        out = rw.report_writer_node(sample_state)

    # Should fall back to HOLD, not crash
    assert out["report"]["verdict"] == "HOLD"
    assert out["report"]["confidence"] == "low"


def test_agent_records_errors_on_tool_failure():
    from src.agents import news_scout as ns

    def boom(*a, **k):
        raise RuntimeError("network down")

    with patch.object(ns, "fetch_news", side_effect=boom):
        out = ns.news_scout_node({"ticker": "FOO"})

    assert out["news_items"] == []
    assert any("news_scout" in e for e in out["errors"])