"""Render the final graph state into a self-contained interactive HTML dashboard.

Uses Plotly for charts (served via CDN in the template) and Jinja2 for the
main HTML. Output is a single .html file you can open in any browser.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import REPORTS_DIR

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent


def _fmt_money(v) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.2f}"


def _fmt_num(v, suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def _price_chart_json(price_history: list[dict]) -> str:
    if not price_history:
        return "null"
    dates = [p["date"] for p in price_history]
    closes = [p["close"] for p in price_history]
    trace = go.Scatter(
        x=dates,
        y=closes,
        mode="lines",
        line=dict(color="#60a5fa", width=2),
        fill="tozeroy",
        fillcolor="rgba(96, 165, 250, 0.1)",
        name="Close",
        hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(height=320)
    # Zoom y-axis in a bit — don't start at zero for price charts
    if closes:
        lo, hi = min(closes), max(closes)
        pad = (hi - lo) * 0.1 or 1
        fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig.to_json()


def _analyst_chart_json(ratings: list[dict]) -> str:
    if not ratings:
        return "null"
    periods = [r.get("period", "?") for r in ratings]
    categories = [
        ("Strong Buy", "strong_buy", "#16a34a"),
        ("Buy", "buy", "#22c55e"),
        ("Hold", "hold", "#f59e0b"),
        ("Sell", "sell", "#f97316"),
        ("Strong Sell", "strong_sell", "#ef4444"),
    ]
    traces = []
    for label, key, color in categories:
        traces.append(
            go.Bar(
                name=label,
                x=periods,
                y=[r.get(key, 0) for r in ratings],
                marker_color=color,
                hovertemplate=f"{label}: %{{y}}<extra></extra>",
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(barmode="stack", height=260, showlegend=True,
                      legend=dict(orientation="h", y=-0.2))
    return fig.to_json()


def _sentiment_marker_pct(score) -> float:
    """Map a -1..+1 score to a 0..100 percent position on the sentiment bar."""
    try:
        s = max(-1.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        s = 0.0
    return (s + 1.0) / 2.0 * 100.0


def render_dashboard(state: dict, output_path: Path | None = None) -> Path:
    """Render the final state to an HTML dashboard and return the file path."""
    ticker = state.get("ticker", "UNKNOWN")
    fund = state.get("fundamentals", {}) or {}
    report = state.get("report", {}) or {}
    sentiment = state.get("sentiment_summary", {}) or {}

    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"{ticker}_{stamp}.html"

    # Price delta (close vs 52w range midpoint, rough heuristic)
    price = fund.get("current_price")
    hi, lo = fund.get("52w_high"), fund.get("52w_low")
    if price and hi and lo:
        pct = (price - lo) / (hi - lo) * 100 if hi != lo else 50
        price_delta = f"{pct:.0f}% of 52w range"
    else:
        price_delta = ""

    # Target delta
    target = fund.get("target_mean_price")
    if price and target:
        upside = (target - price) / price * 100
        cls = "pos" if upside >= 0 else "neg"
        target_delta = f'<span class="{cls}">{upside:+.1f}% vs current</span>'
    else:
        target_delta = ""

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    tmpl = env.get_template("template.html")

    verdict = (report.get("verdict") or "HOLD").upper()
    if verdict not in ("BUY", "HOLD", "SELL"):
        verdict = "HOLD"

    html = tmpl.render(
        ticker=ticker,
        company_name=fund.get("company_name") or state.get("company_name", ticker),
        sector=fund.get("sector", "—"),
        industry=fund.get("industry", "—"),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        # Verdict
        verdict=verdict,
        confidence=report.get("confidence", "low"),
        # KPIs
        price=_fmt_money(price),
        price_delta=price_delta,
        market_cap=_fmt_money(fund.get("market_cap")),
        pe=_fmt_num(fund.get("pe_ratio")),
        range_52w=f"{_fmt_money(lo)} — {_fmt_money(hi)}" if lo and hi else "—",
        target=_fmt_money(target),
        target_delta=target_delta,
        sentiment_label=(sentiment.get("label") or "neutral").title(),
        sentiment_score=_fmt_num(sentiment.get("score")),
        # Thesis
        thesis=report.get("thesis", "No thesis available."),
        bull_case=report.get("bull_case", []) or [],
        bear_case=report.get("bear_case", []) or [],
        key_risks=report.get("key_risks", []) or [],
        price_target_view=report.get("price_target_view", "—"),
        time_horizon=report.get("time_horizon", "—"),
        # Sentiment section
        sentiment_marker_pct=_sentiment_marker_pct(sentiment.get("score", 0)),
        sentiment_synthesis=sentiment.get("summary", "—"),
        wsb_tone=sentiment.get("wsb_tone", "—"),
        analyst_tone=sentiment.get("analyst_tone", "—"),
        # Tables
        news_items=state.get("news_items", []) or [],
        news_count=len(state.get("news_items", []) or []),
        filings=state.get("sec_filings", []) or [],
        filings_count=len(state.get("sec_filings", []) or []),
        reddit_posts=state.get("reddit_posts", []) or [],
        reddit_count=len(state.get("reddit_posts", []) or []),
        # Charts (pre-serialized Plotly JSON)
        price_chart_json=_price_chart_json(state.get("price_history", []) or []),
        analyst_chart_json=_analyst_chart_json(state.get("analyst_ratings", []) or []),
        # Errors
        errors=state.get("errors", []) or [],
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard written to %s", output_path)
    return output_path
