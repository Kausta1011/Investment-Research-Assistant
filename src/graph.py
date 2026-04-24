"""LangGraph orchestration.

Graph topology:

                   ┌──> news_scout ──┐
   START ──────────┼──> analyst ─────┼──> report_writer ──> END
                   └──> sentiment ───┘

The three data-gathering agents run in parallel (LangGraph fans them out from
START when they all have START as their only predecessor). Their outputs
converge into report_writer, which emits the final investment brief.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents import (
    analyst_node,
    news_scout_node,
    report_writer_node,
    sentiment_judge_node,
)
from src.state import ResearchState

logger = logging.getLogger(__name__)


def build_graph():
    """Compile and return the research graph."""
    g: StateGraph = StateGraph(ResearchState)

    g.add_node("news_scout", news_scout_node)
    g.add_node("analyst", analyst_node)
    g.add_node("sentiment_judge", sentiment_judge_node)
    g.add_node("report_writer", report_writer_node)

    # Fan out from START
    g.add_edge(START, "news_scout")
    g.add_edge(START, "analyst")
    g.add_edge(START, "sentiment_judge")

    # Fan in to report_writer (LangGraph barriers by default: report_writer
    # won't execute until ALL three upstream nodes complete)
    g.add_edge("news_scout", "report_writer")
    g.add_edge("analyst", "report_writer")
    g.add_edge("sentiment_judge", "report_writer")

    g.add_edge("report_writer", END)

    return g.compile()


def run_research(ticker: str, company_name: str | None = None) -> dict[str, Any]:
    """Run the full research pipeline for one ticker.

    Returns the final graph state, which includes `report`, `fundamentals`,
    `news_items`, and everything else needed to render a dashboard.
    """
    graph = build_graph()
    initial: dict[str, Any] = {
        "ticker": ticker.upper(),
        "errors": [],
        "messages": [],
    }
    if company_name:
        initial["company_name"] = company_name

    logger.info("Running research for %s", ticker)
    final = graph.invoke(initial)
    return final
