"""The four agents: News Scout, Analyst, Sentiment Judge, Report Writer."""
from src.agents.analyst import analyst_node
from src.agents.news_scout import news_scout_node
from src.agents.report_writer import report_writer_node
from src.agents.sentiment_judge import sentiment_judge_node

__all__ = [
    "news_scout_node",
    "analyst_node",
    "sentiment_judge_node",
    "report_writer_node",
]
