"""Google News RSS fetcher.

Google News exposes a free RSS endpoint keyed by search query. No auth,
no API key required. We query by company name and ticker and merge results.
"""
from __future__ import annotations

import logging
from datetime import datetime
from urllib.parse import quote_plus

import feedparser

from config import settings
from src.tools.cache import cached

logger = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def _parse_date(entry: dict) -> str:
    """Normalize feedparser's published date to ISO format."""
    parsed = entry.get("published_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6]).isoformat()
        except (TypeError, ValueError):
            pass
    return entry.get("published", "")


@cached("news")
def fetch_news(ticker: str, company_name: str | None = None) -> list[dict]:
    """Return a list of news items about the ticker.

    Each item: {title, link, published, source, summary}
    """
    queries = [ticker]
    if company_name and company_name.lower() != ticker.lower():
        queries.append(company_name)

    items: list[dict] = []
    seen_links: set[str] = set()

    for q in queries:
        url = _RSS_URL.format(query=quote_plus(f"{q} stock"))
        logger.info("Fetching news: %s", url)
        try:
            feed = feedparser.parse(url)
        except Exception as e:  # feedparser is permissive but defensive anyway
            logger.warning("News fetch failed for %s: %s", q, e)
            continue

        for entry in feed.entries[: settings.max_news_items]:
            link = entry.get("link", "")
            if link in seen_links:
                continue
            seen_links.add(link)
            items.append(
                {
                    "title": entry.get("title", "").strip(),
                    "link": link,
                    "published": _parse_date(entry),
                    "source": entry.get("source", {}).get("title", "")
                    if isinstance(entry.get("source"), dict)
                    else entry.get("source", ""),
                    "summary": entry.get("summary", "").strip(),
                }
            )

    items.sort(key=lambda x: x["published"], reverse=True)
    return items[: settings.max_news_items]
