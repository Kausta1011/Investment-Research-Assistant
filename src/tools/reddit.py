"""Reddit public JSON fetcher.

Reddit exposes a public JSON API at `https://www.reddit.com/r/<sub>/search.json`
that requires no auth (just a polite User-Agent). We search WallStreetBets,
stocks, and investing for mentions of the ticker.
"""
from __future__ import annotations

import logging
import time

import requests

from config import settings
from src.tools.cache import cached

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.reddit.com/r/{sub}/search.json"
_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


def _headers() -> dict:
    return {
        "User-Agent": "InvestmentResearchAssistant/0.1 (contact@example.com)",
        "Accept": "application/json",
    }


@cached("reddit")
def fetch_reddit_posts(ticker: str) -> list[dict]:
    """Search relevant subreddits for the ticker and return recent posts.

    Each item: {subreddit, title, selftext, score, num_comments, url, created_utc}
    """
    items: list[dict] = []
    per_sub = max(1, settings.max_reddit_posts // len(_SUBREDDITS))

    for sub in _SUBREDDITS:
        params = {
            "q": ticker,
            "restrict_sr": "1",
            "sort": "new",
            "limit": per_sub,
            "t": "month",
        }
        url = _SEARCH_URL.format(sub=sub)
        try:
            r = requests.get(url, headers=_headers(), params=params, timeout=10)
            if r.status_code == 429:
                logger.warning("Reddit rate-limited on r/%s; backing off", sub)
                time.sleep(2)
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("Reddit fetch failed on r/%s: %s", sub, e)
            continue

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            title = post.get("title", "")
            # Require ticker to appear to reduce false matches
            if ticker.upper() not in title.upper() and ticker.upper() not in post.get(
                "selftext", ""
            ).upper():
                continue
            items.append(
                {
                    "subreddit": sub,
                    "title": title,
                    "selftext": post.get("selftext", "")[:500],
                    "score": int(post.get("score", 0)),
                    "num_comments": int(post.get("num_comments", 0)),
                    "url": f"https://www.reddit.com{post.get('permalink', '')}",
                    "created_utc": float(post.get("created_utc", 0)),
                }
            )

    items.sort(key=lambda x: x["created_utc"], reverse=True)
    return items[: settings.max_reddit_posts]
