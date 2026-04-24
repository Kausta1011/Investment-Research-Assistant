"""SEC EDGAR fetcher.

EDGAR is completely free. Requires a descriptive User-Agent per the SEC's
fair-access policy (https://www.sec.gov/os/accessing-edgar-data).
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from config import settings
from src.tools.cache import cached

logger = logging.getLogger(__name__)

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _headers() -> dict:
    return {"User-Agent": settings.sec_user_agent, "Accept": "application/json"}


@cached("sec_ticker_map", ttl=86400)  # ticker map changes rarely
def _ticker_to_cik() -> dict[str, str]:
    """Return a dict of uppercased ticker -> 10-digit zero-padded CIK string."""
    try:
        r = requests.get(_TICKER_MAP_URL, headers=_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("SEC ticker map fetch failed: %s", e)
        return {}

    out: dict[str, str] = {}
    for _, row in data.items():
        ticker = str(row.get("ticker", "")).upper()
        cik = str(row.get("cik_str", "")).zfill(10)
        if ticker and cik:
            out[ticker] = cik
    return out


@cached("sec_filings")
def fetch_recent_filings(ticker: str, max_items: int = 10) -> list[dict]:
    """Return recent SEC filings for the ticker.

    Each item: {form, filing_date, accession, primary_doc, url}.
    Focuses on 10-K, 10-Q, 8-K filings.
    """
    cik_map = _ticker_to_cik()
    cik = cik_map.get(ticker.upper())
    if not cik:
        logger.info("No CIK found for %s", ticker)
        return []

    try:
        url = _SUBMISSIONS_URL.format(cik=cik)
        r = requests.get(url, headers=_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("SEC submissions fetch failed for %s: %s", ticker, e)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    items: list[dict[str, Any]] = []
    target_forms = {"10-K", "10-Q", "8-K", "20-F", "S-1"}
    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form not in target_forms:
            continue
        acc_nodash = acc.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{acc_nodash}/{doc}"
        )
        items.append(
            {
                "form": form,
                "filing_date": date,
                "accession": acc,
                "primary_doc": doc,
                "url": url,
            }
        )
        if len(items) >= max_items:
            break

    return items
