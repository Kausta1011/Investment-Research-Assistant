"""CLI entry point — run investment research on one or more tickers.

Usage:
    python main.py --tickers AAPL
    python main.py --tickers AAPL MSFT NVDA --output ./reports
    python main.py --tickers TSLA --no-cache --model claude-opus-4-6
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from config import REPORTS_DIR, settings
from src.dashboard import render_dashboard
from src.graph import run_research
from src.tools.cache import clear_cache

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # Silence noisy deps
    for noisy in ("urllib3", "yfinance", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def _verdict_color(v: str) -> str:
    return {"BUY": "green", "HOLD": "yellow", "SELL": "red"}.get(v, "white")


def _print_summary(ticker: str, state: dict, dashboard_path: Path) -> None:
    report = state.get("report", {}) or {}
    verdict = (report.get("verdict") or "HOLD").upper()
    confidence = report.get("confidence", "low")

    panel = Panel.fit(
        f"[bold]{ticker}[/bold]  ·  Verdict: [bold {_verdict_color(verdict)}]{verdict}[/]  "
        f"·  Confidence: {confidence}\n"
        f"[dim]{(report.get('thesis') or '')[:280]}[/dim]\n\n"
        f"[cyan]Dashboard:[/cyan] {dashboard_path}",
        title="Investment Brief",
        border_style=_verdict_color(verdict),
    )
    console.print(panel)

    if state.get("errors"):
        console.print(f"[yellow]Warnings during run:[/yellow]")
        for e in state["errors"]:
            console.print(f"  • {e}")


def _print_batch_summary(results: list[dict]) -> None:
    if len(results) <= 1:
        return
    table = Table(title="Batch Results", show_lines=False)
    table.add_column("Ticker", style="bold")
    table.add_column("Verdict")
    table.add_column("Confidence")
    table.add_column("Thesis", overflow="fold", max_width=60)
    for r in results:
        report = r["state"].get("report", {}) or {}
        verdict = (report.get("verdict") or "HOLD").upper()
        table.add_row(
            r["ticker"],
            f"[{_verdict_color(verdict)}]{verdict}[/]",
            report.get("confidence", "low"),
            (report.get("thesis") or "")[:200],
        )
    console.print(table)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-Powered Investment Research Assistant — multi-agent research pipeline."
    )
    p.add_argument("--tickers", "-t", nargs="+", required=True, help="One or more stock tickers (e.g. AAPL MSFT)")
    p.add_argument("--output", "-o", default=str(REPORTS_DIR), help=f"Output directory for HTML reports (default: {REPORTS_DIR})")
    p.add_argument("--no-cache", action="store_true", help="Clear cache before running")
    p.add_argument("--model", help="Override LLM model (e.g. claude-opus-4-6, gpt-4o)")
    p.add_argument("--provider", choices=["anthropic", "openai"], help="Override LLM provider")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    # Allow CLI to override env without editing .env
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    if args.no_cache:
        console.print("[dim]Clearing cache...[/dim]")
        clear_cache()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        settings.validate()
    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        return 2

    results: list[dict] = []
    exit_code = 0

    for ticker in args.tickers:
        ticker = ticker.upper().strip()
        console.rule(f"[bold blue]{ticker}")
        try:
            with console.status(f"Running multi-agent research for {ticker}..."):
                state = run_research(ticker)
        except Exception as e:
            console.print(f"[red]Research failed for {ticker}:[/red] {e}")
            if args.verbose:
                traceback.print_exc()
            exit_code = 1
            continue
        

        try:
            from datetime import datetime
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dashboard_path = render_dashboard(state, output_dir / f"{ticker}_{stamp}.html")
        except Exception as e:
            console.print(f"[red]Dashboard render failed for {ticker}:[/red] {e}")
            if args.verbose:
                traceback.print_exc()
            exit_code = 1
            continue

        _print_summary(ticker, state, dashboard_path)
        results.append({"ticker": ticker, "state": state, "dashboard": dashboard_path})

    _print_batch_summary(results)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
