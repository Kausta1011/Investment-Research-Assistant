"""Central configuration. Loads from .env and exposes typed settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (if present)
ROOT_DIR = Path(__file__).parent.resolve()
load_dotenv(ROOT_DIR / ".env")

CACHE_DIR = ROOT_DIR / "cache"
REPORTS_DIR = ROOT_DIR / "reports"
CACHE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    # Optional: override OpenAI base URL — used for OpenRouter, Together,
    # Groq, local Ollama, etc. Any OpenAI-compatible endpoint works.
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None

    # Per-task model overrides. Empty string means "use llm_model as fallback".
    # Lets you route cheap/fast agents (news, sentiment) to Flash and
    # smarter agents (analyst, report) to Pro — very useful for dodging
    # free-tier rate limits.
    llm_model_news: str = os.getenv("LLM_MODEL_NEWS", "")
    llm_model_analyst: str = os.getenv("LLM_MODEL_ANALYST", "")
    llm_model_sentiment: str = os.getenv("LLM_MODEL_SENTIMENT", "")
    llm_model_report: str = os.getenv("LLM_MODEL_REPORT", "")

    # ─── Budget controls ────────────────────────────────────────────────
    # LLM_MODE:
    #   "multi"  — each of the 4 agents makes its own LLM call (4 calls/ticker)
    #   "single" — the 3 data agents skip their LLM and return raw data only,
    #              leaving the Report Writer as the sole LLM call (1 call/ticker).
    #              Best for tight free-tier quotas.
    llm_mode: str = os.getenv("LLM_MODE", "multi").lower()
    # Global cap on LLM requests per minute across all agents (0 = disabled).
    # Gemini Flash free tier is ~10 RPM; we default to 8 for headroom.
    llm_rpm_limit: int = int(os.getenv("LLM_RPM_LIMIT", "8"))

    # Data source knobs
    sec_user_agent: str = os.getenv(
        "SEC_USER_AGENT", "Investment Research Assistant contact@example.com"
    )
    max_news_items: int = int(os.getenv("MAX_NEWS_ITEMS", "20"))
    max_reddit_posts: int = int(os.getenv("MAX_REDDIT_POSTS", "25"))

    # Cache
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    def model_for(self, task: str) -> str:
        """Resolve the model slug to use for a given agent task.

        Falls back to the global llm_model if no task-specific override exists.
        """
        override = {
            "news": self.llm_model_news,
            "analyst": self.llm_model_analyst,
            "sentiment": self.llm_model_sentiment,
            "report": self.llm_model_report,
        }.get(task, "")
        return override or self.llm_model

    def validate(self) -> None:
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        if self.llm_provider == "google" and not self.google_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=google but GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )


settings = Settings()
