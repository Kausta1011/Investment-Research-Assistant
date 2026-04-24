# AI-Powered Investment Research Assistant

A multi-agent system that researches a public stock and generates an interactive HTML investment brief with a Buy / Hold / Sell recommendation. Four specialised agents work over free public data (yfinance, Google News RSS, SEC EDGAR, Reddit), and a report-writer agent synthesises their findings into a structured thesis.

Built with **LangGraph** + **LangChain**, and designed to run end-to-end on a **free-tier LLM key** (Groq / Gemini / OpenRouter) without hitting rate limits.

![verdict](https://img.shields.io/badge/verdict-BUY%20%7C%20HOLD%20%7C%20SELL-blue) ![framework](https://img.shields.io/badge/LangGraph-0.2%2B-green) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Table of contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [The original plan](#the-original-plan)
4. [The free-tier problem](#the-free-tier-problem)
5. [Workarounds that got it running](#workarounds-that-got-it-running)
6. [Changes vs the original architecture](#changes-vs-the-original-architecture)
7. [What would change with unlimited API budget](#what-would-change-with-unlimited-api-budget)
8. [Project layout](#project-layout)
9. [Dashboard features](#dashboard-features)
10. [Caching and error handling](#caching-and-error-handling)
11. [Testing](#testing)
12. [Disclaimer](#disclaimer)

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your LLM key
cp .env.example .env
# edit .env — the simplest zero-cost setup is Groq:
#   LLM_PROVIDER=openai
#   OPENAI_API_KEY=gsk_your_groq_key
#   OPENAI_BASE_URL=https://api.groq.com/openai/v1
#   LLM_MODEL=llama-3.3-70b-versatile
#   LLM_MODE=single

# 3. Run the research pipeline
python main.py --tickers AAPL

# 4. Open the generated dashboard
open reports/AAPL_*.html
```

More CLI options:

```bash
python main.py --tickers AAPL MSFT NVDA          # multiple tickers
python main.py --tickers TSLA --output ~/reports # custom output folder
python main.py --tickers GOOGL --no-cache        # bypass disk cache
python main.py --tickers META --provider openai --model gpt-4o
python main.py --tickers NVDA -v                 # verbose logging
```

---

## Architecture

```
                   ┌── News Scout      (Google News RSS) ──────┐
                   │                                            │
   ticker ──────►  ├── Analyst         (yfinance + SEC EDGAR) ──┼──► Report Writer ──► investment brief
                   │                                            │        (Buy/Hold/Sell + thesis)
                   │                                            │
                   └── Sentiment Judge (Reddit JSON) ───────────┘
```

The three data-gathering agents fan out from `START` and run in parallel. LangGraph barriers `report_writer` so it only fires once all three upstream agents complete.

| Agent | Data sources | Output |
| --- | --- | --- |
| **News Scout** | Google News RSS (no auth) | themes, catalysts, risk flags, headline tone |
| **Analyst** | yfinance (fundamentals, price history, analyst ratings) + SEC EDGAR (10-K / 10-Q / 8-K filings) | key metrics, strengths, weaknesses, valuation view |
| **Sentiment Judge** | Reddit public JSON (r/wallstreetbets, r/stocks, r/investing) + analyst consensus | -1..+1 score, WSB tone, professional tone |
| **Report Writer** | synthesises the above | `BUY` / `HOLD` / `SELL`, thesis, bull case, bear case, key risks |

---

## The original plan

The first design used the textbook LangGraph parallel-agent pattern, one LLM call per agent:

```
[fetch + LLM summarise news]           ─┐
[fetch + LLM summarise fundamentals]   ─┼─► [LLM synthesise final brief]
[fetch + LLM summarise sentiment]      ─┘
```

That is **4 LLM calls per ticker** — three in parallel, one to finish. On a paid Anthropic / OpenAI key it is perfectly clean: each agent owns its prompt, its JSON schema, and its summary of its slice of the data.

Tech choices at the start:

- Anthropic Claude 3.5 Sonnet as the default model.
- OpenAI GPT-4o as a drop-in fallback (same LangChain factory, different env var).
- Multi-provider factory in `src/llm.py` to make swapping trivial.
- All data fetchers cached via `diskcache` with a 1h TTL so you don't re-hit yfinance/SEC/Reddit on every rerun.
- `pytest` suite that mocks HTTP + LLM so tests run in ~1s with no API key.

This worked — on a paid key.

---

## The free-tier problem

To keep this project zero-cost I tried to move everything to free-tier LLMs. That is where the architecture ran into a wall. Three different flavours of failure:

### 1. Parallel fan-out exceeds RPM caps

Gemini Flash free tier is roughly **10 requests per minute**. The agent graph fans out three data agents simultaneously, all of which hit the LLM within the same second. The report writer then hits it a fourth time a few seconds later. That is four calls in under ten seconds — fine on paper, but any retry/backoff inflates this fast.

### 2. The Gemini SDK's hidden retry storm

`langchain-google-genai` wraps `google-genai`, which has a `tenacity` retry decorator on 429 responses by default. A single 429 turns into **five more 429s** over the next few seconds, each of which counts against quota. So one "real" rate-limit event burns ~6 quota units.

### 3. `limit: 0` on a fresh Google key

Even with a brand new Google AI Studio key, Gemini kept returning `RESOURCE_EXHAUSTED — limit: 0 per GenerateRequestsPerMinutePerProjectPerModel-FreeTier`. Not "used up" — **zero allocation**. This appears to happen when the Google Cloud project backing the key has no free-tier allocation in a given region (common for keys created from India or other restricted regions, or on projects that have billing linked).

OpenRouter's free tier wasn't better: `deepseek/deepseek-chat-v3.1:free` started returning 404 mid-development because the free model slugs rotate frequently.

### Net effect

With the original "4 LLM calls per ticker" design, the app could not complete **even a single ticker research run** on a free tier without hitting `RESOURCE_EXHAUSTED` somewhere. The rate limiter would pause and retry forever, or the report writer would time out waiting for upstream agents.

---

## Workarounds that got it running

Instead of fighting the rate limiter, the architecture was reshaped around a hard budget: **1 LLM call per ticker**. Several interlocking changes made this work.

### A. `LLM_MODE=single` — collapse 4 agents to 1 LLM call

Added a new env var `LLM_MODE` with two values:

- `multi` — original behaviour, each agent runs its own LLM.
- `single` — the three data agents **skip their LLM call entirely** and only fetch + return raw data. The report writer becomes the sole LLM call per ticker, receiving the raw headlines, Reddit posts, fundamentals, filings and analyst ratings directly in its prompt.

Implementation:

- Each of `news_scout.py`, `analyst.py`, `sentiment_judge.py` checks `settings.llm_mode == "single"` early and returns immediately with a heuristic placeholder summary (`heuristic_news_summary`, `heuristic_analyst_summary`, `heuristic_sentiment_summary` in `src/agents/_common.py`). The heuristic summaries carry raw signal (top headlines, post titles, recent filings) but defer all interpretation.
- The report writer's prompt was extended to include `RAW HEADLINES`, `RAW REDDIT POSTS`, `RECENT SEC FILINGS`, `ANALYST RATINGS BREAKDOWN` sections so it can reason directly from raw data when the `*_summary` fields are placeholders.

Result: **1 LLM call per ticker instead of 4**. You can research AAPL, MSFT, TSLA, NVDA, GOOG, META and AMZN in a single minute under a 10 RPM cap.

### B. Global sliding-window rate limiter in `src/llm.py`

Added a thread-safe token bucket that records the timestamp of every LLM call and blocks any new call that would push us past `LLM_RPM_LIMIT` in the trailing 60 seconds. All agents go through `invoke_llm(llm, messages)` instead of `llm.invoke(...)` directly, so the cap is enforced globally across the parallel fan-out.

### C. Disable Gemini SDK's retry storm

`ChatGoogleGenerativeAI(..., max_retries=0)` — a single transient 429 no longer becomes five compounded quota hits.

### D. Per-task model routing

Added `LLM_MODEL_NEWS`, `LLM_MODEL_ANALYST`, `LLM_MODEL_SENTIMENT`, `LLM_MODEL_REPORT` env vars and a `settings.model_for(task)` helper. The idea: route the cheap, high-volume agents to Flash and reserve Pro for the final report. Useful when free quotas differ per model.

### E. Provider pivot to Groq

After repeated `limit: 0` from Gemini, switched the recommended setup to **Groq**:

- `llama-3.3-70b-versatile` — 30 RPM, 14,400 RPD free tier, genuinely zero-cost.
- OpenAI-compatible endpoint, so it plugs into the existing `ChatOpenAI` factory just by setting `OPENAI_BASE_URL=https://api.groq.com/openai/v1`.
- Much better free-tier headroom than Gemini and no regional-allocation surprises.

### F. Resilient JSON extraction for llama outputs

Llama-family models emit JSON with **literal newlines inside string values**, which trips `json.loads`'s strict parser (`Invalid control character at: line 22 column 16`). Rewrote `extract_json()` in `src/agents/_common.py` with three fallback strategies:

1. Strict `json.loads`.
2. Relaxed `json.loads(..., strict=False)` — tolerates raw control chars inside strings.
3. Aggressive manual pass that walks the string character-by-character and escapes unescaped `\n`, `\r`, `\t` inside `"..."` string literals before parsing.

This made the pipeline robust to any OpenAI-compatible provider, not just the well-behaved ones.

### G. Test isolation from local `.env`

The single-mode refactor caused three agent tests to fail because the local `LLM_MODE=single` was bleeding into the test suite and making agents skip their stubbed LLM paths. Fixed with an autouse fixture in `tests/conftest.py` that flips `settings.llm_mode` to `"multi"` for every test using `object.__setattr__` (since `settings` is a frozen dataclass), then restores on teardown.

---

## Changes vs the original architecture

| Concern | Original | Now |
| --- | --- | --- |
| LLM calls per ticker | 4 (3 parallel + 1 synthesis) | **1** in `single` mode, 4 in `multi` mode |
| Agent interpretation | Each agent LLM-summarises its slice | Data agents collect raw data + heuristic placeholder; Report Writer does all synthesis |
| Rate limiting | None (relied on provider retries) | Global thread-safe sliding-window limiter (`LLM_RPM_LIMIT`) |
| Provider SDK retries | Default (can compound 429s) | `max_retries=0` for Google; no stacking retries |
| Supported providers | Anthropic, OpenAI | Anthropic, OpenAI, Google, **Groq / OpenRouter / Together / Ollama** via `OPENAI_BASE_URL` |
| Per-task model routing | — | `LLM_MODEL_NEWS/ANALYST/SENTIMENT/REPORT` overrides |
| JSON parsing | Strict `json.loads` with fence-strip | 3-strategy fallback tolerant of control-chars |
| Test isolation | Implicit | Autouse fixture forcing `llm_mode=multi` |
| Recommended default | Anthropic Claude | Groq `llama-3.3-70b-versatile` in `single` mode |

Functionally the product didn't change: same four agents, same dashboard, same BUY/HOLD/SELL output. What changed is the **budget shape** — we went from "one big LLM per agent" to "one big LLM per ticker" with the data agents becoming pure data fetchers.

---

## What would change with unlimited API budget

If rate limits weren't a constraint (paid Anthropic / OpenAI key, or self-hosted), the system would look closer to the original vision — and then some:

**Keep `LLM_MODE=multi` as the default.** Each agent gets its own focused prompt and its own JSON schema. Better separation of concerns: the News Scout's prompt is tuned for catalyst extraction, the Analyst's for financial reasoning, the Sentiment Judge's for crowd-psychology calibration. Cramming all of that into the Report Writer's prompt works but is coarser.

**Use a stronger model for synthesis.** Claude Opus or GPT-4o for the report writer, a cheaper/faster model for the three data agents. The per-task routing plumbing is already there — just point `LLM_MODEL_REPORT` at the expensive model.

**Add a reflection / critique pass.** A fifth agent reads the Report Writer's output and challenges the thesis (devil's advocate). If the critique surfaces new evidence, re-run the report. ~2x LLM cost per ticker but dramatically improves intellectual honesty.

**More granular agent fan-out.** Split the Analyst into a Fundamentals agent (yfinance) and a Filings agent (SEC EDGAR). Split News into a Recent-News agent and a Catalysts agent (earnings dates, product launches). Each gets its own model call and clean schema — 6-8 calls per ticker instead of 4.

**Tool-use / function calling.** Instead of baking data into the prompt, let agents call the data tools on demand (yfinance, SEC search, Reddit search) via LangChain tool bindings. Each agent becomes a ReAct loop that can ask follow-up questions of the data ("what were Q3 earnings? ok, so how did that compare to Q2?"). Many more LLM calls, much sharper reasoning.

**Streaming + live dashboard.** Stream partial results into the dashboard as each agent completes, instead of waiting for the whole graph to finish. Low value on a free tier, high value for a paid production setup.

**Drop the heuristic placeholders.** They exist solely to keep `single` mode compatible with the downstream report writer. With unlimited budget they can go and the code path simplifies.

**Embed-based news/filings ranking.** Instead of "top 10 most recent headlines", use embeddings to rank by relevance to the ticker's current narrative. Adds an embedding API cost per run, much better signal-to-noise in the prompt.

**Backtesting harness.** Run the pipeline on historical snapshots (e.g. "what would the agent have said about NVDA on 2022-06-01?") and compare against subsequent stock performance. LLM-expensive but the honest way to quantify whether the system has any predictive value beyond "reads like a real analyst report".

---

## Project layout

```
.
├── main.py                   # CLI entry point
├── config.py                 # Settings loaded from .env (frozen dataclass)
├── requirements.txt
├── .env.example
├── src/
│   ├── graph.py              # LangGraph orchestration
│   ├── state.py              # TypedDict state shared across agents
│   ├── llm.py                # Multi-provider factory + rate limiter
│   ├── agents/
│   │   ├── news_scout.py       # Agent 1
│   │   ├── analyst.py          # Agent 2
│   │   ├── sentiment_judge.py  # Agent 3
│   │   ├── report_writer.py    # Agent 4
│   │   └── _common.py          # extract_json, heuristics, error records
│   ├── tools/
│   │   ├── news.py           # Google News RSS
│   │   ├── financials.py     # yfinance wrapper
│   │   ├── sec.py            # SEC EDGAR JSON API
│   │   ├── reddit.py         # Reddit public JSON API
│   │   └── cache.py          # Disk cache with TTL
│   └── dashboard/
│       ├── generator.py      # Jinja2 + Plotly renderer
│       └── template.html     # Self-contained dashboard template
├── tests/                    # pytest suite (mocked HTTP + LLM)
├── cache/                    # disk cache (gitignored)
└── reports/                  # generated HTML dashboards (gitignored)
```

---

## Dashboard features

The generated HTML is a **single self-contained file** — no server, no build step. Open it in any browser:

- Headline **Buy / Hold / Sell** verdict with confidence level
- Six KPI cards (price, market cap, P/E, 52w range, analyst target, sentiment)
- Interactive Plotly **price chart** (1Y close with hover tooltips)
- Full **investment thesis** with bull case, bear case, key risks
- Stacked-bar **analyst consensus** chart
- Sentiment gauge bar (-1 bearish → +1 bullish)
- Tabbed source tables: News, SEC Filings, Reddit posts — all with live text filtering
- Warning panel if any agent hit errors during the run
- Disclaimer footer (this is not financial advice)

---

## Caching and error handling

Every external data fetch is decorated with `@cached(namespace, ttl)`. The cache is a SHA-256 hash of `(namespace, args, kwargs)` stored in a local `diskcache` SQLite DB under `./cache/`. Default TTL is 1 hour (tunable via `CACHE_TTL_SECONDS` in `.env`). Bypass per call with `_skip_cache=True`, or wipe entirely with `--no-cache`.

The pipeline is built to be **partial-failure tolerant**. If the SEC API is rate-limited, Reddit returns 429, or yfinance can't find a ticker, the affected agent records an error to `state["errors"]` and returns what it has. The Report Writer then produces a lower-confidence brief, and the dashboard surfaces a yellow warning panel listing what went wrong.

---

## Testing

```bash
pytest -q
```

All network calls and LLM invocations are mocked, so tests run in under 2 seconds with no API key required. The suite covers cache TTL behaviour, each tool's happy + error paths, each agent's LLM-stubbed happy path, graph end-to-end execution, and dashboard rendering.

An autouse fixture in `tests/conftest.py` pins `settings.llm_mode = "multi"` for every test so a developer's local `LLM_MODE=single` in `.env` doesn't alter test semantics.

---

## Disclaimer

This project is for **educational and research purposes only**. It is not investment advice and should not be used as the sole basis for any financial decision. Always consult a licensed financial advisor. Past performance does not predict future results.
