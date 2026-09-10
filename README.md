# Async Research Assistant

> Ask a research question; the system queries Wikipedia, arXiv, and a web-search API concurrently, then synthesizes one cited answer.

**Topic:** 4 — Async Research Assistant • **Course:** AI-ENG-110 Software Engineering, AI Academy

---

## Quick start

```bash
# 1. Clone & install
git clone <your-repo-url>
cd topic-4-research-assistant
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# 2. Configure
copy .env.example .env          # `cp .env.example .env` on macOS/Linux, then fill in real API keys
# DO NOT commit .env — it is already in .gitignore

# 3. Run the tests
pytest tests/test_ai_smoke.py -v                          # provided, graded smoke tests
pytest --cov=researcher --cov-report=term-missing          # full suite with coverage

# 4. Run the demo (needs LLM + web-search API keys in .env)
python -m researcher demo
python -m researcher ask "What is CRISPR-Cas9 gene editing?"
```

Offline sanity checks that need no API keys at all:

```bash
python demo_ai.py --offline          # exercises the provided ai/ package with canned data
pytest tests/test_ai_smoke.py -v     # all 16 tests, no network
```

## Run with Docker

```bash
docker build -t research-assistant .

# Default command: runs all 5 sample questions end-to-end
docker run --env-file .env research-assistant

# Override the command to ask a single question (no ENTRYPOINT is set, so
# pass the full command, not just the subcommand):
docker run --env-file .env research-assistant python -m researcher ask "What is CRISPR?" --no-cache

# Run the graded smoke tests inside the container
docker run --env-file .env research-assistant pytest tests/test_ai_smoke.py -v
```

## Environment variables

| Variable | Required? | Default | What it controls |
|---|---|---|---|
| `LLM_PROVIDER` | no | `anthropic` | `anthropic` \| `openai` \| `gemini` |
| `LLM_MODEL` | no | provider default | model id passed to the provider |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` | yes (one, matching `LLM_PROVIDER`) | — | key for the chosen LLM provider |
| `WEB_SEARCH_PROVIDER` | no | `tavily` | `tavily` \| `serper` \| `duckduckgo` |
| `TAVILY_API_KEY` / `SERPER_API_KEY` | yes (one, matching `WEB_SEARCH_PROVIDER`) | — | key for the chosen web-search provider (`duckduckgo` needs none) |
| `LOG_LEVEL` | no | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |
| `CACHE_DIR` | no | `./.cache` | filesystem directory for the source cache |
| `CACHE_TTL_SECONDS` | no | `86400` | how long a cached `(source, query)` result stays valid |
| `PER_SOURCE_TIMEOUT_SECONDS` | no | `10` | per-source fetch timeout (covers all retries for that source) |
| `MAX_SOURCES_PER_QUERY` | no | `3` | results requested per source |
| `MAX_CONCURRENT_FETCHES` | no | `5` | semaphore bound on total in-flight external fetches |
| `MAX_QUESTION_LENGTH` | no | `500` | reject questions longer than this |
| `RETRY_MAX_ATTEMPTS` | no | `3` | attempts per `ai.*` call before giving up |
| `RETRY_BACKOFF_SECONDS` | no | `0.5` | base delay for exponential backoff between retries |

The full list with comments is in `.env.example`. **Never commit a real `.env`.**

## CLI usage

```bash
python -m researcher ask "How does CRISPR-Cas9 gene editing work?"
python -m researcher ask "What is fusion energy?" --no-cache
python -m researcher ask "What is photosynthesis?" --sources wiki,arxiv
python -m researcher demo --limit 3
```

Sample output:

```
Q: What is photosynthesis and what are its main stages?

A: Photosynthesis is the process by which plants convert light energy into
chemical energy [1]. The reaction takes place in the chloroplasts and
produces oxygen as a byproduct [2].

References:
  [1] (wikipedia) Photosynthesis
      https://en.wikipedia.org/wiki/Photosynthesis
  [2] (arxiv) Light-Dependent Reactions of Photosynthesis
      https://arxiv.org/abs/...

(timing: wikipedia=0.42s, arxiv=0.61s, web=0.38s; * = cache hit)
```

## Sequential vs concurrent benchmark

```bash
python scripts/bench.py --limit 5
```

| Workload | N | Sequential | Concurrent | Speedup |
|---|---|---|---|---|
| 5 sample research questions | 5 | *run `scripts/bench.py` with real API keys and paste the output here* | | |

`scripts/bench.py` runs the same N questions once with a plain `for` loop and once via `asyncio.gather`, both with `--no-cache` semantics so the numbers reflect real fetch/synthesis concurrency, not cache hits. This development environment has no outbound network access, so the table above is a placeholder — reproduce it locally once `.env` has real `LLM_*`/`TAVILY_API_KEY` values.

**Expected bottleneck:** each `ask` call fans out 3 I/O-bound fetches (Wikipedia, arXiv, web search) concurrently via `asyncio.gather`, so a single call's wall time is bounded by the *slowest* of the three, not their sum — `researcher/concurrency/orchestrator.py` and `tests/test_orchestrator.py::test_sources_fetched_in_parallel_not_sequentially` demonstrate this at the unit level with staggered fake delays. Running multiple `ask` calls concurrently (as the benchmark does) additionally amortizes each call's fixed overhead (LLM synthesis latency, connection setup) across the batch; the `MAX_CONCURRENT_FETCHES` semaphore caps how much of that can happen in parallel before the LLM provider's own rate limits become the bottleneck.

## Testing

```bash
pytest --cov=researcher --cov-report=term-missing
```

- Total coverage on `researcher/`: **94%** (last measured locally; ≥60% required)
- Provided `tests/test_ai_smoke.py`: **16/16 passing, unmodified**
- 64 tests total, all offline — no test touches the live network. `ai.*` calls are mocked via `unittest.mock`/`monkeypatch`; HTTP-layer tests can additionally use `respx`.
- Concurrency is specifically exercised in `tests/test_orchestrator.py`: parallel fan-out timing, graceful degradation when one source raises, per-source timeout isolation, semaphore-bounded concurrency, and cache-hit short-circuiting.
- `mypy researcher --ignore-missing-imports` was run once: **0 errors in `researcher/`**. (`mypy researcher` with no exclusion also follows imports into the provided `ai/providers/*` and reports 27 pre-existing type errors there — those files are unmodified per the assignment's contract.)

## Project layout

```
.
├── ai/                         # PROVIDED — do not modify
├── researcher/
│   ├── config.py                # typed settings (pydantic-settings)
│   ├── models.py                 # SourceOutcome, ResearchSession
│   ├── logging_config.py
│   ├── bootstrap.py              # composition root
│   ├── services/
│   │   ├── ai_service.py         # retry/timeout/logging wrapper around ai.*
│   │   └── cache.py              # TTL-aware (source, query) cache
│   ├── core/
│   │   ├── validation.py
│   │   ├── errors.py
│   │   └── researcher.py         # business logic facade
│   ├── concurrency/
│   │   └── orchestrator.py       # asyncio.gather, per-source timeouts, semaphore
│   ├── storage/
│   │   └── cache_store.py        # CacheBackend ABC + filesystem/in-memory impls
│   └── cli.py                    # `ask` / `demo` subcommands
├── scripts/
│   └── bench.py                  # sequential-vs-concurrent benchmark
├── tests/                        # provided smoke tests + our offline test suite
├── data/                          # sample research questions
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Architecture

```
                     python -m researcher ask "..."
                              |
                              v
                        researcher/cli.py
                              |
                              v
                   researcher/core/researcher.py   (Researcher facade)
                    |  validates question            |
                    v                                 v
   researcher/concurrency/orchestrator.py   researcher/services/ai_service.py
   (asyncio.gather, per-source timeout,        (tenacity retries, timing/
    semaphore, graceful degradation)             debug logging)
                    |                                 |
                    v                                 v
         researcher/services/cache.py  <--->        ai/  (PROVIDED,
         researcher/storage/cache_store.py            fetch_wikipedia/
         (TTL cache: (source,query)->Source[])         fetch_arxiv/fetch_web/
                                                        synthesize)
```

One shared `httpx.AsyncClient` is owned by the orchestrator for its whole lifetime and passed into every `ai.*` fetch, so connections are pooled across questions, not just within one. It is built at composition time over a process-wide SSL context — constructing the trust store costs ~0.35s, which would otherwise be charged to every `ask` call — and released by `Researcher.aclose()` (or `async with researcher:`, which the CLI and benchmark use). Each source's cache check happens before the semaphore is acquired, so cache hits never contend with live fetches for the concurrency budget.

## Limitations

- Only source lists are cached, not final synthesized answers — LLM output isn't deterministic per call, so answer-level caching was out of scope for this assignment's caching requirement.
- Cache expiry is lazy (checked on read); there is no background sweeper removing stale files from `CACHE_DIR`.
- No persistent database — the cache is filesystem JSON, which is sufficient at this scale but wouldn't scale to concurrent multi-process writers.
- The benchmark table above needs to be regenerated with live API keys and network access; the number in this repo's history is a placeholder.

## Tools & acknowledgements

Built with assistance from Claude Code (Anthropic) for the `researcher/` SE layer, test suite, Dockerfile, and this README, working from the `TOPIC.md`/`SOFTWARE_PROJECT.tex` specification. The `ai/` package, sample data, and `tests/test_ai_smoke.py` were provided by the course and are unmodified.

## License

Academic coursework for AI-ENG-110, not a published library.
