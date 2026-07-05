# News Research Tool

The `research-news` virtual tool collects, summarises, and relevance-filters
**recent** news articles from a configurable whitelist of domains and RSS/Atom
feeds. Unlike the interactive [`web-research`](WEB_RESEARCH.md) crawler, it runs
a fixed linear pipeline and persists everything it downloads to the knowledge
base. It is exposed three ways:

1. **CLI**: [`pithos-research-news`](#cli).
2. **Agent tool call**: `RUN: research-news <inquiry>` (virtual tool, no
   external binary required).
3. **Flowchart node**: the [`researchnews` node](#flowchart-node).

## Pipeline

Given an inquiry the tool performs the following steps:

1. **Term extraction** — a small language model turns the inquiry into a short
   list of focused technical search terms (e.g. *machine learning*,
   *transformer*, *cache quantization*).
2. **Scrape recent articles** — a scraper gathers candidate article URLs from
   the configured RSS/Atom feeds (preferred, because entries are dated) and,
   as a fallback, from a search of the whitelisted domains. Articles newer than
   `recency_days` (default 14) are downloaded, their main text extracted, and
   stored in the knowledge base.
3. **Summarise** — a subagent reads each article and produces a concise
   summary. The summary is stored in the knowledge base with a reference back
   to the source article.
4. **Judge relevance** — the subagent decides whether the article is relevant
   to the original inquiry (`RELEVANT` / `NOT RELEVANT` + reason).
5. **Repeat** — steps 3–4 run for every downloaded article.
6. **Collect** — articles judged relevant are listed with their summaries,
   written to a Markdown document under `output_dir`, and returned as the tool
   output.

```
+------------------+     +------------------+     +------------------+
|  Term extractor  | --> |  NewsScraper     | --> |  Assessor        |
|  (small model)   |     |  (feeds + search |     |  (subagent:      |
|                  |     |   + recency)     |     |   summary +      |
|                  |     |                  |     |   relevance)     |
+------------------+     +--------+---------+     +---------+--------+
                                  |                         |
                                  v                         v
                          +------------------+     +------------------+
                          |  Knowledge base  |     |  NewsReport      |
                          |  (MemoryStore:   |     |  (markdown doc + |
                          |   articles +     |     |   relevant list) |
                          |   summaries)     |     |                  |
                          +------------------+     +------------------+
```

## Why a "virtual" tool?

`research-news` is registered through the same `ToolRegistry` that discovers
CLI binaries, but with `tool_type="news_research"`. When an agent invokes it,
dispatch is intercepted inside `OllamaAgent._execute_tools` and routed to an
in-process `NewsResearcherToolExecutor` rather than a subprocess. This matches
the pattern used by the `flowchart` and `web-research` virtual tools.

## Architecture

### Components (`src/pithos/tools/news_researcher/`)

| File | Responsibility |
| --- | --- |
| `models.py` | `NewsArticle`, `ArticleAssessment`, `NewsResearchConfig`, `NewsResearchRequest`, `NewsReport` data classes. |
| `dates.py` | `parse_feed_date` / `parse_iso_date` / `parse_html_date` and the `is_recent` recency check. All dates are normalised to UTC. |
| `feeds.py` | `parse_feed` + `fetch_feed` — RSS 2.0 / Atom parsing with stdlib `xml.etree` (no third-party feed parser). |
| `terms.py` | `extract_terms` — small-model technical term extraction with a naive keyword fallback. |
| `scraper.py` | `NewsScraper` — feed + search discovery, recency filtering, hash dedup, article download, and knowledge-base storage. |
| `assessor.py` | `summarize_article`, `judge_relevance`, `assess_articles` — per-article subagent summary + relevance judgement and summary storage. |
| `researcher.py` | `NewsResearcher` facade + `NewsResearcherToolExecutor` (the agent-side adapter). |
| `cli.py` | `pithos-research-news` entry point. |

The HTTP fetcher (`Fetcher`), main-text extractor (`extract_main_text`), and
search backends (`DuckDuckGoSearch`, native per-domain APIs) are **reused from
the `web_researcher` package** so whitelist, `robots.txt`, rate-limiting, and
byte-cap policy are identical.

## Configuration

Two YAML files ship by default:

- **`configs/tools/news_research_config.yaml`** — runtime knobs:

  ```yaml
  domains:                       # whitelist; only these hosts are fetched
    - arstechnica.com
    - theverge.com
    - techcrunch.com
    - news.mit.edu
    - huggingface.co
  feeds:                         # RSS/Atom feeds (dated article links)
    - https://feeds.arstechnica.com/arstechnica/index
    - https://www.theverge.com/rss/index.xml
    - https://techcrunch.com/feed/
    - https://news.mit.edu/rss/feed
    - https://huggingface.co/blog/feed.xml
  recency_days: 14               # only keep articles newer than this
  skip_undated: true             # drop articles with no parseable date
  max_articles: 15               # hard cap on articles processed per run
  max_articles_per_source: 5     # per-host cap
  search_fallback: true          # search whitelisted domains when feeds are thin
  term_model: null               # small model for search terms (null = subagent model)
  subagent_config_name: news_researcher
  article_category: news_articles      # KB category for article bodies
  summary_category: news_summaries     # KB category for summaries
  output_dir: ./data/research/news     # where the collected doc is written
  ```

- **`configs/agents/news_researcher.yaml`** — the subagent persona used for
  summarisation and relevance judgement. It keeps `tools.enabled: false`
  (the pipeline drives it directly).

The tool is registered through `configs/tools/tool_config.yaml`:

```yaml
include:
  - research-news            # virtual tool

news_research:
  enabled: true              # gate for ToolRegistry discovery

descriptions:
  research-news: "Collect and summarise recent news articles from whitelisted domains. Usage: research-news <inquiry text>"
```

## Knowledge base

Downloaded articles and their summaries are stored in the persistent
`MemoryStore` (ChromaDB), not a throwaway per-run collection:

- **Article bodies** → the `article_category` category (`news_articles` by
  default), with `{url, title, published, source_host, inquiry, terms}`
  metadata.
- **Summaries** → the `summary_category` category (`news_summaries` by
  default), with a reference (`article_entry_id`, `url`) back to the source
  article.

This lets later sessions retrieve previously collected news via the normal
`memory:search` tool.

## Recency

Article age is determined from, in order of preference:

1. The RSS/Atom entry's `pubDate` / `published` / `updated` element.
2. HTML page metadata for search-discovered articles:
   `<meta property="article:published_time">`, JSON-LD `datePublished`, or the
   first `<time datetime=...>` element.

Articles older than `recency_days` are skipped. Articles with **no** parseable
date are skipped when `skip_undated` is `true` (the default) and kept
otherwise.

## Safety guardrails

All fetching goes through the shared `web_researcher.Fetcher`, so the same
protections apply: HTTPS-only, whitelist enforcement on every redirect hop,
`robots.txt` respected for article pages, per-domain rate limiting, and a byte
cap. Feed documents themselves are fetched with `robots.txt` bypassed (like the
search-engine API calls) but the article pages they link to are always fetched
through the normal whitelist + robots path.

## CLI

```bash
pithos-research-news "recent advances in cache quantization"

pithos-research-news "new transformer architectures" \
    --recency-days 7 --domains arxiv.org

pithos-research-news "open-source LLM releases" \
    --feed https://huggingface.co/blog/feed.xml

pithos-research-news "diffusion model efficiency" --json --quiet
```

Flags:

| Flag | Purpose |
| --- | --- |
| `--domains <host>` | Repeatable. Overrides `domains` in the config. |
| `--feed <url>` | Repeatable. Overrides `feeds` in the config. |
| `--recency-days <n>` | Overrides `recency_days` for this run. |
| `--json` | Emit `{terms, relevant, stats, errors, ...}` instead of markdown. |
| `--quiet` | Suppress progress logs. |
| `--config-dir <dir>` | Use a non-default `configs/` directory. |

If the optional `web` extra is not installed the CLI exits with an actionable
error message; no crash.

## Flowchart node

A `researchnews` flow node embeds the tool inside any pithos flowchart:

```yaml
nodes:
  - id: news
    type: researchnews
    inquiry: "{current_input}"     # supports {state} formatting
    save_to: news_report
    error_handling: continue       # or "stop"
    # Optional per-call overrides:
    domains: [arxiv.org]
    feeds: [https://huggingface.co/blog/feed.xml]
    recency_days: 7
```

After execution:

- `context["news_report"]` →
  `{inquiry, terms, markdown, relevant, document_path, stats, errors}`.
- `current_input` is overwritten with the rendered markdown report so the next
  node can consume it directly.

The node looks for a pre-built `NewsResearcher` under
`context["news_researcher"]`. If only `context["config_manager"]` is available,
a `NewsResearcher` is constructed lazily.

## Agent-side usage

```python
from pithos import OllamaAgent, ConfigManager

cm = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")
agent.enable_tools(cm)

response = agent.send(
    "Use the research-news tool to find recent articles about cache "
    "quantization in transformers."
)
print(response)
```

The agent emits a tool call such as `RUN: research-news cache quantization in
transformers`. Dispatch is handled in-process and the returned markdown report
is injected as a system message before generation resumes.

## Optional dependencies

The news research tool shares the `web` extra with `web-research`:

```bash
pip install -e ".[web]"
```

This pulls in `requests`, `beautifulsoup4`, and `trafilatura`. ChromaDB is
already a hard dependency of pithos and backs the knowledge base. If any of the
three optional packages is missing, `NEWS_RESEARCH_AVAILABLE` is `False` and the
tool is omitted from the registry; every entry point reports the missing extra
with a clear message rather than failing at import time.
