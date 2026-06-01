# Web Research Tool

The `web-research` virtual tool is a subagent-driven web crawler that
collects, deduplicates, and summarises information from a configurable
whitelist of web domains. It is exposed three ways:

1. **CLI**: [`pithos-research`](#cli).
2. **Agent tool call**: `RUN: web-research <inquiry>` (virtual tool, no
   external binary required).
3. **Flowchart node**: the [`webresearch` node](#flowchart-node).

## Why a "virtual" tool?

`web-research` is registered through the same `ToolRegistry` that discovers
CLI binaries, but `tool_type="web_research"` instead of `"cli"`. When an
agent invokes it, dispatch is intercepted inside
`OllamaAgent._execute_tools` and routed to an in-process
`WebResearcherToolExecutor` rather than a subprocess. This matches the
pattern already used for the `flowchart` virtual tool.

## Architecture

```
+------------------+      +------------------+      +------------------+
|  ResearchLoop    | ---> |  Fetcher         | ---> |  Extractor       |
|  (subagent)      |      |  (whitelist,     |      |  (trafilatura +  |
|                  |      |   robots, RPS,   |      |   BS4 fallback)  |
|                  |      |   byte cap)      |      |                  |
+--------+---------+      +------------------+      +---------+--------+
         |                                                    |
         | FETCH / SEARCH / NOTE / STOP                       | chunks
         v                                                    v
+------------------+      +------------------+      +------------------+
| DuckDuckGoSearch | <--- |  ExcerptStore    | <--- |  chunk_text      |
| (site: filter)   |      |  (ChromaDB +     |      |  (sentence-aware |
|                  |      |   hash + cosine  |      |   ~600/80)       |
|                  |      |   dedup)         |      |                  |
+------------------+      +------------------+      +------------------+
                                  |
                                  v
                          +------------------+
                          |  Summarizer      |
                          |  ([N] citations) |
                          +------------------+
                                  |
                                  v
                          +------------------+
                          |  ResearchReport  |
                          |  (markdown +     |
                          |   sources)       |
                          +------------------+
```

### Components (`src/pithos/tools/web_researcher/`)

| File | Responsibility |
| --- | --- |
| `models.py` | `Excerpt`, `ResearchReport`, `WebResearchConfig`, `WebResearchRequest` data classes. |
| `fetcher.py` | Whitelisted HTTP fetcher with robots.txt, per-domain rate limiting, manual redirect handling, byte/content-type caps. |
| `search.py` | `DuckDuckGoSearch` wrapper — issues `site:<domain> <query>` against `html.duckduckgo.com`, filters results to the target domain. |
| `extractor.py` | `extract_main_text` (trafilatura → BS4 fallback) + sentence-aware `chunk_text` + `filter_outlinks`. |
| `store.py` | `ExcerptStore` over ChromaDB — hash dedup (SHA-1 of normalised text) and semantic dedup (cosine similarity above `dedup_similarity`). |
| `parser.py` | `extract_actions` — parses regex-based `FETCH:` / `SEARCH:` / `NOTE:` / `STOP` lines from the subagent reply, with a JSON fallback. |
| `agent_loop.py` | `ResearchLoop.run` — primes the subagent, executes its actions, enforces page/iteration budgets. |
| `summarizer.py` | `synthesize` — bundles deduped excerpts with per-URL numbering and asks an agent to produce a summary with `[N]` inline citations. |
| `researcher.py` | `WebResearcher` facade (builds per-run fetcher/search/store, cleans up its collection) and `WebResearcherToolExecutor` (the agent-side adapter). |
| `cli.py` | `pithos-research` entry point. |

### Subagent grammar

The research subagent communicates via one action per line. The parser
accepts case-insensitive verbs and several quoting/bracket variants:

```
FETCH: https://en.wikipedia.org/wiki/HTTP/3
SEARCH en.wikipedia.org "HTTP/3 multiplexing"
NOTE: switching to MDN for client-side details
STOP
```

If the reply contains no parseable actions, a JSON fallback is tried:

```json
{ "actions": [ { "op": "fetch", "url": "https://..." }, { "op": "stop" } ] }
```

## Configuration

Two YAML files ship by default:

- **`configs/tools/web_research_config.yaml`** — runtime knobs:

  ```yaml
  domains:
    - en.wikipedia.org
    - developer.mozilla.org
    - docs.python.org
    - arxiv.org
    - github.com
  max_pages: 20            # hard cap on pages fetched per inquiry
  max_iterations: 8        # hard cap on subagent rounds
  dedup_similarity: 0.92   # cosine threshold for semantic dedup
  persist_directory: ./data/research
  subagent_config_name: web_researcher
  ```

- **`configs/agents/web_researcher.yaml`** — the subagent persona used by
  `ResearchLoop`. It must keep `tools.enabled: false` (the loop drives it
  directly via the grammar above).

The tool is registered through `configs/tools/tool_config.yaml`:

```yaml
include:
  - web-research            # virtual tool

web_research:
  enabled: true             # gate for ToolRegistry discovery

descriptions:
  web-research: "Subagent-driven web crawler restricted to a domain whitelist."
```

## Safety guardrails

Every fetch is checked against the whitelist on *every* redirect hop, with
the following additional protections:

- **HTTPS only** — `http://` URLs are rejected before any network call.
- **robots.txt** respected by default (`respect_robots=True`), with
  fail-open semantics if the robots file itself errors.
- **Per-domain rate limiting** (`per_domain_rps`, default `1.0`).
- **Manual redirects** (`allow_redirects=False`) capped at six hops; each
  hop must re-pass the whitelist + robots checks.
- **Content-Type filter** — only `text/html`/`*/xml` responses are kept.
- **Byte cap** — responses larger than `max_bytes` (default 2 MB) are
  *rejected*, not truncated, to avoid feeding partial documents to the
  extractor.

## ExcerptStore deduplication

`ExcerptStore` uses a two-tier strategy:

1. **Hash dedup** — SHA-1 of the lowercased, whitespace-collapsed text is
   tracked in memory for the lifetime of the run. Exact duplicates are
   skipped before any vector search.
2. **Semantic dedup** — `collection.query(...)` checks the nearest
   neighbour. If the cosine *distance* is below
   `1 - dedup_similarity`, the new excerpt is dropped.

Each excerpt carries `{url, title, relevance, ts, m_<custom>}` metadata so
the summariser can emit cited sources in insertion order.

## Citation verification (editor subagent)

After the summarizer drafts the report, an **editor subagent** audits every
`[N]` citation. Two checks are performed:

1. **Deterministic source check** — every cited URL must be reachable.
   Sources already present in the `ExcerptStore` are trusted (they were
   fetched successfully during the research loop). Any cited URL not in
   the store is probed via `Fetcher.verify_url` (HEAD with GET fallback),
   honouring the same whitelist + `robots.txt` rules as the crawler.
2. **LLM claim check** — for each `[N]` marker, the surrounding sentence
   is sent to the editor agent together with the stored excerpts from
   that source. The agent replies with one of `SUPPORTED`, `PARTIAL`,
   or `UNSUPPORTED` plus a short reason; replies that cannot be parsed
   default to `UNSUPPORTED`.

If any claim comes back `UNSUPPORTED` or its source is dead, the editor is
asked to rewrite the summary, deleting or softening those claims and
stripping only their `[N]` markers. Supported citations are preserved
exactly. The pre-edit text is kept on `ResearchReport.original_summary`
for audit, and `report.stats` gains:

| Key | Meaning |
|---|---|
| `citations_total` | Total `[N]` markers checked |
| `citations_unsupported` | Markers verdicted `UNSUPPORTED` |
| `dead_sources` | Cited URLs that failed reachability |
| `editor_rewrote` | `true` iff the summary was rewritten |
| `editor_duration_seconds` | Wall-clock time for the editor stage |

Editor failures degrade gracefully: an error is appended to
`report.errors`, the original summary is retained, and the rest of the
report is unchanged.

Configuration knobs (`configs/tools/web_research_config.yaml`):

```yaml
verify_citations: true            # set false to skip the editor stage
editor_config_name: "editor"      # configs/agents/editor.yaml
editor_model: null                # null = use the value from the agent config
```

The editor agent is defined in `configs/agents/editor.yaml`. It uses the
same backend as `web_researcher` by default and does not invoke any tools.

## CLI

```bash
pithos-research "What are the major changes in HTTP/3?"

pithos-research "Difference between SSE and WebSockets" \
    --domains developer.mozilla.org --domains en.wikipedia.org

pithos-research "Trafilatura extraction pipeline" \
    --seed-url https://github.com/adbar/trafilatura

pithos-research "Python GIL removal status" --json --quiet
```

Flags:

| Flag | Purpose |
| --- | --- |
| `--domains <host>` | Repeatable. Overrides `domains` in the config. |
| `--seed-url <url>` | Repeatable. URLs to enqueue before the subagent's first round. |
| `--json` | Emit `{summary, sources, stats, errors}` instead of markdown. |
| `--quiet` | Suppress progress logs. |
| `--config-dir <dir>` | Use a non-default `configs/` directory. |

If the optional `web` extra is not installed the CLI exits with an
actionable error message; no crash.

## Flowchart node

A `webresearch` flow node lets you embed the tool inside any pithos
flowchart. Minimal YAML:

```yaml
nodes:
  - id: research
    type: webresearch
    inquiry: "{current_input}"     # supports {state} formatting
    save_to: research_report
    error_handling: continue       # or "stop"
    # Optional per-call overrides:
    domains: [en.wikipedia.org, developer.mozilla.org]
    seed_urls: []
```

After execution:

- `context["research_report"]` →
  `{inquiry, summary, markdown, sources, excerpt_count, stats, errors}`.
- `current_input` is overwritten with the rendered markdown report so the
  next node can consume it directly.

The node looks for a pre-built `WebResearcher` under
`context["web_researcher"]`. If only `context["config_manager"]` is
available, a `WebResearcher` is constructed lazily.

## Agent-side usage

```python
from pithos import OllamaAgent, ConfigManager

cm = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")
agent.enable_tools(cm)

response = agent.send(
    "Use the web-research tool to find recent benchmarks comparing "
    "HTTP/3 and HTTP/2 latency. Cite all sources."
)
print(response)
```

The agent will emit a tool call such as `RUN: web-research HTTP/3 vs
HTTP/2 latency benchmarks`. Dispatch is handled in-process and the
returned markdown report is injected as a system message before
generation resumes.

## Optional dependencies

The web research tool requires the `web` extra:

```bash
pip install -e ".[web]"
```

This pulls in `requests`, `beautifulsoup4`, and `trafilatura`. ChromaDB is
already a hard dependency of pithos and is used for the excerpt store.

If any of the three optional packages is missing,
`WEB_RESEARCH_AVAILABLE` is set to `False` in
`pithos.tools.web_researcher.__init__` and the tool is omitted from the
registry. Every entry point reports the missing extra with a clear,
actionable message rather than failing at import time.
