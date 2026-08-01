# Craft Analysis Tool

The `craft-notes` virtual tool analyzes a story's creative-writing craft and
produces **prescriptive** how-to notes — reusable writing advice, not plot
summaries — grounded in short evidence quotes from the source text. It is
exposed three ways:

1. **CLI**: [`pithos-craft-notes`](#cli).
2. **Agent tool call**: `[RUN]craft-notes <file path or raw text>[/RUN]`
   (virtual tool, no external binary required).
3. **Flowchart node**: the [`craftnotes` node](#flowchart-node).

## Pipeline

Given a story the tool performs the following steps:

1. **Ingest** — the source is resolved from raw text, a single file, or a
   directory/glob collection of text files into one normalized string plus a
   title.
2. **Chunk** — long text is split into overlapping chunks (`chunk_char_cap`,
   `chunk_overlap`) so the subagent only ever sees a bounded amount of context
   per call.
3. **Analyze per dimension** — for each configured craft dimension
   (characterization, scene construction, themes, prose style and voice,
   dialogue, plot structure and pacing), a subagent is prompted with a
   dimension-specific system prompt and asked to produce prescriptive
   `NOTE:`/`EVIDENCE:` blocks for every chunk.
4. **Dedup and cap** — notes are deduplicated (by normalised content hash) and
   capped to `max_notes_per_dimension`.
5. **Store** — the source text and every note are persisted to the knowledge
   base.
6. **Collect** — notes are grouped by dimension, written to a Markdown
   document under `output_dir`, and returned as the tool output.

```
+------------------+     +------------------+     +------------------+
|  Ingest          | --> |  Chunk           | --> |  Per-dimension   |
|  (text/file/dir) |     |  (char cap +     |     |  analysis        |
|                  |     |   overlap)       |     |  (subagent:      |
|                  |     |                  |     |   NOTE/EVIDENCE) |
+------------------+     +------------------+     +---------+--------+
                                                             |
                                                             v
                                                   +------------------+
                                                   |  Dedup + cap     |
                                                   |  per dimension   |
                                                   +---------+--------+
                                                             |
                                  +--------------------------+-------------+
                                  v                                        v
                          +------------------+                  +------------------+
                          |  Knowledge base  |                  |  CraftReport     |
                          |  (MemoryStore:   |                  |  (markdown doc + |
                          |   notes + source)|                  |   notes by dim)  |
                          +------------------+                  +------------------+
```

## Why a "virtual" tool?

`craft-notes` is registered through the same `ToolRegistry` that discovers CLI
binaries, but with `tool_type="craft_analysis"`. When an agent invokes it,
dispatch is intercepted inside `OllamaAgent._execute_tools` and routed to an
in-process `CraftAnalyzerToolExecutor` rather than a subprocess. This matches
the pattern used by the `flowchart`, `web-research`, and `research-news`
virtual tools.

## Architecture

### Components (`src/pithos/tools/craft_analyzer/`)

| File | Responsibility |
| --- | --- |
| `models.py` | `CraftNote`, `CraftAnalysisConfig`, `CraftAnalysisRequest`, `CraftReport` data classes, and the canonical `DIMENSIONS` tuple. |
| `ingest.py` | `resolve_source` — raw text / single file / directory-glob resolution into `(text, title)`; `chunk_text` — overlapping chunk splitting. |
| `dimensions.py` | `dimension_system_prompt`, `build_user_prompt`, `parse_notes`, `dedup_notes` — per-dimension prompt templates and note parsing. |
| `analyzer.py` | `CraftAnalyzer` facade + `CraftAnalyzerToolExecutor` (the agent-side adapter). |
| `cli.py` | `pithos-craft-notes` entry point. |

Directory ingestion reuses the same glob include/exclude enumeration logic as
`pithos.coding_nodes.ListFilesNode`.

## Configuration

Two YAML files ship by default:

- **`configs/tools/craft_analysis_config.yaml`** — runtime knobs:

  ```yaml
  dimensions:                      # craft dimensions analyzed
    - characterization
    - scene_construction
    - themes
    - prose_style_and_voice
    - dialogue
    - plot_structure_and_pacing
  include:                         # glob patterns for directory sources
    - "**/*.txt"
    - "**/*.md"
  exclude: []
  max_files: 200                   # safety cap on files read from a directory
  chunk_char_cap: 6000              # max characters per chunk sent to the model
  chunk_overlap: 300                # chars of overlap between consecutive chunks
  max_chunks: 20                    # hard cap on chunks analyzed per run
  subagent_config_name: craft_analyst
  max_notes_per_dimension: 8
  dedup_notes: true
  note_category: craft_notes        # KB category for prescriptive notes
  source_category: craft_sources    # KB category for analyzed source text
  output_dir: ./data/research/craft # where the collected doc is written
  write_document: true
  ```

- **`configs/agents/craft_analyst.yaml`** — the subagent persona used for
  per-dimension analysis. It keeps `tools.enabled: false` (the pipeline drives
  it directly).

The tool is registered through `configs/tools/tool_config.yaml`:

```yaml
include:
  - craft-notes               # virtual tool

craft_analysis:
  enabled: true                # gate for ToolRegistry discovery

descriptions:
  craft-notes: "Analyze a story's creative-writing craft and produce prescriptive how-to notes. Usage: craft-notes <file path or raw text>"
```

## Knowledge base

Analyzed source text and its notes are stored in the persistent `MemoryStore`
(ChromaDB), not a throwaway per-run collection:

- **Source text** → the `source_category` category (`craft_sources` by
  default), with `{title, chunks, kind}` metadata.
- **Notes** → the `note_category` category (`craft_notes` by default), with
  `{dimension, source_title, evidence, source_entry_id, kind}` metadata. The
  `dimension` field can be used with `MemoryStore.retrieve(..., where=...)` to
  filter recall to a single craft dimension (e.g. only `dialogue` notes) when
  drafting a new story.

This lets later sessions retrieve previously collected craft notes via the
normal `memory:search` tool, or have an agent auto-recall relevant notes
before drafting similar material.

## Note orientation

Notes are **prescriptive**, not descriptive: the subagent is instructed to
produce actionable writing techniques a writer could reuse (e.g. "Reveal a
character's core flaw through a small action before naming it directly"),
each backed by a short evidence quote/paraphrase from the passage — not a
summary of what happens in the story.

## CLI

```bash
pithos-craft-notes story.txt

pithos-craft-notes --roots ./data/research/stories --dimension dialogue

pithos-craft-notes --text "Once upon a time, a knight faced a dragon."

pithos-craft-notes story.txt --json --quiet
```

Flags:

| Flag | Purpose |
| --- | --- |
| `source` (positional) | Path to a single story file, or a directory to scan for text files. |
| `--text <text>` | Analyze raw text instead of a file/directory. |
| `--roots <dir>` | Repeatable. Analyze a directory/glob collection of files. |
| `--title <title>` | Override the report/notes title. Defaults to the file/directory name. |
| `--dimension <name>` | Repeatable. Limit analysis to specific dimensions. Defaults to config. |
| `--json` | Emit `{title, notes, stats, errors, ...}` instead of markdown. |
| `--quiet` | Suppress progress logs. |
| `--config-dir <dir>` | Use a non-default `configs/` directory. |

Exactly one of `source`, `--text`, or `--roots` must be given.

## Flowchart node

A `craftnotes` flow node embeds the tool inside any pithos flowchart:

```yaml
nodes:
  - id: craft
    type: craftnotes
    source: "{current_input}"      # supports {state} formatting; raw text or a file path
    save_to: craft_report
    title: "My Story"
    error_handling: continue       # or "stop"
    # Optional per-call override:
    dimensions: [dialogue, themes]
```

After execution:

- `context["craft_report"]` →
  `{title, markdown, notes, document_path, stats, errors}`.
- `current_input` is overwritten with the rendered markdown report so the next
  node can consume it directly.

The node looks for a pre-built `CraftAnalyzer` under
`context["craft_analyzer"]`. If only `context["config_manager"]` is available,
a `CraftAnalyzer` is constructed lazily. If `source` resolves to an existing
file path it is analyzed as a file; otherwise it is treated as raw text.

## Agent-side usage

```python
from pithos import OllamaAgent, ConfigManager

cm = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")
agent.enable_tools(cm)

response = agent.send(
    "Use the craft-notes tool to analyze story.txt and give me notes on "
    "dialogue and characterization."
)
print(response)
```

The agent emits a tool call such as `[RUN]craft-notes story.txt[/RUN]`.
Dispatch is handled in-process and the returned markdown report is injected as
a system message before generation resumes.

## Optional dependencies

Unlike `web-research` and `research-news`, the craft-analysis tool has no
optional third-party dependencies beyond what pithos already requires (an
LLM backend and, for knowledge-base storage, ChromaDB). `CRAFT_ANALYSIS_AVAILABLE`
is always `True`; it exists purely for interface parity with the other virtual
tools. If ChromaDB is unavailable, note/source storage is skipped gracefully
and analysis still runs and returns a report.
