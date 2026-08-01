# Craft Writing Tool

The `craft-write` virtual tool writes a short story from freeform user
direction, guided by prescriptive craft notes previously produced by the
[`craft-notes`](CRAFT_ANALYSIS.md) tool. It is exposed three ways:

1. **CLI**: [`pithos-craft-write`](#cli).
2. **Agent tool call**: `[RUN]craft-write <direction text>[/RUN]`
   (virtual tool, no external binary required).
3. **Flowchart node**: the [`craftwrite` node](#flowchart-node).

## Pipeline

Given a direction (e.g. "a heist gone wrong, melancholy tone"), the tool runs
a 3-stage subagent-driven pipeline:

1. **Retrieve notes** — for each configured craft dimension, the direction is
   used as a semantic query against the `craft_notes` knowledge base
   (optionally restricted to a specific analyzed story via `source_title`).
2. **Outline** — a planner subagent proposes a title, premise, and a
   section-by-section breakdown, guided by characterization/themes/plot
   notes.
3. **Draft** — a writer subagent drafts each section in turn, keeping the
   running story-so-far (capped by `story_context_char_cap`) in context for
   continuity and applying scene-construction/dialogue/prose-style notes.
4. **Revise** *(optional)* — one or more whole-draft revision passes apply
   prose-style/voice notes and tidy up consistency, without changing plot or
   structure.
5. **Store & collect** — the finished story is persisted to the knowledge
   base and written to a Markdown document under `output_dir`.

```
+------------------+     +------------------+     +------------------+
|  Retrieve notes  | --> |  Outline         | --> |  Per-section     |
|  (per dimension,  |     |  (subagent:      |     |  draft           |
|   direction as    |     |   TITLE/PREMISE/ |     |  (subagent, keeps|
|   query)          |     |   SECTION)       |     |   story-so-far)  |
+------------------+     +------------------+     +---------+--------+
                                                             |
                                                             v
                                                   +------------------+
                                                   |  Revision        |
                                                   |  (optional pass) |
                                                   +---------+--------+
                                                             |
                                  +--------------------------+-------------+
                                  v                                        v
                          +------------------+                  +------------------+
                          |  Knowledge base  |                  |  CraftStory      |
                          |  (MemoryStore:   |                  |  (markdown doc + |
                          |   craft_stories) |                  |   sections)      |
                          +------------------+                  +------------------+
```

## Why a "virtual" tool?

`craft-write` is registered through the same `ToolRegistry` that discovers CLI
binaries, but with `tool_type="craft_writing"`. When an agent invokes it,
dispatch is intercepted inside `OllamaAgent._execute_tools` and routed to an
in-process `CraftWriterToolExecutor` rather than a subprocess. This matches
the pattern used by the `craft-notes`, `flowchart`, `web-research`, and
`research-news` virtual tools.

## Architecture

### Components (`src/pithos/tools/craft_writer/`)

| File | Responsibility |
| --- | --- |
| `models.py` | `CraftWriteConfig`, `CraftWriteRequest`, `StorySection`, `StoryOutline`, `CraftStory` data classes. |
| `notes.py` | `retrieve_craft_notes` — per-dimension semantic retrieval from the `craft_notes` knowledge base; `format_notes_for_prompt` — renders retrieved notes for prompt injection. |
| `prompts.py` | Per-stage system/user prompt builders (`outline_system_prompt`, `section_system_prompt`, `revision_system_prompt`, and their `build_*_user_prompt` counterparts) and `parse_outline` — parses the planner's `TITLE:`/`PREMISE:`/`SECTION:` reply. |
| `writer.py` | `CraftWriter` facade (the 3-stage pipeline) + `CraftWriterToolExecutor` (the agent-side adapter). |
| `cli.py` | `pithos-craft-write` entry point. |

## Configuration

Two YAML files ship by default:

- **`configs/tools/craft_writing_config.yaml`** — runtime knobs:

  ```yaml
  dimensions:                          # craft dimensions retrieved
    - characterization
    - scene_construction
    - themes
    - prose_style_and_voice
    - dialogue
    - plot_structure_and_pacing
  notes_per_dimension: 5                # notes retrieved per dimension
  min_relevance: null                   # null = MemoryStore default threshold
  note_category: craft_notes            # KB category to read notes from
  story_category: craft_stories         # KB category to store finished stories
  subagent_config_name: craft_writer
  target_word_count: 2000
  num_sections: null                    # null = derived from target_word_count
  revision_passes: 1                    # 0 skips the revision stage
  story_context_char_cap: 8000          # story-so-far context cap per section
  output_dir: ./data/research/stories   # where the collected doc is written
  write_document: true
  store_story: true
  ```

- **`configs/agents/craft_writer.yaml`** — the subagent persona used across
  all three pipeline stages. It keeps `tools.enabled: false` (the pipeline
  drives it directly with per-stage system prompts).

The tool is registered through `configs/tools/tool_config.yaml`:

```yaml
include:
  - craft-write                # virtual tool

craft_writing:
  enabled: true                 # gate for ToolRegistry discovery

descriptions:
  craft-write: "Write a short story guided by previously analyzed craft notes. Usage: craft-write <direction text>"
```

## Knowledge base

- **Notes read** — from the `note_category` category (`craft_notes` by
  default, the same category `craft-notes` writes to), filtered per
  dimension via `MemoryStore.retrieve(..., where={"dimension": ...})`, and
  additionally by `source_title` (via a `$and` filter) when the request
  restricts retrieval to a specific analyzed story.
- **Stories written** — to the `story_category` category (`craft_stories` by
  default), with `{title, direction, genre, tone, source_title, kind}`
  metadata, so generated stories can themselves be recalled or analyzed
  later.

If the knowledge base is unavailable (ChromaDB not installed, or no notes
have been stored yet), retrieval degrades gracefully to an empty note set per
dimension and the pipeline still runs — sections are drafted with general
craft best practices instead of story-specific guidance.

## CLI

```bash
# Write a story from a direction
pithos-craft-write "a heist gone wrong, melancholy tone"

# Guide genre/tone and give an explicit title
pithos-craft-write --title "The Last Job" --genre thriller --tone tense "a heist gone wrong"

# Restrict retrieved notes to one previously analyzed story
pithos-craft-write --source-title "Some Analyzed Story" --json "a quiet reunion"

# Skip the revision pass and target a shorter story
pithos-craft-write --no-revise --words 800 "a quiet reunion"
```

Flags:

| Flag | Purpose |
| --- | --- |
| `direction` (positional) | Freeform direction describing what to write. |
| `--title <title>` | Story title (otherwise proposed by the outline stage). |
| `--genre <genre>` | Genre guidance for the outline stage. |
| `--tone <tone>` | Tone guidance for the outline stage. |
| `--words <n>` | Approximate target word count for the story. |
| `--sections <n>` | Number of sections to plan (defaults to derived from `--words` or config). |
| `--source-title <title>` | Restrict retrieved craft notes to those derived from a specific analyzed story. |
| `--dimension <name>` | Repeatable. Limit note retrieval to specific dimensions. Defaults to config. |
| `--no-revise` | Skip the final whole-draft revision pass. |
| `--json` | Emit `{title, premise, sections, full_text, stats, errors, ...}` instead of markdown. |
| `--quiet` | Suppress progress logs. |
| `--config-dir <dir>` | Use a non-default `configs/` directory. |

## Flowchart node

A `craftwrite` flow node embeds the tool inside any pithos flowchart:

```yaml
nodes:
  - id: write
    type: craftwrite
    direction: "{current_input}"   # supports {state} formatting
    save_to: craft_story
    title: "The Last Job"
    genre: thriller
    tone: tense
    error_handling: continue       # or "stop"
    # Optional per-call overrides:
    source_title: "Some Analyzed Story"
    dimensions: [dialogue, scene_construction]
```

After execution:

- `context["craft_story"]` →
  `{title, premise, markdown, full_text, sections, document_path, stats, errors}`.
- `current_input` is overwritten with the story's `full_text` (falling back to
  the rendered markdown) so the next node can consume it directly.

The node looks for a pre-built `CraftWriter` under `context["craft_writer"]`.
If only `context["config_manager"]` is available, a `CraftWriter` is
constructed lazily.

## Agent-side usage

```python
from pithos import OllamaAgent, ConfigManager

cm = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")
agent.enable_tools(cm)

response = agent.send(
    "Use the craft-write tool to write a short story about a heist gone "
    "wrong, in a melancholy tone."
)
print(response)
```

The agent emits a tool call such as
`[RUN]craft-write a heist gone wrong, melancholy tone[/RUN]`. Dispatch is
handled in-process and the returned markdown story is injected as a system
message before generation resumes.

## Optional dependencies

Like `craft-notes`, this tool has no optional third-party dependencies beyond
what pithos already requires (an LLM backend and, for note retrieval/story
storage, ChromaDB). `CRAFT_WRITING_AVAILABLE` is always `True`; it exists
purely for interface parity with the other virtual tools. If ChromaDB is
unavailable, note retrieval and story storage are skipped gracefully and the
pipeline still runs, producing a story guided by general craft practices
rather than previously analyzed notes.
