# Coding Harness (`code` tool)

The coding harness is a full agentic coding loop built on the pithos ecosystem.
It is exposed as the subagent-driven **`code`** virtual tool (and the
`pithos-code` CLI). A single coding agent plans, edits files, runs tests, and
self-corrects in a tight **query → act → observe** loop, confined to a
workspace root.

It replaces the older flowchart-based coding workflow
(`configs/flowcharts/general_coding.yaml`, still available as a legacy option)
with a simpler, more capable single-agent harness inspired by `mini-swe-agent`.

## At a glance

- **Single agent, one loop.** No planner/coder/reviewer/repairer roles — one
  agent iterates until the task is done.
- **Native structured tool-calling.** Each step the agent emits `[RUN]`,
  `[EDIT]`, and `[SUBMIT]` blocks, which the harness parses and executes through
  a workspace-jailed `ToolRegistry` (reusing `CLIToolProvider`, the
  `CommandSafetyChecker`, and root-jailed file-edit/git helpers).
- **Local-first, root-jailed.** File edits and git are confined to an
  `allowed_root`; shell commands run with the working directory pinned to that
  root and pass through the standard safety checks. No Docker.
- **Bounded.** Step, wall-clock, and consecutive-format-error limits guarantee
  termination; command output is truncated to keep the context small.
- **Observable.** Every run writes a JSON trajectory to `data/coding/`.

## Usage

### As a tool (from an agent or subagent)

```
[RUN]code fix the failing test in tests/test_foo.py[/RUN]
[RUN]code --root ./myrepo add a --verbose flag to the CLI[/RUN]
```

An optional `--root <path>` prefix pins the workspace root; otherwise the
configured `allowed_root` (or the current working directory) is used.

### From the CLI

```bash
pithos-code "fix the failing test in tests/test_foo.py"
pithos-code --root ./myrepo "add a --verbose flag to the CLI"
pithos-code --json --steps 20 "refactor utils.py into smaller modules"
```

## The action grammar

Each turn the agent responds with a short `THOUGHT` followed by one or more
action blocks:

**Run a command** (shell / CLI tool, executed in the workspace root):

```
[RUN]python -m pytest -q[/RUN]
```

**Edit a file** (path relative to the workspace root):

```
[EDIT path=src/foo.py mode=replace]
<the complete new file contents>
[/EDIT]
```

`mode` is one of:

| mode             | body meaning                                             |
| ---------------- | -------------------------------------------------------- |
| `replace`        | the complete new file contents                           |
| `create`         | contents for a new file (fails if it already exists)     |
| `insert_at_line` | text inserted before 1-based `line=N` (add the `line` attr) |
| `patch`          | a unified diff applied to the file                       |

A `.bak` snapshot is written before any in-place edit (configurable).

**Submit the final result** and end the run:

```
[SUBMIT]
<summary of what you did and how you verified it>
[/SUBMIT]
```

A reply containing no recognised block is a *format error*; too many
consecutive format errors ends the run.

## How the loop works

1. The harness builds a coding subagent from the `code_harness` agent config and
   installs the harness **system prompt** (the action grammar + rules). The
   subagent's own tool loop is **disabled** — the harness owns execution.
2. The first user message states the task, the workspace root, system info, an
   optional test command, the available commands, and a file listing.
3. Each step:
   - the agent replies;
   - the harness parses the reply into actions;
   - `run` and `edit` actions execute against the
     [`WorkspaceEnvironment`](../src/pithos/tools/coder/environment.py);
   - a truncated **observation** (return code + output) is fed back;
   - a `submit` action ends the run with a summary.
4. After the loop, the harness captures a `git diff` (if the root is a repo),
   optionally auto-commits, and writes the trajectory.

Limits that stop the loop: `step_limit`, `wall_time_limit_seconds`, and
`max_consecutive_format_errors`.

## Safety model

- **Commands** go through the `CommandSafetyChecker`: shell-injection operators
  and known-dangerous patterns are `BLOCK`ed outright; destructive flags are
  `REVIEW` (handled per `review_policy`). Only allow-listed CLI tools run, with
  the working directory pinned to the workspace root.
- **File edits** go through the root-jailed helpers in
  [`file_ops.py`](../src/pithos/tools/coder/file_ops.py): any path that escapes
  `allowed_root` raises `PermissionError`, binary files are refused, and `.bak`
  snapshots are written before in-place edits.
- **git** is restricted to a whitelist of subcommands; destructive ones
  (`reset`, `clean`, `push`, `rebase`, `merge`) require
  `allow_destructive_git: true`.

## Configuration

Settings live in `configs/tools/code_config.yaml`. Enable/disable the tool via
the `code:` block in `configs/tools/tool_config.yaml`.

| Key | Default | Meaning |
| --- | --- | --- |
| `subagent_config_name` | `code_harness` | Agent config used for the coding subagent |
| `subagent_model` | `null` | Override the subagent model (null = agent config value) |
| `subagent_timeout` | `300` | Per-generation wall-clock cap (seconds) |
| `step_limit` | `40` | Max agent steps (0 = unlimited) |
| `wall_time_limit_seconds` | `0` | Overall wall-clock cap (0 = no limit) |
| `max_consecutive_format_errors` | `3` | Stop after this many unparseable replies |
| `command_timeout` | `60` | Per-command subprocess timeout (seconds) |
| `review_policy` | `allow` | How to treat `REVIEW` safety verdicts: `allow` or `block` |
| `max_output_chars` / `output_head_chars` / `output_tail_chars` | `10000` / `5000` / `5000` | Observation truncation |
| `allowed_root` | `null` | Workspace root (null = per-request root, else cwd) |
| `backup_edits` | `true` | Write `.bak` before in-place edits |
| `include_file_listing` / `max_listed_files` | `true` / `200` | File listing in the first prompt |
| `allow_git` / `allow_destructive_git` / `auto_commit` | `true` / `false` / `false` | git behaviour |
| `output_dir` / `write_trajectory` | `./data/coding` / `true` | Trajectory output |

## Module layout

```
src/pithos/tools/coder/
  models.py        # CodeHarnessConfig, CodeTaskRequest, CodeTaskResult, ExitStatus, StepRecord
  prompts.py       # system / instance / observation / format-error templates
  actions.py       # parse [RUN]/[EDIT]/[SUBMIT] blocks into Action objects
  file_ops.py      # shared root-jailed edit helpers (also used by coding_nodes.EditFileNode)
  environment.py   # WorkspaceEnvironment: jailed command + file-edit execution
  harness.py       # CodeHarness: the query -> act -> observe loop
  coder.py         # Coder facade + CoderToolExecutor (the `code` virtual tool)
  cli.py           # pithos-code entry point
```

## Programmatic use

```python
from pithos.config_manager import ConfigManager
from pithos.tools.coder import Coder
from pithos.tools.coder.models import CodeTaskRequest

coder = Coder(ConfigManager())
result = coder.code(CodeTaskRequest(task="add type hints to utils.py", root="./myrepo"))
print(result.exit_status, result.summary)
print(result.diff)
```

`Coder` and `CodeHarness` accept an `agent_factory` callable, which makes the
harness easy to unit-test with a scripted stub agent (see
`tests/test_coder_harness.py`).

## Testing

```bash
pytest tests/test_coder_file_ops.py tests/test_coder_actions.py \
       tests/test_coder_environment.py tests/test_coder_harness.py \
       tests/test_coder_tool.py -q
```
