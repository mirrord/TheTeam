# Evaluating pithos agents and workflows

`pithos.eval` is the evaluation suite for pithos agents, teams, and
flowcharts.

## Quick start

```powershell
.\.venv\Scripts\Activate.ps1
pithos-eval list-suites
pithos-eval run --config configs/eval/example.yaml
pithos-eval report --run-dir results/2026-05-23-example
```

## Concepts

| Concept | Type | Responsibility |
|---|---|---|
| **Subject** | `pithos.eval.subjects.Subject` | A thing under test — `AgentSubject`, `FlowchartSubject`, `TeamSubject`. |
| **Task** | `pithos.eval.tasks.Task` | A dataset + grader bundle. Built-ins: `multiple_choice`, `free_form`, `tool_use`, `memory_recall`, `self_reflection`. |
| **Grader** | `pithos.eval.graders.Grader` | Scores a `TaskCase` output. Built-ins: `letter_match`, `exact_match`, `regex_match`, `llm_judge`, `composite`, `tool_trace`, `memory_recall`. |
| **Analyzer** | `pithos.eval.trace.analyzers.Analyzer` | Inspects an `EvalTrace` for trajectory issues (loops, redundant calls, tool hallucination, cost spikes, latency, stability). |
| **Runner** | `pithos.eval.runner.EvalRunner` | Iterates `rounds × subjects × tasks × cases`, retries on failure, writes resumable JSONL. |
| **Reporter** | `pithos.eval.reporter.Reporter` | Aggregates `CaseRecord`s into the C.L.A.S.S. report (`Correctness, Latency, Adaptability, Stability, Steerability`). |

## YAML schema

```yaml
name: example

subjects:
  planner:
    type: agent
    agent: planner            # configs/agents/planner.yaml
  reflector_flow:
    type: flowchart
    flowchart: simple_reflect
    agents: [planner, reflector]

tasks:
  linguistic:
    type: multiple_choice
    dataset: { type: multiple_choice, builtin: linguistic_basic }
    grader:  { type: letter_match }
  tool_basic:
    type: tool_use
    dataset: { type: tool_use, builtin: tool_use_basic }
    grader:  { type: tool_trace }
  memory_basic:
    type: memory_recall
    dataset: { type: memory_recall, builtin: memory_recall_basic }
    grader:  { type: memory_recall }

analyzers:
  - { type: loop_detector }
  - { type: redundancy }
  - { type: tool_hallucination }
  - { type: cost }
  - { type: latency }
  - { type: stability }

execution:
  rounds: 3
  num_retries: 1
  parallelism: 4

output:
  base_dir: ./results
```

## Built-in datasets

Located under `src/pithos/eval/datasets/builtins/` and accessible via
`{"builtin": "<name>"}`:

* `linguistic_basic` — 30 multi-choice linguistic reasoning items.
* `tool_use_basic` — 8 cases asserting which tools a subject should call.
* `memory_recall_basic` — 6 two-turn recall scenarios using
  `setup_prompts` to seed information in turn 1.
* `self_reflection_basic` — 6 planted-error prompts requiring
  self-correction.

## CLI

| Command | Purpose |
|---|---|
| `pithos-eval list-suites` | Show registered task types. |
| `pithos-eval list-configs [--dir configs/eval]` | List YAML configs in a directory. |
| `pithos-eval run --config <yaml>` | Execute a config. Supports `--rounds`, `--max-cases`, `--no-resume`, `--dry-run`, `--output-dir`, `--price-map`. |
| `pithos-eval report --run-dir <dir>` | Re-aggregate an existing run into `stats/report.json` + `stats/class_report.csv`. |
| `pithos-eval analyze --run-dir <dir>` | Alias for `report` in v1. |

## Run artifacts

```
<output.base_dir>/<YYYY-MM-DD>-<name>/
  cases/
    round_1/<subject>__<task>.jsonl
    round_2/...
  stats/
    report.json
    class_report.csv
```

JSONL is one `CaseRecord` per line; runs are resumable — re-invoking
`pithos-eval run` skips case IDs already present unless `--no-resume`
is passed.

## Programmatic usage

```python
from pithos.eval import EvalRunner, Reporter
from pithos.eval.config import EvalConfig

cfg = EvalConfig.from_yaml("configs/eval/example.yaml")
records = EvalRunner(cfg).run()
report = Reporter(cfg.name, rounds=cfg.execution.rounds).build_report(records)
```
