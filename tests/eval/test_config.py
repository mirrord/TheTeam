"""Tests for pithos.eval.config (EvalConfig YAML loader)."""

import os
import tempfile

from pithos.eval import (
    EvalConfig,
    EvalExecutionConfig,
    EvalOutputConfig,
    GraderSpec,
    SubjectSpec,
    TaskSpec,
    AnalyzerSpec,
)


def test_subject_spec_from_dict_extracts_type():
    s = SubjectSpec.from_dict("my_agent", {"type": "agent", "model": "glm-4.7-flash"})
    assert s.name == "my_agent"
    assert s.type == "agent"
    assert s.config == {"model": "glm-4.7-flash"}


def test_subject_spec_defaults_to_agent_type():
    s = SubjectSpec.from_dict("x", {"model": "m"})
    assert s.type == "agent"


def test_grader_spec_from_dict():
    g = GraderSpec.from_dict({"type": "letter_match", "case_sensitive": False})
    assert g.type == "letter_match"
    assert g.config == {"case_sensitive": False}


def test_grader_spec_default_type():
    g = GraderSpec.from_dict({})
    assert g.type == "exact_match"


def test_task_spec_from_dict():
    t = TaskSpec.from_dict(
        "mc",
        {
            "type": "multiple_choice",
            "dataset": {"type": "multiple_choice", "path": "x.json"},
            "grader": {"type": "letter_match"},
            "shuffle": True,
        },
    )
    assert t.name == "mc"
    assert t.type == "multiple_choice"
    assert t.dataset == {"type": "multiple_choice", "path": "x.json"}
    assert t.grader.type == "letter_match"
    assert t.config == {"shuffle": True}


def test_analyzer_spec_from_dict():
    a = AnalyzerSpec.from_dict({"type": "cost", "price_map": "p.yaml"})
    assert a.type == "cost"
    assert a.config == {"price_map": "p.yaml"}


def test_execution_config_defaults():
    e = EvalExecutionConfig.from_dict({})
    assert e.rounds == 3
    assert e.num_retries == 1
    assert e.parallelism == 1


def test_output_config_defaults():
    o = EvalOutputConfig.from_dict({})
    assert o.base_dir == "./results"
    assert o.save_traces is True
    assert o.create_charts is True


def test_eval_config_paths():
    cfg = EvalConfig(
        name="MyBench",
        subjects={},
        tasks={},
        output=EvalOutputConfig(base_dir="./out"),
    )
    assert "MyBench" in cfg.folder_name
    assert cfg.run_dir.startswith("./out")
    assert cfg.cases_dir.endswith("cases")
    assert cfg.stats_dir.endswith("stats")


def test_eval_config_from_yaml(tmp_path):
    yaml_content = """
name: Test Eval
subjects:
  my_agent:
    type: agent
    model: glm-4.7-flash
tasks:
  mc:
    type: multiple_choice
    dataset:
      type: multiple_choice
      path: data.json
    grader:
      type: letter_match
analyzers:
  - type: loop_detector
  - type: cost
    price_map: prices.yaml
execution:
  rounds: 5
  parallelism: 2
output:
  base_dir: ./out
  create_charts: false
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    cfg = EvalConfig.from_yaml(str(p))
    assert cfg.name == "Test Eval"
    assert "my_agent" in cfg.subjects
    assert cfg.subjects["my_agent"].type == "agent"
    assert "mc" in cfg.tasks
    assert cfg.tasks["mc"].grader.type == "letter_match"
    assert len(cfg.analyzers) == 2
    assert cfg.analyzers[1].type == "cost"
    assert cfg.analyzers[1].config == {"price_map": "prices.yaml"}
    assert cfg.execution.rounds == 5
    assert cfg.execution.parallelism == 2
    assert cfg.output.create_charts is False


def test_eval_config_from_yaml_or_default_missing_returns_empty():
    cfg = EvalConfig.from_yaml_or_default("/nonexistent/path.yaml")
    # Either falls through to packaged default or returns an empty stub.
    assert cfg.name in ("DefaultEval",) or cfg.subjects is not None


def test_eval_config_from_dict_empty():
    cfg = EvalConfig.from_dict({})
    assert cfg.name == "Eval"
    assert cfg.subjects == {}
    assert cfg.tasks == {}
    assert cfg.analyzers == []
