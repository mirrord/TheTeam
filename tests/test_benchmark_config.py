"""Tests for benchmark configuration management."""

import os
import tempfile

from benchmarks.easy_problems.benchmark_config import (
    BenchmarkConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
)


def test_dataset_config_from_dict():
    """Test DatasetConfig creation from dictionary."""
    data = {"type": "multiple_choice", "path": "test/path.json"}
    config = DatasetConfig.from_dict(data)

    assert config.type == "multiple_choice"
    assert config.path == "test/path.json"


def test_dataset_config_defaults():
    """Test DatasetConfig with missing fields uses defaults."""
    data = {}
    config = DatasetConfig.from_dict(data)

    assert config.type == "multiple_choice"
    assert config.path == ""


def test_execution_config_from_dict():
    """Test ExecutionConfig creation from dictionary."""
    data = {"rounds": 5, "num_retries": 2, "temperature": 0.5, "max_tokens": 1024}
    config = ExecutionConfig.from_dict(data)

    assert config.rounds == 5
    assert config.num_retries == 2
    assert config.temperature == 0.5
    assert config.max_tokens == 1024


def test_execution_config_defaults():
    """Test ExecutionConfig defaults."""
    config = ExecutionConfig.from_dict({})

    assert config.rounds == 10
    assert config.num_retries == 3
    assert config.temperature == 0.0
    assert config.max_tokens == 2048


def test_output_config_from_dict():
    """Test OutputConfig creation from dictionary."""
    data = {
        "base_dir": "./my_results",
        "save_answers": False,
        "save_stats": True,
        "create_charts": False,
    }
    config = OutputConfig.from_dict(data)

    assert config.base_dir == "./my_results"
    assert config.save_answers is False
    assert config.save_stats is True
    assert config.create_charts is False


def test_output_config_defaults():
    """Test OutputConfig defaults."""
    config = OutputConfig.from_dict({})

    assert config.base_dir == "./results"
    assert config.save_answers is True
    assert config.save_stats is True
    assert config.create_charts is True


def test_benchmark_config_properties():
    """Test BenchmarkConfig computed properties."""
    config = BenchmarkConfig(
        name="TestBench",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models={},
        execution=ExecutionConfig(rounds=5),
        output=OutputConfig(base_dir="./out"),
    )

    # Test folder name includes timestamp and name
    assert "TestBench" in config.folder_name
    assert "-" in config.folder_name

    # Test paths
    assert config.answers_save_path.startswith("./out")
    assert "TestBench" in config.answers_save_path
    assert "llm_outputs" in config.answers_save_path

    assert config.stats_save_path.startswith("./out")
    assert "tables_and_charts" in config.stats_save_path


def test_benchmark_config_backward_compatibility():
    """Test backward compatibility properties."""
    config = BenchmarkConfig(
        name="TestBench",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models={},
        execution=ExecutionConfig(rounds=7, num_retries=2, temperature=0.3),
        output=OutputConfig(),
    )

    # Test answer_rounds alias
    assert config.answer_rounds == 7

    # Test answer_hyperparams
    hyperparams = config.answer_hyperparams
    assert hyperparams["num_retries"] == 2
    assert hyperparams["temperature"] == 0.3
    assert hyperparams["max_tokens"] == 2048

    # Test SaveFolders object
    assert hasattr(config, "SaveFolders")
    assert hasattr(config.SaveFolders, "answers_save_path")
    assert hasattr(config.SaveFolders, "stats_save_path")
    assert hasattr(config.SaveFolders, "sub_answer_folders")
    assert len(config.SaveFolders.sub_answer_folders) == 7


def test_benchmark_config_from_yaml():
    """Test loading BenchmarkConfig from YAML file."""
    yaml_content = """
name: "Test Benchmark"

dataset:
  type: "multiple_choice"
  path: "test/data.json"

models:
  model1:
    model: "test-model"
    service: "ollama"

execution:
  rounds: 3
  temperature: 0.2

output:
  base_dir: "./test_results"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = BenchmarkConfig.from_yaml(temp_path)

        assert config.name == "Test Benchmark"
        assert config.dataset.type == "multiple_choice"
        assert config.dataset.path == "test/data.json"
        assert "model1" in config.models
        assert config.models["model1"]["model"] == "test-model"
        assert config.execution.rounds == 3
        assert config.execution.temperature == 0.2
        assert config.output.base_dir == "./test_results"
    finally:
        os.unlink(temp_path)


def test_benchmark_config_from_yaml_or_default_with_file():
    """Test from_yaml_or_default with existing file."""
    yaml_content = """
name: "Custom Benchmark"
dataset:
  type: "free_form"
  path: "custom.json"
models: {}
execution:
  rounds: 1
output:
  base_dir: "./custom"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = BenchmarkConfig.from_yaml_or_default(temp_path)
        assert config.name == "Custom Benchmark"
    finally:
        os.unlink(temp_path)


def test_benchmark_config_from_yaml_or_default_fallback():
    """Test from_yaml_or_default with non-existent file falls back to defaults."""
    config = BenchmarkConfig.from_yaml_or_default("/nonexistent/path.yaml")

    # Should get default values
    assert config.execution.rounds == 10
    assert config.output.base_dir == "./results"


def test_benchmark_config_models():
    """Test models dictionary handling."""
    models = {
        "model1": {"model": "gemma3:27b", "service": "ollama"},
        "model2": {
            "model": "llama3.3",
            "service": "ollama_flow",
            "flowchart": "simple_reflect",
        },
    }

    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models=models,
        execution=ExecutionConfig(),
        output=OutputConfig(),
    )

    assert len(config.models) == 2
    assert "model1" in config.models
    assert "model2" in config.models
    assert config.models["model2"]["flowchart"] == "simple_reflect"
