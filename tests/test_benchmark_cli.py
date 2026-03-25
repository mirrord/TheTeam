"""Tests for benchmark CLI interface."""

from benchmarks.easy_problems.cli import (
    parse_args,
    apply_cli_overrides,
)
from benchmarks.easy_problems.benchmark_config import (
    BenchmarkConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
)


def test_parse_args_default():
    """Test parsing with no arguments defaults to 'run' command."""
    args = parse_args([])

    assert args.command == "run"
    assert args.config is None
    assert args.models is None
    assert args.rounds is None


def test_parse_args_run_with_config():
    """Test parsing 'run' command with config file."""
    args = parse_args(["run", "--config", "my_config.yaml"])

    assert args.command == "run"
    assert args.config == "my_config.yaml"


def test_parse_args_run_with_models():
    """Test parsing 'run' command with specific models."""
    args = parse_args(["run", "--models", "model1", "model2"])

    assert args.command == "run"
    assert args.models == ["model1", "model2"]


def test_parse_args_run_with_rounds():
    """Test parsing 'run' command with rounds override."""
    args = parse_args(["run", "--rounds", "5"])

    assert args.command == "run"
    assert args.rounds == 5


def test_parse_args_run_with_output_dir():
    """Test parsing 'run' command with output directory."""
    args = parse_args(["run", "--output-dir", "./my_results"])

    assert args.command == "run"
    assert args.output_dir == "./my_results"


def test_parse_args_run_with_dataset():
    """Test parsing 'run' command with dataset override."""
    args = parse_args(["run", "--dataset", "path/to/dataset.json"])

    assert args.command == "run"
    assert args.dataset == "path/to/dataset.json"


def test_parse_args_run_no_charts():
    """Test parsing 'run' command with --no-charts flag."""
    args = parse_args(["run", "--no-charts"])

    assert args.command == "run"
    assert args.no_charts is True


def test_parse_args_run_with_temperature():
    """Test parsing 'run' command with temperature override."""
    args = parse_args(["run", "--temperature", "0.7"])

    assert args.command == "run"
    assert args.temperature == 0.7


def test_parse_args_report():
    """Test parsing 'report' command."""
    args = parse_args(["report", "./results/2024-03-12-Test"])

    assert args.command == "report"
    assert args.path == "./results/2024-03-12-Test"


def test_parse_args_report_with_date():
    """Test parsing 'report' command with date."""
    args = parse_args(["report", "--date", "2024-03-12"])

    assert args.command == "report"
    assert args.date == "2024-03-12"


def test_parse_args_list_configs():
    """Test parsing 'list-configs' command."""
    args = parse_args(["list-configs"])

    assert args.command == "list-configs"


def test_parse_args_list_datasets():
    """Test parsing 'list-datasets' command."""
    args = parse_args(["list-datasets"])

    assert args.command == "list-datasets"


def test_apply_cli_overrides_models():
    """Test applying models override."""
    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models={
            "model1": {"model": "test1", "service": "ollama"},
            "model2": {"model": "test2", "service": "ollama"},
            "model3": {"model": "test3", "service": "ollama"},
        },
        execution=ExecutionConfig(),
        output=OutputConfig(),
    )

    args = parse_args(["run", "--models", "model1", "model3"])
    config = apply_cli_overrides(config, args)

    assert len(config.models) == 2
    assert "model1" in config.models
    assert "model3" in config.models
    assert "model2" not in config.models


def test_apply_cli_overrides_rounds():
    """Test applying rounds override."""
    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models={},
        execution=ExecutionConfig(rounds=10),
        output=OutputConfig(),
    )

    args = parse_args(["run", "--rounds", "3"])
    config = apply_cli_overrides(config, args)

    assert config.execution.rounds == 3


def test_apply_cli_overrides_output_dir():
    """Test applying output directory override."""
    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models={},
        execution=ExecutionConfig(),
        output=OutputConfig(base_dir="./results"),
    )

    args = parse_args(["run", "--output-dir", "./custom"])
    config = apply_cli_overrides(config, args)

    assert config.output.base_dir == "./custom"


def test_apply_cli_overrides_dataset():
    """Test applying dataset override."""
    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="original.json"),
        models={},
        execution=ExecutionConfig(),
        output=OutputConfig(),
    )

    args = parse_args(["run", "--dataset", "new.json"])
    config = apply_cli_overrides(config, args)

    assert config.dataset.path == "new.json"


def test_apply_cli_overrides_no_charts():
    """Test applying no-charts flag."""
    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models={},
        execution=ExecutionConfig(),
        output=OutputConfig(create_charts=True),
    )

    args = parse_args(["run", "--no-charts"])
    config = apply_cli_overrides(config, args)

    assert config.output.create_charts is False


def test_apply_cli_overrides_temperature():
    """Test applying temperature override."""
    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models={},
        execution=ExecutionConfig(temperature=0.0),
        output=OutputConfig(),
    )

    args = parse_args(["run", "--temperature", "0.5"])
    config = apply_cli_overrides(config, args)

    assert config.execution.temperature == 0.5


def test_apply_cli_overrides_all():
    """Test applying multiple overrides at once."""
    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="original.json"),
        models={
            "model1": {"model": "test1", "service": "ollama"},
            "model2": {"model": "test2", "service": "ollama"},
        },
        execution=ExecutionConfig(rounds=10, temperature=0.0),
        output=OutputConfig(base_dir="./results", create_charts=True),
    )

    args = parse_args(
        [
            "run",
            "--models",
            "model1",
            "--rounds",
            "2",
            "--output-dir",
            "./custom",
            "--dataset",
            "new.json",
            "--no-charts",
            "--temperature",
            "0.3",
        ]
    )
    config = apply_cli_overrides(config, args)

    assert len(config.models) == 1
    assert "model1" in config.models
    assert config.execution.rounds == 2
    assert config.output.base_dir == "./custom"
    assert config.dataset.path == "new.json"
    assert config.output.create_charts is False
    assert config.execution.temperature == 0.3


def test_apply_cli_overrides_invalid_models():
    """Test that invalid model names filter out and show warning."""
    config = BenchmarkConfig(
        name="Test",
        dataset=DatasetConfig(type="multiple_choice", path="test.json"),
        models={
            "model1": {"model": "test1", "service": "ollama"},
        },
        execution=ExecutionConfig(),
        output=OutputConfig(),
    )

    args = parse_args(["run", "--models", "nonexistent"])

    # Should filter out invalid models
    config = apply_cli_overrides(config, args)
    assert len(config.models) == 0


def test_shorthand_arguments():
    """Test that shorthand argument flags work."""
    args = parse_args(
        [
            "run",
            "-c",
            "config.yaml",
            "-m",
            "model1",
            "-r",
            "5",
            "-o",
            "./output",
            "-d",
            "dataset.json",
            "-t",
            "0.5",
        ]
    )

    assert args.config == "config.yaml"
    assert args.models == ["model1"]
    assert args.rounds == 5
    assert args.output_dir == "./output"
    assert args.dataset == "dataset.json"
    assert args.temperature == 0.5
