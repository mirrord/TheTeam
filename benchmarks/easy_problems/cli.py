"""CLI interface for pithos benchmarking."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .benchmark_config import BenchmarkConfig


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="pithos-benchmark",
        description="Run benchmarks for pithos LLM agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default benchmark
  pithos-benchmark

  # Run with custom config
  pithos-benchmark --config my_benchmark.yaml

  # Run specific models only
  pithos-benchmark --models gemma3-27-base gemma3-27-simple-reflect

  # Run with custom number of rounds
  pithos-benchmark --rounds 5

  # Generate report from existing results
  pithos-benchmark report --path ./results/2024-03-12-Multi-Benchmark

  # List available benchmark configs
  pithos-benchmark list-configs
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Main run command (default)
    run_parser = subparsers.add_parser(
        "run",
        help="Run benchmark tests (default command)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML config file (default: configs/benchmarks/default_benchmark.yaml)",
    )
    run_parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=str,
        help="Specific models to run (default: all models in config)",
    )
    run_parser.add_argument(
        "--rounds", "-r", type=int, help="Number of rounds to run (overrides config)"
    )
    run_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for results (overrides config)",
    )
    run_parser.add_argument(
        "--dataset", "-d", type=str, help="Path to dataset JSON file (overrides config)"
    )
    run_parser.add_argument(
        "--no-charts", action="store_true", help="Skip chart generation"
    )
    run_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        help="Model temperature (0-1, overrides config)",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate report from existing results"
    )
    report_parser.add_argument(
        "path", nargs="?", type=str, help="Path to results directory"
    )
    report_parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")

    # List configs command
    list_parser = subparsers.add_parser(
        "list-configs", help="List available benchmark configurations"
    )

    # List datasets command
    datasets_parser = subparsers.add_parser(
        "list-datasets", help="List available benchmark datasets"
    )

    return parser


def apply_cli_overrides(
    config: BenchmarkConfig, args: argparse.Namespace
) -> BenchmarkConfig:
    """Apply CLI argument overrides to the loaded config."""
    if args.models:
        # Filter to only specified models
        config.models = {k: v for k, v in config.models.items() if k in args.models}
        if not config.models:
            print(f"Warning: None of the specified models found in config")
            print(f"Available models: {list(config.models.keys())}")

    if args.rounds:
        config.execution.rounds = args.rounds

    if args.output_dir:
        config.output.base_dir = args.output_dir

    if args.dataset:
        config.dataset.path = args.dataset

    if args.no_charts:
        config.output.create_charts = False

    if args.temperature is not None:
        config.execution.temperature = args.temperature

    return config


def list_available_configs():
    """List all available benchmark configuration files."""
    configs_dir = Path(__file__).parent.parent.parent / "configs" / "benchmarks"

    if not configs_dir.exists():
        print("No benchmark configs directory found.")
        return

    yaml_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

    if not yaml_files:
        print("No benchmark configuration files found.")
        return

    print("Available benchmark configurations:")
    for yaml_file in sorted(yaml_files):
        print(f"  - {yaml_file.name}")
        try:
            config = BenchmarkConfig.from_yaml(str(yaml_file))
            print(f"    Name: {config.name}")
            print(f"    Models: {len(config.models)}")
            print(f"    Rounds: {config.execution.rounds}")
        except Exception as e:
            print(f"    Error loading: {e}")


def list_available_datasets():
    """List all available benchmark datasets."""
    datasets_dir = Path(__file__).parent

    json_files = list(datasets_dir.glob("*benchmark*.json"))

    if not json_files:
        print("No benchmark dataset files found.")
        return

    print("Available benchmark datasets:")
    for json_file in sorted(json_files):
        print(f"  - {json_file.name}")
        try:
            import json

            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"    Questions: {len(data)}")
        except Exception as e:
            print(f"    Error loading: {e}")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Default to 'run' command if no command specified
    if args.command is None:
        args.command = "run"
        # Set default values for run command
        args.config = None
        args.models = None
        args.rounds = None
        args.output_dir = None
        args.dataset = None
        args.no_charts = False
        args.temperature = None

    return args
