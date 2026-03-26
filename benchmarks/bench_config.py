"""Benchmark configuration module.

This module provides a simple interface to load benchmark configurations.
It now uses the YAML-based configuration system in easy_problems/benchmark_config.py

For backward compatibility, you can still import from this module, but all
functionality is delegated to the new system.
"""

from .easy_problems.benchmark_config import (
    BenchmarkConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
)

__all__ = [
    "BenchmarkConfig",
    "DatasetConfig",
    "ExecutionConfig",
    "OutputConfig",
]
