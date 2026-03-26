"""Benchmark configuration management using YAML files."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""

    type: str
    path: str

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        return cls(type=data.get("type", "multiple_choice"), path=data.get("path", ""))


@dataclass
class ExecutionConfig:
    """Configuration for benchmark execution parameters."""

    rounds: int = 10
    num_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = -1
    timeout: int = 120

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionConfig":
        return cls(
            rounds=data.get("rounds", 10),
            num_retries=data.get("num_retries", 3),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", -1),
            timeout=data.get("timeout", 120),
        )


@dataclass
class OutputConfig:
    """Configuration for benchmark output paths and options."""

    base_dir: str = "./results"
    save_answers: bool = True
    save_stats: bool = True
    create_charts: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "OutputConfig":
        return cls(
            base_dir=data.get("base_dir", "./results"),
            save_answers=data.get("save_answers", True),
            save_stats=data.get("save_stats", True),
            create_charts=data.get("create_charts", True),
        )


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    name: str
    dataset: DatasetConfig
    models: dict[str, dict[str, Any]]
    execution: ExecutionConfig
    output: OutputConfig
    _timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    @property
    def folder_name(self) -> str:
        """Generate folder name based on timestamp and benchmark name."""
        return f"{self._timestamp}-{self.name}"

    @property
    def answers_save_path(self) -> str:
        """Path to save LLM answers."""
        return os.path.join(self.output.base_dir, self.folder_name, "llm_outputs")

    @property
    def stats_save_path(self) -> str:
        """Path to save statistics and charts."""
        return os.path.join(self.output.base_dir, self.folder_name, "tables_and_charts")

    @property
    def answer_rounds(self) -> int:
        """Alias for execution.rounds for backwards compatibility."""
        return self.execution.rounds

    @property
    def answer_hyperparams(self) -> dict:
        """Hyperparameters dictionary for backwards compatibility."""
        return {
            "temperature": self.execution.temperature,
            "max_tokens": self.execution.max_tokens,
            "num_retries": self.execution.num_retries,
            "timeout": self.execution.timeout,
        }

    def __post_init__(self):
        """Set up SaveFolders object for backwards compatibility."""
        # Create SaveFolders as a simple namespace object
        SaveFoldersClass = type("SaveFolders", (), {})
        save_folders = SaveFoldersClass()
        save_folders.answers_save_path = self.answers_save_path
        save_folders.stats_save_path = self.stats_save_path
        save_folders.sub_answer_folders = [
            f"/round_{r + 1}" for r in range(self.execution.rounds)
        ]
        save_folders.sub_eval_folders = [""]
        # Use object.__setattr__ to set attribute on dataclass instance
        object.__setattr__(self, "SaveFolders", save_folders)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BenchmarkConfig":
        """Load benchmark configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            name=data.get("name", "Benchmark"),
            dataset=DatasetConfig.from_dict(data.get("dataset", {})),
            models=data.get("models", {}),
            execution=ExecutionConfig.from_dict(data.get("execution", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
        )

    @classmethod
    def from_yaml_or_default(cls, yaml_path: Optional[str] = None) -> "BenchmarkConfig":
        """Load config from YAML file, or use default if not specified."""
        if yaml_path and os.path.exists(yaml_path):
            return cls.from_yaml(yaml_path)

        # Try default config location
        default_config = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "benchmarks"
            / "default_benchmark.yaml"
        )
        if default_config.exists():
            return cls.from_yaml(str(default_config))

        # Fallback to hardcoded defaults
        return cls(
            name="Multi-Benchmark",
            dataset=DatasetConfig(
                type="multiple_choice",
                path="benchmarks/easy_problems/linguistic_benchmark_multi_choice.json",
            ),
            models={},
            execution=ExecutionConfig(),
            output=OutputConfig(),
        )
