"""Datasets — load :class:`TaskCase` collections from JSON or builtins."""

from .base import Dataset, build_dataset, register_dataset_type
from .capability_suites import (
    MemoryRecallDataset,
    SelfReflectionDataset,
    ToolUseDataset,
)
from .json_loader import (
    FreeFormDataset,
    JsonDataset,
    MultipleChoiceDataset,
    load_dataset,
)

__all__ = [
    "Dataset",
    "JsonDataset",
    "MultipleChoiceDataset",
    "FreeFormDataset",
    "ToolUseDataset",
    "MemoryRecallDataset",
    "SelfReflectionDataset",
    "build_dataset",
    "register_dataset_type",
    "load_dataset",
]
