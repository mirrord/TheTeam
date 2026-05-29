"""Capability-suite datasets — tool use, memory recall, self reflection.

Each dataset reads a JSON list of records and yields :class:`TaskCase`
objects shaped for its corresponding task class. Records may live on
disk (via ``path``) or be loaded from the bundled
:mod:`pithos.eval.datasets.builtins` directory by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..models import TaskCase
from .json_loader import JsonDataset


def _resolve_builtin(name: str) -> str:
    """Resolve a builtin dataset name (without extension) to a path."""
    here = Path(__file__).resolve().parent / "builtins"
    candidate = here / f"{name}.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Builtin dataset not found: {name}")
    return str(candidate)


class ToolUseDataset(JsonDataset):
    """Cases: ``{prompt, expected_tools[, expected_args, ...]}``."""

    task_type = "tool_use"

    def cases(self) -> Iterable[TaskCase]:
        for idx, item in enumerate(self._load_raw()):
            prompt = str(item.get("prompt", "")).strip()
            expected_tools = list(item.get("expected_tools", []) or [])
            metadata = {
                k: v for k, v in item.items() if k not in ("prompt", "expected_tools")
            }
            metadata["expected_tools"] = expected_tools
            yield TaskCase(
                case_id=f"tu_{idx}",
                task_type=self.task_type,
                prompt=prompt,
                expected={"tools": expected_tools},
                metadata=metadata,
            )


class MemoryRecallDataset(JsonDataset):
    """Cases: ``{seed_prompts[], recall_prompt, expected_recall}``."""

    task_type = "memory_recall"

    def cases(self) -> Iterable[TaskCase]:
        for idx, item in enumerate(self._load_raw()):
            seeds_raw = item.get("seed_prompts") or item.get("seed_prompt") or []
            if isinstance(seeds_raw, str):
                seeds = [seeds_raw]
            else:
                seeds = [str(s) for s in seeds_raw]
            recall = str(item.get("recall_prompt", "")).strip()
            expected = str(item.get("expected_recall", "")).strip()
            metadata = {
                k: v
                for k, v in item.items()
                if k
                not in (
                    "seed_prompts",
                    "seed_prompt",
                    "recall_prompt",
                    "expected_recall",
                )
            }
            metadata["expected_recall"] = expected
            yield TaskCase(
                case_id=f"mr_{idx}",
                task_type=self.task_type,
                prompt=recall,
                expected=expected,
                metadata=metadata,
                setup_prompts=seeds,
            )


class SelfReflectionDataset(JsonDataset):
    """Cases: ``{prompt, expected_correction}``.

    The graded answer is :attr:`TaskCase.expected`; the prompt is
    expected to contain a planted error the subject must correct.
    """

    task_type = "self_reflection"

    def cases(self) -> Iterable[TaskCase]:
        for idx, item in enumerate(self._load_raw()):
            prompt = str(item.get("prompt", "")).strip()
            expected = str(item.get("expected_correction", "")).strip()
            metadata = {
                k: v
                for k, v in item.items()
                if k not in ("prompt", "expected_correction")
            }
            yield TaskCase(
                case_id=f"sr_{idx}",
                task_type=self.task_type,
                prompt=prompt,
                expected=expected,
                metadata=metadata,
            )


def _path_from_spec(spec: dict) -> str:
    if "path" in spec and spec["path"]:
        return str(spec["path"])
    if "builtin" in spec and spec["builtin"]:
        return _resolve_builtin(str(spec["builtin"]))
    raise ValueError("Dataset spec requires either 'path' or 'builtin'")


def build_tool_use_dataset(spec: dict) -> ToolUseDataset:
    return ToolUseDataset(path=_path_from_spec(spec))


def build_memory_recall_dataset(spec: dict) -> MemoryRecallDataset:
    return MemoryRecallDataset(path=_path_from_spec(spec))


def build_self_reflection_dataset(spec: dict) -> SelfReflectionDataset:
    return SelfReflectionDataset(path=_path_from_spec(spec))
