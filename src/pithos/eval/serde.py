"""JSON serialisation helpers for eval dataclasses.

The runner persists :class:`~pithos.eval.models.CaseRecord` objects as
JSONL so runs are resumable and reports can be regenerated offline.
``MetricsCollector`` instances are *not* serialised — analyzers and the
reporter consume ``metrics_snapshot`` instead.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def _default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Enum):
        return o.value
    if is_dataclass(o):
        return asdict(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def record_to_dict(record: Any) -> dict[str, Any]:
    """Convert a dataclass (e.g. CaseRecord) to a JSON-safe dict."""
    if is_dataclass(record):
        record = asdict(record)
    return json.loads(json.dumps(record, default=_default))


def dump_record(record: Any) -> str:
    """Serialise a record to a single-line JSON string."""
    payload = asdict(record) if is_dataclass(record) else record
    return json.dumps(payload, default=_default, ensure_ascii=False)


def load_records(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file and return decoded dicts (one per line).

    Missing files return an empty list so callers can use the result
    directly for resume logic.
    """
    import os

    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
