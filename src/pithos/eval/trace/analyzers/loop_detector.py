"""Loop detector — flags repeated ``(node_id, input_hash)`` revisits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Optional

from ...models import EvalTrace, TrajectoryIssue, TrajectoryIssueSeverity
from .base import Analyzer, AnalyzerContext


def _hash_inputs(inputs: dict) -> str:
    try:
        serialized = json.dumps(inputs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = repr(inputs)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


class LoopDetector(Analyzer):
    """Flag suspicious cycles within a single trace.

    Config keys:

    * ``max_repeats`` *(int, default 2)* — a ``(node_id, input_hash)``
      pair occurring strictly more than this many times triggers a
      ``cycle`` issue.
    * ``max_node_repeats`` *(int, default 5)* — a node_id appearing
      strictly more than this many times triggers a lower-severity
      ``repeated_node`` issue.
    """

    analyzer_name = "loop_detector"

    def analyze(
        self, trace: EvalTrace, ctx: Optional[AnalyzerContext] = None
    ) -> list[TrajectoryIssue]:
        max_repeats = int(self.config.get("max_repeats", 2))
        max_node_repeats = int(self.config.get("max_node_repeats", 5))

        keyed = Counter()
        node_counts = Counter()
        first_step: dict[Any, int] = {}
        for node in trace.nodes:
            h = _hash_inputs(node.inputs or {})
            key = (node.node_id, h)
            keyed[key] += 1
            node_counts[node.node_id] += 1
            first_step.setdefault(key, node.step)

        issues: list[TrajectoryIssue] = []
        for (node_id, h), count in keyed.items():
            if count > max_repeats:
                issues.append(
                    TrajectoryIssue(
                        analyzer=self.analyzer_name,
                        code="cycle",
                        message=(
                            f"Node {node_id!r} revisited {count} times with "
                            f"identical inputs (limit {max_repeats})."
                        ),
                        severity=TrajectoryIssueSeverity.ERROR,
                        step=first_step[(node_id, h)],
                        detail={
                            "node_id": node_id,
                            "input_hash": h,
                            "count": count,
                        },
                    )
                )

        for node_id, count in node_counts.items():
            if count > max_node_repeats:
                issues.append(
                    TrajectoryIssue(
                        analyzer=self.analyzer_name,
                        code="repeated_node",
                        message=(
                            f"Node {node_id!r} executed {count} times "
                            f"(soft limit {max_node_repeats})."
                        ),
                        severity=TrajectoryIssueSeverity.WARNING,
                        detail={"node_id": node_id, "count": count},
                    )
                )
        return issues
