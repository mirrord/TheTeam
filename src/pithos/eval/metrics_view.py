"""C.L.A.S.S. report builder.

Aggregates :class:`pithos.eval.models.CaseRecord` objects into a single
table with one row per subject and one column per **C**ost, **L**atency,
**A**ccuracy, **S**tability, **S**ecurity pillar.

Output is a plain ``list[dict]`` so callers without pandas can use it;
:func:`to_dataframe` converts to a ``pandas.DataFrame`` on demand.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean as _mean
from typing import Any, Iterable, Optional

from .models import CaseRecord, EvalReport
from .trace.analyzers.stability import stability_for_subject


def _round_score_lists(
    records: Iterable[CaseRecord],
) -> dict[tuple[str, int], list[float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in records:
        grouped[(r.subject_name, r.round_num)].append(r.grade.score)
    return grouped


def build_class_report(
    report: EvalReport,
    price_map: Optional[dict[str, dict[str, float]]] = None,
) -> list[dict[str, Any]]:
    """Build the C.L.A.S.S. summary rows from *report*.

    Args:
        report: Populated :class:`EvalReport`.
        price_map: Optional model -> ``{prompt_per_1k, completion_per_1k}``
            map for cost estimation. When omitted, cost falls back to
            the per-trace ``cost.detail.estimated_cost_usd`` issue
            (if a :class:`CostAnalyzer` ran) and finally to 0.
    """
    by_subject: dict[str, list[CaseRecord]] = defaultdict(list)
    for r in report.case_records:
        by_subject[r.subject_name].append(r)

    rows: list[dict[str, Any]] = []
    for subject, records in by_subject.items():
        accuracy = _mean(r.grade.score for r in records) if records else 0.0
        latencies = [
            r.trace.end_to_end_ms
            for r in records
            if r.trace is not None and r.trace.end_to_end_ms
        ]
        latency_avg = _mean(latencies) if latencies else 0.0

        # Cost: aggregate from per-record snapshot + price_map; fall back
        # to any "cost_estimate" issues emitted by CostAnalyzer.
        cost_total = 0.0
        if price_map:
            for r in records:
                snap = (
                    r.metrics_snapshot
                    or (r.trace.metrics_snapshot if r.trace else {})
                    or {}
                )
                for model, data in (snap.get("token_usage") or {}).items():
                    entry = price_map.get(model, {})
                    cost_total += float(entry.get("prompt_per_1k", 0.0)) * (
                        float(data.get("prompt_tokens", 0) or 0) / 1000.0
                    )
                    cost_total += float(entry.get("completion_per_1k", 0.0)) * (
                        float(data.get("completion_tokens", 0) or 0) / 1000.0
                    )
        else:
            for r in records:
                for issue in r.issues:
                    if issue.analyzer == "cost" and issue.code == "cost_estimate":
                        cost_total += float(issue.detail.get("estimated_cost_usd", 0.0))

        # Stability: variance of per-round mean scores.
        round_means = []
        round_groups: dict[int, list[float]] = defaultdict(list)
        for r in records:
            round_groups[r.round_num].append(r.grade.score)
        for scores in round_groups.values():
            round_means.append(_mean(scores) if scores else 0.0)
        stab_mean, stab_std, _ = stability_for_subject(round_means)

        security_issue_count = sum(
            1 for r in records for i in r.issues if i.analyzer == "security_stub"
        )

        rows.append(
            {
                "subject": subject,
                "cost_usd": round(cost_total, 6),
                "latency_ms_avg": round(latency_avg, 1),
                "accuracy_mean": round(accuracy, 2),
                "stability_std_dev": round(stab_std, 2),
                "stability_rounds": len(round_means),
                "security": (
                    "n/a" if security_issue_count == 0 else security_issue_count
                ),
                "case_count": len(records),
            }
        )

    rows.sort(key=lambda row: row["accuracy_mean"], reverse=True)
    return rows


def to_dataframe(rows: list[dict[str, Any]]):
    """Convert :func:`build_class_report` output to a pandas DataFrame.

    Imported lazily so the eval package stays usable without pandas.
    """
    import pandas as pd  # local import — optional dependency for tabular view

    df = pd.DataFrame(rows)
    if "subject" in df.columns:
        df = df.set_index("subject")
    return df
