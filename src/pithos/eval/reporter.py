"""Eval report aggregation.

Reads a list of :class:`~pithos.eval.models.CaseRecord` (either fresh
from a runner or rehydrated from JSONL) and produces an
:class:`~pithos.eval.models.EvalReport` containing per-subject summary
statistics plus the C.L.A.S.S. row dump from
:func:`pithos.eval.metrics_view.build_class_report`.

Bootstrap CI math is reproduced from the legacy benchmark
``eval_utils.calculate_llm_stats`` using only stdlib so pandas / numpy
remain optional.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from collections import defaultdict
from statistics import mean as _mean, pstdev as _pstdev
from typing import Any, Iterable, Optional

from .metrics_view import build_class_report
from .models import CaseRecord, EvalReport
from .serde import _default

logger = logging.getLogger(__name__)


def _bootstrap_ci(
    scores: list[float], n: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Return a percentile bootstrap CI for the mean of *scores*.

    Uses the stdlib ``random`` module so we do not require numpy. ``n``
    defaults to 1000 (vs. legacy 10000) which is adequate for CI display
    and keeps the test suite fast; callers that want tighter intervals
    can pass a higher *n*.
    """
    if not scores:
        return (0.0, 0.0)
    if len(scores) == 1:
        return (float(scores[0]), float(scores[0]))
    rng = random.Random(1234)
    k = len(scores)
    means: list[float] = []
    for _ in range(n):
        sample = [scores[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo_idx = max(0, int(n * (alpha / 2)) - 1)
    hi_idx = min(n - 1, int(n * (1 - alpha / 2)))
    return (means[lo_idx], means[hi_idx])


class Reporter:
    """Build and persist :class:`EvalReport` instances."""

    def __init__(
        self,
        config_name: str = "eval",
        rounds: int = 1,
        *,
        price_map: Optional[dict[str, dict[str, float]]] = None,
        bootstrap_n: int = 1000,
    ) -> None:
        self.config_name = config_name
        self.rounds = rounds
        self.price_map = price_map or {}
        self.bootstrap_n = bootstrap_n

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def build_report(self, records: Iterable[CaseRecord]) -> EvalReport:
        records = list(records)
        report = EvalReport(
            config_name=self.config_name,
            rounds=self.rounds,
            case_records=records,
        )
        report.per_subject_stats = self._per_subject_stats(records)
        # build_class_report returns sorted list[dict]; index by subject for
        # the structured class_report attribute.
        rows = build_class_report(report, price_map=self.price_map)
        report.class_report = {row["subject"]: row for row in rows}
        report.issues_by_subject = self._issues_by_subject(records)
        return report

    def _per_subject_stats(
        self, records: list[CaseRecord]
    ) -> dict[str, dict[str, Any]]:
        by_subject: dict[str, list[CaseRecord]] = defaultdict(list)
        for r in records:
            by_subject[r.subject_name].append(r)

        stats: dict[str, dict[str, Any]] = {}
        for subject, recs in by_subject.items():
            scores = [r.grade.score for r in recs]
            ci_lo, ci_hi = _bootstrap_ci(scores, n=self.bootstrap_n)
            stats[subject] = {
                "mean_score": round(_mean(scores), 4) if scores else 0.0,
                "std_dev_score": round(_pstdev(scores), 4) if len(scores) > 1 else 0.0,
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "pass_rate": (
                    round(sum(1 for r in recs if r.grade.passed) / len(recs), 4)
                    if recs
                    else 0.0
                ),
                "case_count": len(recs),
            }
        return stats

    @staticmethod
    def _issues_by_subject(records: list[CaseRecord]) -> dict[str, list]:
        out: dict[str, list] = defaultdict(list)
        for r in records:
            out[r.subject_name].extend(r.issues)
        return dict(out)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self, report: EvalReport, run_dir: str) -> dict[str, str]:
        """Write ``report.json`` + ``class_report.csv`` under *run_dir*.

        Returns the paths written.
        """
        stats_dir = os.path.join(run_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        report_path = os.path.join(stats_dir, "report.json")
        csv_path = os.path.join(stats_dir, "class_report.csv")

        payload = {
            "config_name": report.config_name,
            "rounds": report.rounds,
            "generated_at": report.generated_at,
            "per_subject_stats": report.per_subject_stats,
            "class_report": report.class_report,
            "issues_by_subject": {
                k: [i for i in v] for k, v in report.issues_by_subject.items()
            },
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=_default, indent=2)

        rows = list(report.class_report.values())
        if rows:
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        return {"report_json": report_path, "class_report_csv": csv_path}


def load_records_from_run_dir(run_dir: str) -> list[CaseRecord]:
    """Rehydrate all ``CaseRecord`` objects from a previous run directory.

    Walks ``{run_dir}/cases/round_*/*.jsonl``. Traces are *not*
    reconstructed (analyzers should be re-run on the original traces
    if needed); this is enough for re-aggregation via
    :meth:`Reporter.build_report`.
    """
    from .runner import EvalRunner  # local import — avoid module cycle

    cases_root = os.path.join(run_dir, "cases")
    records: list[CaseRecord] = []
    if not os.path.isdir(cases_root):
        return records
    for round_dir in sorted(os.listdir(cases_root)):
        full = os.path.join(cases_root, round_dir)
        if not os.path.isdir(full):
            continue
        for fname in sorted(os.listdir(full)):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(full, fname), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(EvalRunner._record_from_dict(data))
    return records
