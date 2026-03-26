"""Metrics collection for pithos - automatic tracking of token usage, response
times, tool call success rates, memory hit rates, and flowchart execution paths.

Usage::

    from pithos.metrics import MetricsCollector

    collector = MetricsCollector()
    agent.attach_metrics(collector)
    flowchart.attach_metrics(collector)

    # Optional: persist every 60 s
    collector.start_auto_save("./data/metrics.json", interval_seconds=60)

    # Inspect at any time
    snapshot = collector.get_snapshot()

    collector.stop_auto_save()
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-category metric containers
# ---------------------------------------------------------------------------


class TokenMetrics:
    """Accumulated token usage and timing for one model."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_calls: int = 0
        self.total_response_time_ms: float = 0.0
        self.min_response_time_ms: Optional[float] = None
        self.max_response_time_ms: Optional[float] = None

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        response_time_ms: float,
    ) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_calls += 1
        self.total_response_time_ms += response_time_ms
        if (
            self.min_response_time_ms is None
            or response_time_ms < self.min_response_time_ms
        ):
            self.min_response_time_ms = response_time_ms
        if (
            self.max_response_time_ms is None
            or response_time_ms > self.max_response_time_ms
        ):
            self.max_response_time_ms = response_time_ms

    @property
    def avg_response_time_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_response_time_ms / self.total_calls

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_calls": self.total_calls,
            "total_response_time_ms": self.total_response_time_ms,
            "avg_response_time_ms": self.avg_response_time_ms,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenMetrics":
        m = cls()
        m.prompt_tokens = d.get("prompt_tokens", 0)
        m.completion_tokens = d.get("completion_tokens", 0)
        m.total_calls = d.get("total_calls", 0)
        m.total_response_time_ms = d.get("total_response_time_ms", 0.0)
        m.min_response_time_ms = d.get("min_response_time_ms")
        m.max_response_time_ms = d.get("max_response_time_ms")
        return m


class ToolCallMetrics:
    """Accumulated success/failure counts and timing for one tool."""

    def __init__(self) -> None:
        self.successes: int = 0
        self.failures: int = 0
        self.total_execution_time_ms: float = 0.0

    def record(self, success: bool, execution_time_ms: float) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.total_execution_time_ms += execution_time_ms

    @property
    def total_calls(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        total = self.total_calls
        if total == 0:
            return 0.0
        return self.successes / total

    @property
    def avg_execution_time_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_calls

    def to_dict(self) -> dict:
        return {
            "successes": self.successes,
            "failures": self.failures,
            "total_calls": self.total_calls,
            "success_rate": self.success_rate,
            "total_execution_time_ms": self.total_execution_time_ms,
            "avg_execution_time_ms": self.avg_execution_time_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCallMetrics":
        m = cls()
        m.successes = d.get("successes", 0)
        m.failures = d.get("failures", 0)
        m.total_execution_time_ms = d.get("total_execution_time_ms", 0.0)
        return m


class MemoryMetrics:
    """Accumulated memory retrieve hit/miss rates and store counts."""

    def __init__(self) -> None:
        self.retrieve_hits: int = 0  # retrieves that returned ≥1 result
        self.retrieve_misses: int = 0  # retrieves that returned 0 results
        self.store_count: int = 0
        self.total_results_returned: int = 0

    def record_retrieve(self, hit: bool, result_count: int) -> None:
        if hit:
            self.retrieve_hits += 1
        else:
            self.retrieve_misses += 1
        self.total_results_returned += result_count

    def record_store(self) -> None:
        self.store_count += 1

    @property
    def retrieve_total(self) -> int:
        return self.retrieve_hits + self.retrieve_misses

    @property
    def hit_rate(self) -> float:
        total = self.retrieve_total
        if total == 0:
            return 0.0
        return self.retrieve_hits / total

    def to_dict(self) -> dict:
        return {
            "retrieve_hits": self.retrieve_hits,
            "retrieve_misses": self.retrieve_misses,
            "retrieve_total": self.retrieve_total,
            "hit_rate": self.hit_rate,
            "store_count": self.store_count,
            "total_results_returned": self.total_results_returned,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryMetrics":
        m = cls()
        m.retrieve_hits = d.get("retrieve_hits", 0)
        m.retrieve_misses = d.get("retrieve_misses", 0)
        m.store_count = d.get("store_count", 0)
        m.total_results_returned = d.get("total_results_returned", 0)
        return m


class FlowchartPathEntry:
    """One recorded node execution step in a flowchart run."""

    __slots__ = (
        "flowchart_name",
        "node_id",
        "node_type",
        "duration_ms",
        "from_node",
        "timestamp",
    )

    def __init__(
        self,
        flowchart_name: str,
        node_id: str,
        node_type: str,
        duration_ms: float,
        from_node: Optional[str],
        timestamp: str,
    ) -> None:
        self.flowchart_name = flowchart_name
        self.node_id = node_id
        self.node_type = node_type
        self.duration_ms = duration_ms
        self.from_node = from_node
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "flowchart_name": self.flowchart_name,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "duration_ms": self.duration_ms,
            "from_node": self.from_node,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FlowchartPathEntry":
        return cls(
            flowchart_name=d.get("flowchart_name", ""),
            node_id=d.get("node_id", ""),
            node_type=d.get("node_type", ""),
            duration_ms=d.get("duration_ms", 0.0),
            from_node=d.get("from_node"),
            timestamp=d.get("timestamp", ""),
        )


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Thread-safe collector for pithos runtime metrics.

    Tracks:

    * **Token usage per model** — prompt/completion tokens and call counts,
      recorded by :class:`~pithos.agent.OllamaAgent` after each LLM call.
    * **Response times** — min/avg/max latency per model, recorded alongside
      token usage.
    * **Tool call success rates** — per-tool success/failure counts and
      execution time, recorded by the agent's tool execution loop.
    * **Memory hit rates** — retrieve hit/miss ratio and store counts,
      recorded by the agent's memory operation loop.
    * **Flowchart execution paths** — ordered node execution log with
      timing and edge information, recorded by :class:`~pithos.flowchart.Flowchart`.

    Metrics accumulate in memory and may be persisted to a JSON file via
    :meth:`save` (manual) or :meth:`start_auto_save` (background thread).
    Previously saved metrics can be merged back with :meth:`load`.

    Attach to an agent::

        collector = MetricsCollector()
        agent.attach_metrics(collector)

    Attach to a flowchart::

        flowchart.attach_metrics(collector, name="my_flow")

    Configure auto-save::

        collector.start_auto_save("./data/metrics.json", interval_seconds=60)

    Inspect at any time::

        snapshot = collector.get_snapshot()
    """

    def __init__(self, max_path_entries: int = 10_000) -> None:
        """Create a new MetricsCollector.

        Args:
            max_path_entries: Maximum number of flowchart path entries to keep
                in memory.  Oldest entries are dropped when the limit is
                reached.  Default is 10,000.
        """
        self._lock = threading.Lock()
        self._token_usage: dict[str, TokenMetrics] = {}
        self._tool_calls: dict[str, ToolCallMetrics] = {}
        self._memory = MemoryMetrics()
        self._flowchart_paths: list[FlowchartPathEntry] = []
        self._max_path_entries = max_path_entries

        # Auto-save background thread state
        self._save_path: Optional[str] = None
        self._save_interval: float = 60.0
        self._save_thread: Optional[threading.Thread] = None
        self._save_stop = threading.Event()

    # ------------------------------------------------------------------
    # Recording helpers (called from integration points)
    # ------------------------------------------------------------------

    def record_token_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time_ms: float,
    ) -> None:
        """Record one completed LLM call's token usage and wall-clock time.

        Args:
            model: Model identifier string (e.g. ``"glm-4.7-flash"``).
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens generated.
            response_time_ms: Total wall-clock time of the call in milliseconds.
        """
        with self._lock:
            if model not in self._token_usage:
                self._token_usage[model] = TokenMetrics()
            self._token_usage[model].record(
                prompt_tokens, completion_tokens, response_time_ms
            )

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
    ) -> None:
        """Record one tool call's outcome.

        Args:
            tool_name: Name of the CLI tool that was executed.
            success: Whether the tool exited successfully.
            execution_time_ms: Wall-clock execution time in milliseconds.
        """
        with self._lock:
            if tool_name not in self._tool_calls:
                self._tool_calls[tool_name] = ToolCallMetrics()
            self._tool_calls[tool_name].record(success, execution_time_ms)

    def record_memory_retrieve(self, result_count: int) -> None:
        """Record a memory retrieve operation.

        Args:
            result_count: Number of results returned (0 = miss, >0 = hit).
        """
        with self._lock:
            self._memory.record_retrieve(
                hit=result_count > 0, result_count=result_count
            )

    def record_memory_store(self) -> None:
        """Record a memory store operation."""
        with self._lock:
            self._memory.record_store()

    def record_flowchart_step(
        self,
        flowchart_name: str,
        node_id: str,
        node_type: str,
        duration_ms: float,
        from_node: Optional[str] = None,
    ) -> None:
        """Record one node execution step in a flowchart run.

        Args:
            flowchart_name: Identifying name of the flowchart.
            node_id: ID of the node that executed.
            node_type: Class name of the node type.
            duration_ms: Node execution duration in milliseconds.
            from_node: ID of the predecessor node, or ``None`` for the start.
        """
        entry = FlowchartPathEntry(
            flowchart_name=flowchart_name,
            node_id=node_id,
            node_type=node_type,
            duration_ms=duration_ms,
            from_node=from_node,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            if len(self._flowchart_paths) >= self._max_path_entries:
                # Ring-buffer: drop oldest entries to stay within bounds.
                self._flowchart_paths = self._flowchart_paths[
                    -(self._max_path_entries - 1) :
                ]
            self._flowchart_paths.append(entry)

    # ------------------------------------------------------------------
    # Query / snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return a complete metrics snapshot as a plain dictionary.

        The returned dict is safe to serialize with ``json.dump``.  Keys:

        * ``token_usage`` — ``{model: {...}}`` with tokens and timing.
        * ``tool_calls`` — ``{tool_name: {...}}`` with success rates.
        * ``memory`` — retrieve hit/miss rates and store counts.
        * ``flowchart_paths`` — list of recorded node steps.
        """
        with self._lock:
            return {
                "token_usage": {
                    model: m.to_dict() for model, m in self._token_usage.items()
                },
                "tool_calls": {
                    tool: m.to_dict() for tool, m in self._tool_calls.items()
                },
                "memory": self._memory.to_dict(),
                "flowchart_paths": [e.to_dict() for e in self._flowchart_paths],
            }

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        with self._lock:
            self._token_usage.clear()
            self._tool_calls.clear()
            self._memory = MemoryMetrics()
            self._flowchart_paths.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the metrics snapshot to a JSON file.

        Writes atomically (temp file + rename) so concurrent readers never
        observe a partial write.  Parent directories are created as needed.

        Args:
            path: Destination JSON file path.
        """
        snapshot = self.get_snapshot()
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            tmp_path.replace(file_path)
        except Exception:
            logger.exception("Failed to save metrics to %s", path)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def load(self, path: str) -> None:
        """Load previously persisted metrics, merging with current data.

        All numeric counters are summed.  Flowchart path entries are appended
        (the combined list is capped at ``max_path_entries``).

        Args:
            path: JSON file produced by a previous :meth:`save` call.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            self._merge_from_dict(data)

    def _merge_from_dict(self, data: dict) -> None:
        """Merge data loaded from JSON into current state (caller holds lock)."""
        for model, m_dict in data.get("token_usage", {}).items():
            loaded = TokenMetrics.from_dict(m_dict)
            if model in self._token_usage:
                existing = self._token_usage[model]
                existing.prompt_tokens += loaded.prompt_tokens
                existing.completion_tokens += loaded.completion_tokens
                existing.total_calls += loaded.total_calls
                existing.total_response_time_ms += loaded.total_response_time_ms
                if loaded.min_response_time_ms is not None:
                    if (
                        existing.min_response_time_ms is None
                        or loaded.min_response_time_ms < existing.min_response_time_ms
                    ):
                        existing.min_response_time_ms = loaded.min_response_time_ms
                if loaded.max_response_time_ms is not None:
                    if (
                        existing.max_response_time_ms is None
                        or loaded.max_response_time_ms > existing.max_response_time_ms
                    ):
                        existing.max_response_time_ms = loaded.max_response_time_ms
            else:
                self._token_usage[model] = loaded

        for tool, t_dict in data.get("tool_calls", {}).items():
            loaded_t = ToolCallMetrics.from_dict(t_dict)
            if tool in self._tool_calls:
                existing_t = self._tool_calls[tool]
                existing_t.successes += loaded_t.successes
                existing_t.failures += loaded_t.failures
                existing_t.total_execution_time_ms += loaded_t.total_execution_time_ms
            else:
                self._tool_calls[tool] = loaded_t

        mem_dict = data.get("memory", {})
        loaded_mem = MemoryMetrics.from_dict(mem_dict)
        self._memory.retrieve_hits += loaded_mem.retrieve_hits
        self._memory.retrieve_misses += loaded_mem.retrieve_misses
        self._memory.store_count += loaded_mem.store_count
        self._memory.total_results_returned += loaded_mem.total_results_returned

        incoming = [
            FlowchartPathEntry.from_dict(e) for e in data.get("flowchart_paths", [])
        ]
        combined = self._flowchart_paths + incoming
        self._flowchart_paths = combined[-self._max_path_entries :]

    # ------------------------------------------------------------------
    # Auto-save background thread
    # ------------------------------------------------------------------

    def start_auto_save(self, path: str, interval_seconds: float = 60.0) -> None:
        """Start a background thread that periodically saves metrics to *path*.

        Any previously running auto-save thread is stopped first.

        Args:
            path: Destination JSON file path.
            interval_seconds: Seconds between saves.  Must be > 0.

        Raises:
            ValueError: If *interval_seconds* is not positive.
        """
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self._save_path = path
        self._save_interval = interval_seconds
        self.stop_auto_save()
        self._save_stop.clear()
        self._save_thread = threading.Thread(
            target=self._auto_save_loop,
            daemon=True,
            name="pithos-metrics-autosave",
        )
        self._save_thread.start()
        logger.debug(
            "Metrics auto-save started → %s (every %.1fs)", path, interval_seconds
        )

    def stop_auto_save(self) -> None:
        """Stop the background auto-save thread, if running.

        Blocks until the thread exits (up to 5 s).  Triggers one final save
        so the most recent data is always flushed before shutdown.
        """
        self._save_stop.set()
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join(timeout=5.0)
        self._save_thread = None
        self._save_stop.clear()
        # Flush final state when explicitly stopped
        if self._save_path:
            self.save(self._save_path)

    @property
    def is_auto_saving(self) -> bool:
        """``True`` if a background auto-save thread is currently active."""
        return self._save_thread is not None and self._save_thread.is_alive()

    def _auto_save_loop(self) -> None:
        """Background thread: save every ``_save_interval`` seconds."""
        while not self._save_stop.wait(self._save_interval):
            if self._save_path:
                self.save(self._save_path)
                logger.debug("Metrics auto-saved to %s", self._save_path)
