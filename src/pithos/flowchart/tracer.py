"""Execution tracing and checkpoint/restore for flowcharts."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

from ..message import Message, MessageRouter
from .models import (
    EdgeInfo,
    ExecutionTrace,
    TraceEntry,
    serialise_message,
    deserialise_message,
)

if TYPE_CHECKING:
    pass


class ExecutionTracer:
    """Records per-step trace entries and supports checkpoint/restore."""

    def __init__(self) -> None:
        self._enabled: bool = False
        self._entries: list[TraceEntry] = []
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable execution tracing for subsequent runs."""
        self._enabled = True

    def begin_run(self) -> None:
        """Reset trace state at the start of a new run."""
        if self._enabled:
            self._entries = []
            self._start_time = datetime.now()
            self._end_time = None

    def end_run(self) -> None:
        """Mark the end of a run."""
        if self._enabled:
            self._end_time = datetime.now()

    def record_step(
        self,
        step: int,
        node_id: str,
        node_type: str,
        ts_start: datetime,
        ts_end: datetime,
        duration_ms: float,
        inputs: dict[str, Any],
        outputs: list[Any],
        edge: Optional[EdgeInfo],
        checkpoint: dict,
    ) -> None:
        """Append a trace entry for one executed step."""
        if not self._enabled:
            return
        entry = TraceEntry(
            step=step,
            node_id=node_id,
            node_type=(
                type(node_type).__name__
                if not isinstance(node_type, str)
                else node_type
            ),
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            duration_ms=duration_ms,
            inputs=inputs,
            outputs=outputs,
            edge=edge,
            _checkpoint=checkpoint,
        )
        self._entries.append(entry)

    def get_execution_trace(
        self, finished: bool, total_steps: int
    ) -> Optional[ExecutionTrace]:
        """Return the execution trace from the most recent run.

        Returns ``None`` if tracing is not enabled or no run has been
        performed since the last :meth:`enable` call.
        """
        if not self._enabled:
            return None
        return ExecutionTrace(
            entries=list(self._entries),
            completed=finished,
            total_steps=total_steps,
            start_time=self._start_time,
            end_time=self._end_time,
        )

    # ------------------------------------------------------------------
    # Checkpoint capture / restore
    # ------------------------------------------------------------------

    @staticmethod
    def capture_checkpoint(
        step_counter: int,
        finished: bool,
        node_route_info: dict[str, EdgeInfo],
        prev_output_messages: list[Message],
        message_router: MessageRouter,
    ) -> dict:
        """Snapshot the current execution state for later restore."""
        node_states_snap: dict[str, dict] = {}
        for node_id, state in message_router.node_states.items():
            if state.received_inputs:
                node_states_snap[node_id] = {
                    key: serialise_message(msg)
                    for key, msg in state.received_inputs.items()
                }

        return {
            "step_counter": step_counter,
            "finished": finished,
            "node_route_info": copy.deepcopy(node_route_info),
            "prev_output_messages": [
                serialise_message(m) for m in prev_output_messages
            ],
            "node_states": node_states_snap,
            "pending_messages": [
                serialise_message(m) for m in message_router.pending_messages
            ],
            "message_history": [
                serialise_message(m) for m in message_router.message_history
            ],
        }

    @staticmethod
    def apply_checkpoint(
        checkpoint: dict,
        message_router: MessageRouter,
    ) -> dict[str, Any]:
        """Restore execution state from a captured checkpoint dict.

        Returns a dict with ``finished``, ``step_counter``,
        ``node_route_info``, and ``prev_output_messages`` for the caller to
        install on the executor / flowchart.
        """
        for state in message_router.node_states.values():
            state.reset()
        for node_id, inputs in checkpoint["node_states"].items():
            if node_id in message_router.node_states:
                for input_key, msg_data in inputs.items():
                    msg = deserialise_message(msg_data)
                    message_router.node_states[node_id].receive_message(msg)

        message_router.pending_messages.clear()
        for msg_data in checkpoint["pending_messages"]:
            message_router.pending_messages.append(deserialise_message(msg_data))

        message_router.message_history.clear()
        for msg_data in checkpoint["message_history"]:
            message_router.message_history.append(deserialise_message(msg_data))

        return {
            "finished": checkpoint["finished"],
            "step_counter": checkpoint["step_counter"],
            "node_route_info": copy.deepcopy(checkpoint["node_route_info"]),
            "prev_output_messages": [
                deserialise_message(m) for m in checkpoint["prev_output_messages"]
            ],
        }

    def validate_restore_source(
        self, state: Union[ExecutionTrace, TraceEntry]
    ) -> TraceEntry:
        """Validate and extract the :class:`TraceEntry` to restore from.

        Raises:
            TypeError:  If *state* is not an expected type.
            ValueError: If the trace is empty or has no checkpoint data.
        """
        if isinstance(state, ExecutionTrace):
            if not state.entries:
                raise ValueError("Cannot restore from an empty ExecutionTrace")
            entry = state.entries[-1]
        elif isinstance(state, TraceEntry):
            entry = state
        else:
            raise TypeError(
                f"restore_state() expects ExecutionTrace or TraceEntry, "
                f"got {type(state).__name__}"
            )

        if not entry._checkpoint:
            raise ValueError(
                "TraceEntry has no checkpoint data. "
                "Ensure tracing was enabled before the run."
            )
        return entry
