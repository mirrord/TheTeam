"""File-change watcher for hot-reloading flowchart definitions."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import yaml

if TYPE_CHECKING:
    from .flowchart import Flowchart

logger = logging.getLogger(__name__)


class FlowchartWatcher:
    """Watches a YAML file and hot-reloads a flowchart when it changes."""

    def __init__(self) -> None:
        self._watch_path: Optional[Path] = None
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop: threading.Event = threading.Event()
        self._reload_lock: threading.Lock = threading.Lock()
        self._on_reload: Optional[Callable[["Flowchart"], None]] = None

    def start_watching(
        self,
        flowchart: "Flowchart",
        path: Union[str, Path],
        poll_interval: float = 1.0,
        on_reload: Optional[Callable[["Flowchart"], None]] = None,
    ) -> None:
        """Watch a YAML file and hot-reload the flowchart whenever it changes.

        Starts a daemon background thread that polls the file's modification
        time every *poll_interval* seconds.  When a change is detected the
        flowchart is re-parsed and its internal state is replaced in-place so
        that any existing reference to the object reflects the new definition.

        Ongoing executions are *not* interrupted; the new definition takes
        effect on the next :meth:`~Flowchart.run` call.

        Args:
            flowchart: The :class:`Flowchart` instance to reload into.
            path: Path to the YAML file to watch.
            poll_interval: Seconds between modification-time checks (default 1).
            on_reload: Optional callback invoked after each successful reload.

        Raises:
            FileNotFoundError: If *path* does not exist at call time.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot watch non-existent file: {path}")
        self.stop_watching()
        self._watch_path = path
        self._on_reload = on_reload
        self._watcher_stop.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            args=(flowchart, path, poll_interval),
            daemon=True,
            name=f"flowchart-watcher-{path.name}",
        )
        self._watcher_thread.start()

    def stop_watching(self) -> None:
        """Stop the file-change watcher thread, if running."""
        self._watcher_stop.set()
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=5.0)
        self._watcher_thread = None
        self._watch_path = None
        self._watcher_stop.clear()

    @property
    def is_watching(self) -> bool:
        """Return ``True`` if a file-change watcher thread is currently active."""
        return self._watcher_thread is not None and self._watcher_thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _watch_loop(
        self, flowchart: "Flowchart", path: Path, poll_interval: float
    ) -> None:
        """Background thread: poll *path* mtime and reload on change."""
        try:
            last_mtime = path.stat().st_mtime
        except OSError:
            logger.warning("Flowchart watcher: cannot stat %s, stopping.", path)
            return

        while not self._watcher_stop.wait(poll_interval):
            try:
                current_mtime = path.stat().st_mtime
            except OSError:
                continue
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                try:
                    self._reload_from_path(flowchart, path)
                    logger.debug("Flowchart hot-reloaded from %s", path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Flowchart hot-reload failed for %s: %s", path, exc)

    def _reload_from_path(self, flowchart: "Flowchart", path: Path) -> None:
        """Reload the flowchart in-place from *path*.

        Parses the YAML, rebuilds the graph, and replaces the instance's
        internal state atomically under ``_reload_lock``.
        """
        from .serialization import FlowchartSerializer

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        fresh = FlowchartSerializer.from_dict(data, flowchart.config_manager)

        with self._reload_lock:
            flowchart.graph = fresh.graph
            flowchart.start_node = fresh.start_node
            flowchart.condition_manager = fresh.condition_manager
            flowchart.message_router = fresh.message_router
            flowchart.reset()

        if self._on_reload is not None:
            try:
                self._on_reload(flowchart)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_reload callback raised: %s", exc)
