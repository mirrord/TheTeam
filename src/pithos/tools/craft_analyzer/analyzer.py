"""Top-level :class:`CraftAnalyzer` facade + virtual-tool executor wrapper."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Optional

from ...config_manager import ConfigManager
from ..models import ToolMetadata, ToolResult
from ..provider import ToolProvider
from .dimensions import (
    build_user_prompt,
    dedup_notes,
    dimension_system_prompt,
    parse_notes,
)
from .ingest import chunk_text, resolve_source
from .models import CraftAnalysisConfig, CraftAnalysisRequest, CraftNote, CraftReport

logger = logging.getLogger(__name__)


class CraftAnalyzer:
    """High-level entry point: ingest → per-dimension analysis → notes.

    Construction is cheap; heavy resources (subagent, knowledge base) are
    created per :meth:`analyze` call so concurrent runs stay isolated.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        config: Optional[CraftAnalysisConfig] = None,
        agent_factory: Optional[Any] = None,
        memory_store: Optional[Any] = None,
    ) -> None:
        """Initialise the analyzer.

        Args:
            config_manager: ConfigManager used to load tool/agent configs.
            config: Optional pre-built config; otherwise loaded from
                ``configs/tools/craft_analysis_config.yaml``.
            agent_factory: Optional callable returning the craft-analyst
                subagent. When None the subagent is built from the registered
                agent config named by
                :attr:`CraftAnalysisConfig.subagent_config_name`.
            memory_store: Optional pre-built knowledge-base store (mainly for
                tests). When None a :class:`MemoryStore` is created lazily.
        """
        self.config_manager = config_manager
        self.config = config or self._load_config(config_manager)
        self._agent_factory = agent_factory
        self._memory_store = memory_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, request_or_source: Any, **kwargs: Any) -> CraftReport:
        """Run a craft analysis pass and return a :class:`CraftReport`.

        Accepts either a :class:`CraftAnalysisRequest` or a plain string,
        which is treated as raw source text for ergonomics.
        """
        if isinstance(request_or_source, CraftAnalysisRequest):
            request = request_or_source
        else:
            request = CraftAnalysisRequest(text=str(request_or_source), **kwargs)

        cfg = self.config
        started = time.time()
        errors: list[str] = []

        text, title = resolve_source(
            request,
            include=cfg.include,
            exclude=cfg.exclude,
            max_files=cfg.max_files,
        )

        dimensions = request.dimensions_override or list(cfg.dimensions)
        chunks = chunk_text(
            text,
            char_cap=cfg.chunk_char_cap,
            overlap=cfg.chunk_overlap,
            max_chunks=cfg.max_chunks,
        )
        if not chunks:
            return CraftReport(
                title=title,
                notes=[],
                errors=["no analyzable text after ingestion"],
                stats={"duration_seconds": round(time.time() - started, 2)},
            )

        memory_store = self._get_memory_store()
        source_entry_id = self._store_source(memory_store, text, title, len(chunks))

        subagent = self._build_agent()
        all_notes: list[CraftNote] = []
        for dimension in dimensions:
            dim_notes = self._analyze_dimension(
                dimension, chunks, title, subagent, cfg, errors
            )
            if cfg.dedup_notes:
                dim_notes = dedup_notes(dim_notes)
            dim_notes = dim_notes[: cfg.max_notes_per_dimension]
            all_notes.extend(dim_notes)

        self._store_notes(memory_store, all_notes, title, source_entry_id)

        stats = {
            "dimensions_analyzed": len(dimensions),
            "chunks_analyzed": len(chunks),
            "notes_produced": len(all_notes),
            "duration_seconds": round(time.time() - started, 2),
        }
        report = CraftReport(title=title, notes=all_notes, stats=stats, errors=errors)

        if cfg.write_document:
            try:
                document_path = _write_document(report, cfg.output_dir)
                report.document_path = document_path
                report.stats["document_path"] = document_path
            except Exception as exc:
                errors.append(f"failed to write document: {exc}")
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _analyze_dimension(
        self,
        dimension: str,
        chunks: list[str],
        title: str,
        agent: Any,
        cfg: CraftAnalysisConfig,
        errors: list[str],
    ) -> list[CraftNote]:
        """Run the subagent on ``dimension`` for every chunk and collect notes."""
        system_prompt = dimension_system_prompt(dimension, cfg.max_notes_per_dimension)
        try:
            agent.set_system_prompt(system_prompt)
        except Exception:
            pass

        notes: list[CraftNote] = []
        for chunk in chunks:
            prompt = build_user_prompt(dimension, chunk, title)
            try:
                reply = _send(agent, prompt, cfg.subagent_model)
            except Exception as exc:
                logger.warning(
                    "craft analysis failed [dimension=%s]: %s", dimension, exc
                )
                errors.append(f"{dimension}: analysis failed: {exc}")
                continue
            notes.extend(
                parse_notes(
                    reply,
                    dimension=dimension,
                    source_title=title,
                    max_notes=cfg.max_notes_per_dimension,
                )
            )
        return notes

    @staticmethod
    def _load_config(cm: ConfigManager) -> CraftAnalysisConfig:
        try:
            raw = cm.get_config("craft_analysis_config", "tools")
        except Exception:
            raw = None
        return CraftAnalysisConfig.from_dict(raw)

    def _get_memory_store(self) -> Optional[Any]:
        """Build (once) the persistent knowledge-base store."""
        if self._memory_store is not None:
            return self._memory_store
        try:
            from ..memory_tool import MemoryStore
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("MemoryStore unavailable: %s", exc)
            return None
        try:
            self._memory_store = MemoryStore(
                config_manager=self.config_manager,
                persist_directory=self.config.memory_persist_directory,
            )
        except Exception as exc:
            logger.warning("failed to initialise knowledge base: %s", exc)
            self._memory_store = None
        return self._memory_store

    def _store_source(
        self,
        memory_store: Optional[Any],
        text: str,
        title: str,
        chunk_count: int,
    ) -> Optional[str]:
        """Persist the source text for provenance; returns the entry id."""
        if memory_store is None:
            return None
        try:
            return memory_store.store(
                self.config.source_category,
                text,
                metadata={
                    "title": title,
                    "chunks": chunk_count,
                    "kind": "craft_source",
                },
            )
        except Exception as exc:
            logger.warning("failed to store source in knowledge base: %s", exc)
            return None

    def _store_notes(
        self,
        memory_store: Optional[Any],
        notes: list[CraftNote],
        title: str,
        source_entry_id: Optional[str],
    ) -> None:
        """Persist each note to the knowledge base, recording its entry id."""
        if memory_store is None or not notes:
            return
        for note in notes:
            try:
                entry_id = memory_store.store(
                    self.config.note_category,
                    note.note,
                    metadata={
                        "dimension": note.dimension,
                        "source_title": title,
                        "evidence": note.evidence,
                        "source_entry_id": source_entry_id,
                        "kind": "craft_note",
                    },
                )
                note.metadata["entry_id"] = entry_id
            except Exception as exc:
                logger.warning("failed to store craft note in knowledge base: %s", exc)

    def _build_agent(self) -> Any:
        """Build the craft-analyst subagent."""
        if self._agent_factory is not None:
            return self._agent_factory()

        from ...agent import OllamaAgent

        cfg = self.config
        agent_cfg = None
        try:
            agent_cfg = self.config_manager.get_config(
                cfg.subagent_config_name, "agents"
            )
        except Exception as exc:
            logger.debug("failed to load subagent config: %s", exc)

        if agent_cfg:
            agent = OllamaAgent.from_dict(agent_cfg, self.config_manager)
        else:
            model = cfg.subagent_model or "llama3.2"
            agent = OllamaAgent(
                default_model=model,
                agent_name="craft_analyst",
                system_prompt="",
            )
        if cfg.subagent_model:
            agent.default_model = cfg.subagent_model
        return agent


def _send(agent: Any, prompt: str, model: Optional[str]) -> str:
    """Send a prompt to the agent, tolerating backends without ``model``."""
    try:
        if model is not None:
            return agent.send(prompt, model=model) or ""
        return agent.send(prompt) or ""
    except TypeError:
        return agent.send(prompt) or ""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:40]
    return slug or "story"


def _write_document(report: CraftReport, output_dir: str) -> str:
    """Write the report markdown to ``output_dir`` and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"craft_{_slugify(report.title)}_{ts}.md"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report.to_markdown())
    return path


# ---------------------------------------------------------------------------
# Virtual-tool executor (mirrors NewsResearcherToolExecutor's shape)
# ---------------------------------------------------------------------------


class CraftAnalyzerToolExecutor(ToolProvider):
    """Adapts :class:`CraftAnalyzer` for use as a virtual ``craft-notes`` tool."""

    TOOL_NAME = "craft-notes"

    def __init__(
        self,
        config_manager: ConfigManager,
        analyzer: Optional[CraftAnalyzer] = None,
    ) -> None:
        self.config_manager = config_manager
        self._analyzer = analyzer

    @property
    def analyzer(self) -> CraftAnalyzer:
        if self._analyzer is None:
            self._analyzer = CraftAnalyzer(self.config_manager)
        return self._analyzer

    def discover(self, platform: str = "cross-platform") -> dict[str, ToolMetadata]:
        """Return the metadata entry for this virtual tool."""
        return {
            self.TOOL_NAME: ToolMetadata(
                name=self.TOOL_NAME,
                path="",
                description=(
                    "Analyze a story's creative-writing craft (characterization, "
                    "scene construction, themes, prose style, dialogue, plot "
                    "structure) and produce prescriptive how-to notes. Usage: "
                    "craft-notes <file path or raw text>"
                ),
                platform=platform,
                source="virtual",
                tool_type="craft_analysis",
            )
        }

    def can_execute(self, tool_name: str) -> bool:
        """Return True for the ``craft-notes`` tool name."""
        return tool_name == self.TOOL_NAME

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the craft-notes tool call extracted from *command*.

        Strips the leading ``craft-notes`` token; if the remainder is an
        existing file path it is analyzed as a file, otherwise it is treated
        as raw source text.
        """
        parts = command.strip().split(None, 1)
        arg = parts[1].strip() if len(parts) > 1 else ""
        if arg and os.path.isfile(arg):
            request = CraftAnalysisRequest(file_path=arg)
        else:
            request = CraftAnalysisRequest(text=arg)
        return self.run(request)

    def run(self, request: Any) -> ToolResult:
        """Execute a craft analysis pass and wrap the report as a ToolResult."""
        started = time.time()
        try:
            report = self.analyzer.analyze(request)
        except Exception as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exit_code=1,
                execution_time=round(time.time() - started, 2),
                command=self.TOOL_NAME,
                error_hint=str(exc),
            )
        return ToolResult(
            success=True,
            stdout=report.to_markdown(),
            stderr="",
            exit_code=0,
            execution_time=round(time.time() - started, 2),
            command=self.TOOL_NAME,
        )
