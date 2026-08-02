"""Top-level :class:`WebResearcher` facade + virtual-tool executor wrapper."""

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
from .agent_loop import ResearchLoop, per_domain_stats
from .editor import edit_summary, verify_citations, verify_sources
from .fetcher import Fetcher
from .models import ResearchReport, WebResearchConfig, WebResearchRequest
from .search import DuckDuckGoSearch
from .store import ExcerptStore
from .summarizer import synthesize

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:40]
    return slug or "inquiry"


def _write_document(report: ResearchReport, output_dir: str) -> str:
    """Write the report markdown to ``output_dir`` and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"web_{_slugify(report.inquiry)}_{ts}.md"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report.to_markdown())
    return path


class WebResearcher:
    """High-level entry point: orchestrates loop + summarizer.

    Construction is cheap; heavy resources (ChromaDB client, requests
    session, subagent) are created per :meth:`research` call so concurrent
    runs stay isolated.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        config: Optional[WebResearchConfig] = None,
        agent_factory: Optional[Any] = None,
        editor_agent_factory: Optional[Any] = None,
    ) -> None:
        """Initialise the researcher.

        Args:
            config_manager: ConfigManager used to load tool/agent configs.
            config: Optional pre-built config; otherwise loaded from
                ``configs/tools/web_research_config.yaml``.
            agent_factory: Optional callable returning a subagent. When None
                the subagent is built from the registered agent config named
                by :attr:`WebResearchConfig.subagent_config_name`.
            editor_agent_factory: Optional callable returning the editor
                subagent used for citation verification. Falls back to
                :attr:`agent_factory` when None, then to the registered
                ``editor`` agent config.
        """
        self.config_manager = config_manager
        self.config = config or self._load_config(config_manager)
        self._agent_factory = agent_factory
        self._editor_agent_factory = editor_agent_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, request_or_inquiry, **kwargs: Any) -> ResearchReport:
        """Run a research pass for ``request_or_inquiry`` and return a report.

        Accepts either a :class:`WebResearchRequest` or a plain inquiry
        string for ergonomics.
        """
        if isinstance(request_or_inquiry, WebResearchRequest):
            request = request_or_inquiry
        else:
            request = WebResearchRequest(inquiry=str(request_or_inquiry), **kwargs)

        if not request.inquiry or not request.inquiry.strip():
            raise ValueError("inquiry cannot be empty")

        cfg = self.config
        domains = request.domains_override or list(cfg.domains)
        if not domains:
            return ResearchReport(
                inquiry=request.inquiry,
                summary=(
                    "No domains are configured for web research. Add domains under "
                    "`domains:` in `configs/tools/web_research_config.yaml` or pass "
                    "`domains_override` on the request."
                ),
                excerpts=[],
                sources=[],
                errors=["no whitelisted domains configured"],
            )

        started = time.time()
        agent = self._build_agent()

        fetcher = Fetcher(
            whitelist=domains,
            user_agent=cfg.user_agent,
            timeout=cfg.request_timeout,
            max_bytes=cfg.max_page_bytes,
            per_domain_rps=cfg.per_domain_rps,
            respect_robots=cfg.respect_robots,
        )
        search = DuckDuckGoSearch(
            fetcher=fetcher,
            results_per_domain=cfg.search_results_per_domain,
        )
        collection_name = _make_collection_name(request.inquiry)
        store = ExcerptStore(
            collection_name=collection_name,
            persist_directory=cfg.persist_directory,
            similarity_threshold=cfg.dedup_similarity,
        )

        # Seed with any extra URLs the caller provided up front.
        loop = ResearchLoop(
            config=WebResearchConfig(**{**cfg.__dict__, "domains": domains}),
            agent=agent,
            fetcher=fetcher,
            search=search,
            store=store,
        )
        for url in request.extra_seed_urls or []:
            if url not in loop.candidates:
                loop.candidates.append(url)

        try:
            loop.run(request.inquiry)
        except Exception as exc:
            logger.exception("research loop crashed")
            loop.errors.append(f"loop crashed: {exc}")

        excerpts = store.all()
        summary, ordered_sources = synthesize(
            inquiry=request.inquiry,
            excerpts=excerpts,
            agent=agent,
            model=cfg.summarizer_model,
        )

        # --- Citation verification (editor subagent) ---------------------
        citation_checks = []
        source_statuses = []
        original_summary = None
        editor_stats: dict[str, Any] = {}
        if cfg.verify_citations and ordered_sources:
            editor_started = time.time()
            try:
                editor_agent = self._build_editor_agent()
                source_statuses = verify_sources(ordered_sources, store, fetcher)
                citation_checks = verify_citations(
                    summary=summary,
                    sources=ordered_sources,
                    excerpts=excerpts,
                    agent=editor_agent,
                    model=cfg.editor_model,
                )
                dead = sum(1 for s in source_statuses if not s.exists)
                unsupported = sum(
                    1 for c in citation_checks if c.verdict == "unsupported"
                )
                rewrote = False
                if dead > 0 or unsupported > 0:
                    new_summary = edit_summary(
                        inquiry=request.inquiry,
                        summary=summary,
                        citation_checks=citation_checks,
                        source_statuses=source_statuses,
                        agent=editor_agent,
                        model=cfg.editor_model,
                    )
                    if new_summary and new_summary.strip() != summary.strip():
                        original_summary = summary
                        summary = new_summary
                        rewrote = True
                editor_stats = {
                    "citations_total": len(citation_checks),
                    "citations_unsupported": unsupported,
                    "dead_sources": dead,
                    "editor_rewrote": rewrote,
                    "editor_duration_seconds": round(time.time() - editor_started, 2),
                }
            except Exception as exc:
                logger.exception("citation verification failed")
                loop.errors.append(f"citation verification failed: {exc}")
                editor_stats = {
                    "editor_error": str(exc),
                    "editor_duration_seconds": round(time.time() - editor_started, 2),
                }

        # Cleanup the per-run collection unless the operator wants it kept.
        if not cfg.keep_collection:
            store.cleanup()

        duration = time.time() - started
        stats = {
            "pages_fetched": loop.pages_fetched,
            "excerpts_stored": len(excerpts),
            "sources": len(ordered_sources),
            "duration_seconds": round(duration, 2),
            "per_domain_pages": per_domain_stats(loop),
            "notes": loop.notes,
        }
        stats.update(editor_stats)
        report = ResearchReport(
            inquiry=request.inquiry,
            summary=summary,
            excerpts=excerpts,
            sources=ordered_sources,
            stats=stats,
            errors=loop.errors,
            citation_checks=citation_checks,
            source_statuses=source_statuses,
            original_summary=original_summary,
        )
        if cfg.write_document:
            try:
                report.document_path = _write_document(report, cfg.output_dir)
                report.stats["document_path"] = report.document_path
            except Exception as exc:
                logger.warning("failed to write web research document: %s", exc)
                report.errors.append(f"failed to write document: {exc}")
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(cm: ConfigManager) -> WebResearchConfig:
        try:
            raw = cm.get_config("web_research_config", "tools")
        except Exception:
            raw = None
        return WebResearchConfig.from_dict(raw)

    def _build_agent(self) -> Any:
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
                agent_name="web_researcher",
                system_prompt="",
            )

        if cfg.subagent_model:
            agent.default_model = cfg.subagent_model
        return agent

    def _build_editor_agent(self) -> Any:
        """Build the editor subagent for citation verification + rewrite.

        Falls back to a minimal :class:`OllamaAgent` if no
        ``editor_config_name`` agent config is registered. The system
        prompt is set per call by the editor module.
        """
        if self._editor_agent_factory is not None:
            return self._editor_agent_factory()
        if self._agent_factory is not None:
            return self._agent_factory()

        from ...agent import OllamaAgent

        cfg = self.config
        agent_cfg = None
        try:
            agent_cfg = self.config_manager.get_config(cfg.editor_config_name, "agents")
        except Exception as exc:
            logger.debug("failed to load editor agent config: %s", exc)

        if agent_cfg:
            agent = OllamaAgent.from_dict(agent_cfg, self.config_manager)
        else:
            model = cfg.editor_model or cfg.subagent_model or "llama3.2"
            agent = OllamaAgent(
                default_model=model,
                agent_name="editor",
                system_prompt="",
            )

        if cfg.editor_model:
            agent.default_model = cfg.editor_model
        return agent


def _make_collection_name(inquiry: str) -> str:
    """Build a unique collection name keyed off the inquiry and timestamp."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "_", inquiry.lower()).strip("_")[:32] or "inquiry"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"wr_{slug}_{ts}"


# ---------------------------------------------------------------------------
# Virtual-tool executor (mirrors FlowchartToolExecutor's shape)
# ---------------------------------------------------------------------------


class WebResearcherToolExecutor(ToolProvider):
    """Adapts :class:`WebResearcher` for use as a virtual ``web-research`` tool."""

    TOOL_NAME = "web-research"

    def __init__(
        self,
        config_manager: ConfigManager,
        researcher: Optional[WebResearcher] = None,
    ) -> None:
        self.config_manager = config_manager
        self._researcher = researcher

    @property
    def researcher(self) -> WebResearcher:
        if self._researcher is None:
            self._researcher = WebResearcher(self.config_manager)
        return self._researcher

    def discover(self, platform: str = "cross-platform") -> dict[str, ToolMetadata]:
        """Return the metadata entry for this virtual tool."""
        return {
            self.TOOL_NAME: ToolMetadata(
                name=self.TOOL_NAME,
                path="",
                description=(
                    "Subagent-controlled web research over a whitelisted set of "
                    "domains. Usage: web-research <inquiry text>"
                ),
                platform=platform,
                source="virtual",
                tool_type="web_research",
            )
        }

    def can_execute(self, tool_name: str) -> bool:
        """Return True for the ``web-research`` tool name."""
        return tool_name == self.TOOL_NAME

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the web-research tool call extracted from *command*.

        Strips the leading ``web-research`` token and treats the rest as the
        inquiry string.

        Args:
            command: Full command string, e.g. ``"web-research how does X work"``.
            context: Unused; accepted for interface consistency.
        """
        parts = command.strip().split(None, 1)
        inquiry = parts[1].strip() if len(parts) > 1 else ""
        return self.run(inquiry)

    def run(self, inquiry: str) -> ToolResult:
        """Execute a research pass and wrap the report as a :class:`ToolResult`."""
        start = time.time()
        command = f"web-research {inquiry}"
        if not inquiry or not inquiry.strip():
            return ToolResult(
                success=False,
                stdout="",
                stderr="empty inquiry",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint="Usage: web-research <inquiry>",
            )
        try:
            report = self.researcher.research(inquiry)
        except Exception as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"web research failed: {exc}",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint="Check web_research_config.yaml and that the 'web' extra is installed.",
            )
        return ToolResult(
            success=True,
            stdout=report.to_markdown(),
            stderr="\n".join(report.errors),
            exit_code=0,
            execution_time=time.time() - start,
            command=command,
            report_paths=[report.document_path] if report.document_path else [],
        )
