"""Top-level :class:`NewsResearcher` facade + virtual-tool executor wrapper."""

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
from ..web_researcher.fetcher import Fetcher
from ..web_researcher.search import DuckDuckGoSearch
from .assessor import assess_articles
from .models import NewsReport, NewsResearchConfig, NewsResearchRequest
from .models import ArticleAssessment, NewsArticle
from .ranking import rank_articles
from .scraper import NewsScraper
from .terms import extract_terms

logger = logging.getLogger(__name__)


class NewsResearcher:
    """High-level entry point: term extraction → scrape → summarise/judge.

    Construction is cheap; heavy resources (fetcher, subagent, knowledge
    base) are created per :meth:`research` call so concurrent runs stay
    isolated.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        config: Optional[NewsResearchConfig] = None,
        agent_factory: Optional[Any] = None,
        term_agent_factory: Optional[Any] = None,
        memory_store: Optional[Any] = None,
    ) -> None:
        """Initialise the researcher.

        Args:
            config_manager: ConfigManager used to load tool/agent configs.
            config: Optional pre-built config; otherwise loaded from
                ``configs/tools/news_research_config.yaml``.
            agent_factory: Optional callable returning the summarise/relevance
                subagent. When None the subagent is built from the registered
                agent config named by
                :attr:`NewsResearchConfig.subagent_config_name`.
            term_agent_factory: Optional callable returning the small
                term-extraction agent. Falls back to ``agent_factory`` then a
                minimal agent built from :attr:`NewsResearchConfig.term_model`.
            memory_store: Optional pre-built knowledge-base store (mainly for
                tests). When None a :class:`MemoryStore` is created lazily.
        """
        self.config_manager = config_manager
        self.config = config or self._load_config(config_manager)
        self._agent_factory = agent_factory
        self._term_agent_factory = term_agent_factory
        self._memory_store = memory_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, request_or_inquiry, **kwargs: Any) -> NewsReport:
        """Run a news research pass and return a :class:`NewsReport`.

        Accepts either a :class:`NewsResearchRequest` or a plain inquiry
        string for ergonomics.
        """
        if isinstance(request_or_inquiry, NewsResearchRequest):
            request = request_or_inquiry
        else:
            request = NewsResearchRequest(inquiry=str(request_or_inquiry), **kwargs)

        if not request.inquiry or not request.inquiry.strip():
            raise ValueError("inquiry cannot be empty")

        cfg = self.config
        domains = request.domains_override or list(cfg.domains)
        feeds = (
            request.feeds_override
            if request.feeds_override is not None
            else list(cfg.feeds)
        )
        recency_days = (
            request.recency_days_override
            if request.recency_days_override is not None
            else cfg.recency_days
        )

        if not domains and not feeds:
            return NewsReport(
                inquiry=request.inquiry,
                terms=[],
                assessments=[],
                errors=["no whitelisted domains or feeds configured"],
                stats={},
            )

        run_cfg = NewsResearchConfig(
            **{
                **cfg.__dict__,
                "domains": domains,
                "feeds": feeds,
                "recency_days": recency_days,
            }
        )

        started = time.time()
        errors: list[str] = []

        # Step 1: derive technical search terms with the small model.
        term_agent = self._build_term_agent()
        terms = extract_terms(
            request.inquiry,
            term_agent,
            max_terms=run_cfg.max_terms,
            model=run_cfg.term_model,
        )

        # Shared HTTP + search stack.
        fetcher = Fetcher(
            whitelist=domains or [],
            user_agent=run_cfg.user_agent,
            timeout=run_cfg.request_timeout,
            max_bytes=run_cfg.max_page_bytes,
            per_domain_rps=run_cfg.per_domain_rps,
            respect_robots=run_cfg.respect_robots,
        )
        search = (
            DuckDuckGoSearch(
                fetcher=fetcher,
                results_per_domain=run_cfg.search_results_per_domain,
            )
            if run_cfg.search_fallback
            else None
        )
        memory_store = self._get_memory_store()

        # Step 2: gather + download recent articles into the knowledge base.
        scraper = NewsScraper(
            config=run_cfg,
            fetcher=fetcher,
            search=search,
            memory_store=memory_store,
        )
        try:
            articles = scraper.gather(request.inquiry, terms)
        except Exception as exc:
            logger.exception("news scraper crashed")
            errors.append(f"scraper crashed: {exc}")
            articles = []
        errors.extend(scraper.errors)

        # Pre-filter: only spend the (expensive) LLM assessment on the top-K
        # most relevant articles by a cheap term-overlap score. The remainder
        # are recorded as reviewed-but-not-assessed for transparency.
        to_assess = articles
        skipped: list[NewsArticle] = []
        top_k = run_cfg.prefilter_top_k
        if top_k and len(articles) > top_k:
            ranked = rank_articles(request.inquiry, terms, articles)
            to_assess = ranked[:top_k]
            skipped = ranked[top_k:]

        # Steps 3-5: summarise + judge each retained article.
        subagent = self._build_agent()
        assessments = assess_articles(
            inquiry=request.inquiry,
            articles=to_assess,
            agent=subagent,
            config=run_cfg,
            memory_store=memory_store,
            errors=errors,
            agent_factory=self._build_agent,
        )
        assessments.extend(_skipped_assessments(skipped))

        # Step 6: write the collected document.
        document_path = None
        relevant = [a for a in assessments if a.relevant]
        stats = {
            "terms": terms,
            "articles_downloaded": len(articles),
            "articles_assessed": len(to_assess),
            "articles_relevant": len(relevant),
            "recency_days": recency_days,
            "duration_seconds": round(time.time() - started, 2),
        }
        report = NewsReport(
            inquiry=request.inquiry,
            terms=terms,
            assessments=assessments,
            stats=stats,
            errors=errors,
        )
        if run_cfg.write_document:
            try:
                document_path = _write_document(report, run_cfg.output_dir)
                report.document_path = document_path
                report.stats["document_path"] = document_path
            except Exception as exc:
                errors.append(f"failed to write document: {exc}")
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(cm: ConfigManager) -> NewsResearchConfig:
        try:
            raw = cm.get_config("news_research_config", "tools")
        except Exception:
            raw = None
        return NewsResearchConfig.from_dict(raw)

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

    def _build_agent(self) -> Any:
        """Build the summarise/relevance subagent."""
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
                agent_name="news_researcher",
                system_prompt="",
            )
        if cfg.subagent_model:
            agent.default_model = cfg.subagent_model
        return agent

    def _build_term_agent(self) -> Any:
        """Build the small term-extraction agent."""
        if self._term_agent_factory is not None:
            return self._term_agent_factory()

        cfg = self.config
        # Prefer a dedicated term agent config when one is registered.
        if cfg.term_config_name:
            from ...agent import OllamaAgent

            try:
                agent_cfg = self.config_manager.get_config(
                    cfg.term_config_name, "agents"
                )
            except Exception:
                agent_cfg = None
            if agent_cfg:
                agent = OllamaAgent.from_dict(agent_cfg, self.config_manager)
                if cfg.term_model:
                    agent.default_model = cfg.term_model
                return agent

        # Otherwise reuse the subagent (its model can be overridden by
        # ``term_model`` at the call site).
        if self._agent_factory is not None:
            return self._agent_factory()

        from ...agent import OllamaAgent

        model = cfg.term_model or cfg.subagent_model or "llama3.2"
        return OllamaAgent(
            default_model=model,
            agent_name="news_terms",
            system_prompt="",
        )


def _skipped_assessments(articles: list[NewsArticle]) -> list[ArticleAssessment]:
    """Record pre-filtered articles as reviewed-but-not-assessed (non-relevant)."""
    return [
        ArticleAssessment(
            url=a.url,
            title=a.title,
            summary="",
            relevant=False,
            reason="not assessed (below pre-filter threshold)",
            published_iso=a.published_iso,
            source_host=a.source_host,
            article_entry_id=a.metadata.get("article_entry_id"),
        )
        for a in articles
    ]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:40]
    return slug or "inquiry"


def _write_document(report: NewsReport, output_dir: str) -> str:
    """Write the report markdown to ``output_dir`` and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"news_{_slugify(report.inquiry)}_{ts}.md"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report.to_markdown())
    return path


# ---------------------------------------------------------------------------
# Virtual-tool executor (mirrors WebResearcherToolExecutor's shape)
# ---------------------------------------------------------------------------


class NewsResearcherToolExecutor(ToolProvider):
    """Adapts :class:`NewsResearcher` for use as a virtual ``research-news`` tool."""

    TOOL_NAME = "research-news"

    def __init__(
        self,
        config_manager: ConfigManager,
        researcher: Optional[NewsResearcher] = None,
    ) -> None:
        self.config_manager = config_manager
        self._researcher = researcher

    @property
    def researcher(self) -> NewsResearcher:
        if self._researcher is None:
            self._researcher = NewsResearcher(self.config_manager)
        return self._researcher

    def discover(self, platform: str = "cross-platform") -> dict[str, ToolMetadata]:
        """Return the metadata entry for this virtual tool."""
        return {
            self.TOOL_NAME: ToolMetadata(
                name=self.TOOL_NAME,
                path="",
                description=(
                    "Collect and summarise recent news articles from whitelisted "
                    "domains relevant to an inquiry. Usage: research-news <inquiry text>"
                ),
                platform=platform,
                source="virtual",
                tool_type="news_research",
            )
        }

    def can_execute(self, tool_name: str) -> bool:
        """Return True for the ``research-news`` tool name."""
        return tool_name == self.TOOL_NAME

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the research-news tool call extracted from *command*.

        Strips the leading ``research-news`` token and treats the rest as the
        inquiry string.
        """
        parts = command.strip().split(None, 1)
        inquiry = parts[1].strip() if len(parts) > 1 else ""
        return self.run(inquiry)

    def run(self, inquiry: str) -> ToolResult:
        """Execute a news research pass and wrap the report as a ToolResult."""
        start = time.time()
        command = f"research-news {inquiry}"
        if not inquiry or not inquiry.strip():
            return ToolResult(
                success=False,
                stdout="",
                stderr="empty inquiry",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint="Usage: research-news <inquiry>",
            )
        try:
            report = self.researcher.research(inquiry)
        except Exception as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"news research failed: {exc}",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint=(
                    "Check news_research_config.yaml and that the 'web' extra is "
                    "installed."
                ),
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
