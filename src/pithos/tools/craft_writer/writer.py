"""Top-level :class:`CraftWriter` facade + virtual-tool executor wrapper."""

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
from .models import (
    CraftStory,
    CraftWriteConfig,
    CraftWriteRequest,
    StoryOutline,
    StorySection,
)
from .notes import format_notes_for_prompt, retrieve_craft_notes
from .prompts import (
    build_outline_user_prompt,
    build_revision_user_prompt,
    build_section_user_prompt,
    outline_system_prompt,
    parse_outline,
    revision_system_prompt,
    section_system_prompt,
)

logger = logging.getLogger(__name__)

#: Craft dimensions most relevant to each pipeline stage. Used to narrow the
#: (single) retrieved note set down to what's useful for a given prompt;
#: falls back to the full retrieved set when none of these were retrieved.
_OUTLINE_DIMENSIONS = ("characterization", "themes", "plot_structure_and_pacing")
_DRAFT_DIMENSIONS = ("scene_construction", "dialogue", "prose_style_and_voice")
_REVISION_DIMENSIONS = ("prose_style_and_voice",)


def _subset(
    notes_by_dim: dict[str, list[Any]], dims: tuple[str, ...]
) -> dict[str, list[Any]]:
    subset = {d: notes_by_dim[d] for d in dims if notes_by_dim.get(d)}
    return subset or notes_by_dim


def _default_num_sections(target_words: int) -> int:
    return max(1, round(target_words / 400))


def _truncate_story_so_far(text: str, char_cap: int) -> str:
    if len(text) <= char_cap:
        return text
    return text[-char_cap:]


class CraftWriter:
    """High-level entry point: retrieve notes → outline → draft → revise.

    Construction is cheap; heavy resources (subagent, knowledge base) are
    created per :meth:`write` call so concurrent runs stay isolated.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        config: Optional[CraftWriteConfig] = None,
        agent_factory: Optional[Any] = None,
        memory_store: Optional[Any] = None,
    ) -> None:
        """Initialise the writer.

        Args:
            config_manager: ConfigManager used to load tool/agent configs.
            config: Optional pre-built config; otherwise loaded from
                ``configs/tools/craft_writing_config.yaml``.
            agent_factory: Optional callable returning the craft-writer
                subagent. When None the subagent is built from the registered
                agent config named by
                :attr:`CraftWriteConfig.subagent_config_name`.
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

    def write(self, request_or_direction: Any, **kwargs: Any) -> CraftStory:
        """Run the craft-guided writing pipeline and return a :class:`CraftStory`.

        Accepts either a :class:`CraftWriteRequest` or a plain string, which
        is treated as the story direction for ergonomics.

        Raises:
            ValueError: If no direction is given.
        """
        if isinstance(request_or_direction, CraftWriteRequest):
            request = request_or_direction
        else:
            request = CraftWriteRequest(direction=str(request_or_direction), **kwargs)

        if not request.direction or not request.direction.strip():
            raise ValueError("direction is required")

        cfg = self.config
        started = time.time()
        errors: list[str] = []

        dimensions = request.dimensions_override or list(cfg.dimensions)
        target_words = request.target_word_count or cfg.target_word_count
        num_sections = (
            request.num_sections
            or cfg.num_sections
            or _default_num_sections(target_words)
        )
        if request.revise is False:
            revision_passes = 0
        elif request.revise is True:
            revision_passes = max(cfg.revision_passes, 1)
        else:
            revision_passes = cfg.revision_passes

        memory_store = self._get_memory_store()
        notes_by_dim = retrieve_craft_notes(
            memory_store,
            dimensions=dimensions,
            query=request.direction,
            per_dimension=cfg.notes_per_dimension,
            min_relevance=cfg.min_relevance,
            source_title=request.source_title,
            category=cfg.note_category,
        )

        agent = self._build_agent()

        outline = self._plan_outline(
            agent, request, num_sections, notes_by_dim, cfg, errors
        )
        title = request.title or outline.title or "Untitled"

        sections, full_text = self._draft_sections(
            agent, title, outline, notes_by_dim, cfg, target_words, errors
        )

        if revision_passes > 0 and full_text.strip():
            full_text = self._revise(
                agent, title, full_text, notes_by_dim, cfg, revision_passes, errors
            )

        story = CraftStory(
            title=title,
            premise=outline.premise,
            outline=outline,
            sections=sections,
            full_text=full_text,
            notes_used={
                d: [getattr(h, "content", h) for h in hits]
                for d, hits in notes_by_dim.items()
                if hits
            },
            stats={
                "sections_drafted": len(sections),
                "revision_passes": revision_passes if full_text.strip() else 0,
                "notes_retrieved": sum(len(v) for v in notes_by_dim.values()),
                "duration_seconds": round(time.time() - started, 2),
            },
            errors=errors,
        )

        if cfg.store_story:
            self._store_story(memory_store, story, request)

        if cfg.write_document:
            try:
                document_path = _write_document(story, cfg.output_dir)
                story.document_path = document_path
                story.stats["document_path"] = document_path
            except Exception as exc:
                errors.append(f"failed to write document: {exc}")

        return story

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _plan_outline(
        self,
        agent: Any,
        request: CraftWriteRequest,
        num_sections: int,
        notes_by_dim: dict[str, list[Any]],
        cfg: CraftWriteConfig,
        errors: list[str],
    ) -> StoryOutline:
        """Stage 1: ask the subagent to plan a title/premise/section outline."""
        try:
            agent.set_system_prompt(outline_system_prompt(num_sections))
        except Exception:
            pass
        notes_text = format_notes_for_prompt(_subset(notes_by_dim, _OUTLINE_DIMENSIONS))
        prompt = build_outline_user_prompt(
            direction=request.direction,
            genre=request.genre or "",
            tone=request.tone or "",
            notes_text=notes_text,
        )
        try:
            reply = _send(agent, prompt, cfg.subagent_model)
        except Exception as exc:
            logger.warning("outline generation failed: %s", exc)
            errors.append(f"outline generation failed: {exc}")
            return StoryOutline(
                title=request.title or "Untitled", premise="", sections=[]
            )

        outline = parse_outline(reply)
        if not outline.sections:
            # Fall back to a single section spanning the whole story so
            # drafting can still proceed.
            outline.sections = [
                StorySection(
                    heading="Story", summary=outline.premise or request.direction
                )
            ]
        return outline

    def _draft_sections(
        self,
        agent: Any,
        title: str,
        outline: StoryOutline,
        notes_by_dim: dict[str, list[Any]],
        cfg: CraftWriteConfig,
        target_words: int,
        errors: list[str],
    ) -> tuple[list[StorySection], str]:
        """Stage 2: draft each outlined section in turn, keeping continuity."""
        section_words = max(50, round(target_words / max(1, len(outline.sections))))
        try:
            agent.set_system_prompt(section_system_prompt(section_words))
        except Exception:
            pass
        notes_text = format_notes_for_prompt(_subset(notes_by_dim, _DRAFT_DIMENSIONS))

        sections: list[StorySection] = []
        story_so_far = ""
        for planned in outline.sections:
            prompt = build_section_user_prompt(
                title=title,
                premise=outline.premise,
                section=planned,
                story_so_far=_truncate_story_so_far(
                    story_so_far, cfg.story_context_char_cap
                ),
                notes_text=notes_text,
            )
            try:
                text = (_send(agent, prompt, cfg.subagent_model) or "").strip()
            except Exception as exc:
                logger.warning("section '%s' draft failed: %s", planned.heading, exc)
                errors.append(f"section '{planned.heading}' draft failed: {exc}")
                text = ""
            sections.append(
                StorySection(
                    heading=planned.heading, summary=planned.summary, text=text
                )
            )
            if text:
                story_so_far = (
                    f"{story_so_far}\n\n{text}".strip() if story_so_far else text
                )
        return sections, story_so_far

    def _revise(
        self,
        agent: Any,
        title: str,
        draft_text: str,
        notes_by_dim: dict[str, list[Any]],
        cfg: CraftWriteConfig,
        revision_passes: int,
        errors: list[str],
    ) -> str:
        """Stage 3: run one or more whole-draft revision passes."""
        try:
            agent.set_system_prompt(revision_system_prompt())
        except Exception:
            pass
        notes_text = format_notes_for_prompt(
            _subset(notes_by_dim, _REVISION_DIMENSIONS)
        )

        text = draft_text
        for _ in range(revision_passes):
            prompt = build_revision_user_prompt(
                title=title, draft_text=text, notes_text=notes_text
            )
            try:
                revised = (_send(agent, prompt, cfg.subagent_model) or "").strip()
            except Exception as exc:
                logger.warning("revision pass failed: %s", exc)
                errors.append(f"revision pass failed: {exc}")
                break
            if revised:
                text = revised
        return text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(cm: ConfigManager) -> CraftWriteConfig:
        try:
            raw = cm.get_config("craft_writing_config", "tools")
        except Exception:
            raw = None
        return CraftWriteConfig.from_dict(raw)

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

    def _store_story(
        self,
        memory_store: Optional[Any],
        story: CraftStory,
        request: CraftWriteRequest,
    ) -> None:
        """Persist the generated story to the knowledge base."""
        if memory_store is None:
            return
        try:
            entry_id = memory_store.store(
                self.config.story_category,
                story.full_text or story.to_markdown(),
                metadata={
                    "title": story.title,
                    "direction": request.direction,
                    "genre": request.genre or "",
                    "tone": request.tone or "",
                    "source_title": request.source_title or "",
                    "kind": "craft_story",
                },
            )
            story.stats["entry_id"] = entry_id
        except Exception as exc:
            logger.warning("failed to store story in knowledge base: %s", exc)

    def _build_agent(self) -> Any:
        """Build the craft-writer subagent."""
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
                agent_name="craft_writer",
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


def _write_document(story: CraftStory, output_dir: str) -> str:
    """Write the story markdown to ``output_dir`` and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"story_{_slugify(story.title)}_{ts}.md"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(story.to_markdown())
    return path


# ---------------------------------------------------------------------------
# Virtual-tool executor (mirrors CraftAnalyzerToolExecutor's shape)
# ---------------------------------------------------------------------------


class CraftWriterToolExecutor(ToolProvider):
    """Adapts :class:`CraftWriter` for use as a virtual ``craft-write`` tool."""

    TOOL_NAME = "craft-write"

    def __init__(
        self,
        config_manager: ConfigManager,
        writer: Optional[CraftWriter] = None,
    ) -> None:
        self.config_manager = config_manager
        self._writer = writer

    @property
    def writer(self) -> CraftWriter:
        if self._writer is None:
            self._writer = CraftWriter(self.config_manager)
        return self._writer

    def discover(self, platform: str = "cross-platform") -> dict[str, ToolMetadata]:
        """Return the metadata entry for this virtual tool."""
        return {
            self.TOOL_NAME: ToolMetadata(
                name=self.TOOL_NAME,
                path="",
                description=(
                    "Write a short story guided by previously analyzed craft notes "
                    "(from craft-notes). Usage: craft-write <direction text>"
                ),
                platform=platform,
                source="virtual",
                tool_type="craft_writing",
            )
        }

    def can_execute(self, tool_name: str) -> bool:
        """Return True for the ``craft-write`` tool name."""
        return tool_name == self.TOOL_NAME

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the craft-write tool call extracted from *command*.

        Strips the leading ``craft-write`` token; the remainder is treated
        as the freeform story direction.
        """
        parts = command.strip().split(None, 1)
        direction = parts[1].strip() if len(parts) > 1 else ""
        return self.run(CraftWriteRequest(direction=direction))

    def run(self, request: Any) -> ToolResult:
        """Run the writing pipeline and wrap the story as a ToolResult."""
        started = time.time()
        try:
            story = self.writer.write(request)
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
            stdout=story.to_markdown(),
            stderr="",
            exit_code=0,
            execution_time=round(time.time() - started, 2),
            command=self.TOOL_NAME,
        )
