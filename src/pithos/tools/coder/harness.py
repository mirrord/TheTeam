"""The coding harness: a single-agent query -> act -> observe loop.

:class:`CodeHarness` drives one coding subagent through a bounded loop. Each
step the subagent produces a reply; the harness parses it into actions
(:mod:`pithos.tools.coder.actions`), executes them against a workspace-jailed
:class:`~pithos.tools.coder.environment.WorkspaceEnvironment`, and feeds a
truncated observation back. The loop ends when the agent submits a result or a
limit (steps / wall-clock / consecutive format errors) is reached.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ...config_manager import ConfigManager
from ..registry import ToolRegistry
from .actions import parse_actions
from .environment import WorkspaceEnvironment
from .models import (
    CodeHarnessConfig,
    CodeTaskRequest,
    CodeTaskResult,
    ExitStatus,
    StepRecord,
)
from .prompts import (
    build_instance_prompt,
    build_observation,
    format_error_prompt,
    system_prompt,
)

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:40]
    return slug or "task"


class CodeHarness:
    """Run a single coding task through the agentic loop."""

    def __init__(
        self,
        config_manager: ConfigManager,
        config: Optional[CodeHarnessConfig] = None,
        agent_factory: Optional[Any] = None,
        tool_config: Optional[dict] = None,
        confirm_callback: Optional[Any] = None,
    ) -> None:
        self.config_manager = config_manager
        self.config = config or self._load_config(config_manager)
        self._agent_factory = agent_factory
        # Optional overrides supplied by the calling agent so the coding
        # subagent inherits the caller's tool policy (allow/deny/confirm) and
        # user-confirmation channel. When ``tool_config`` is None the harness
        # falls back to loading ``tool_config.yaml`` itself.
        self._tool_config_override = tool_config
        self._confirm_callback = confirm_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, request_or_task, **kwargs: Any) -> CodeTaskResult:
        """Run a coding task and return a :class:`CodeTaskResult`.

        Accepts either a :class:`CodeTaskRequest` or a plain task string.
        """
        if isinstance(request_or_task, CodeTaskRequest):
            request = request_or_task
        else:
            request = CodeTaskRequest(task=str(request_or_task), **kwargs)

        result = CodeTaskResult(task=request.task)
        if not request.task or not request.task.strip():
            result.exit_status = ExitStatus.ERROR
            result.errors.append("empty task")
            return result

        root = self._resolve_root(request)
        tool_config = self._tool_config()
        env = WorkspaceEnvironment(
            root,
            self.config,
            tool_config,
            confirm_callback=self._confirm_callback,
        )
        result.root = str(env.root)

        try:
            agent = self._build_agent()
        except Exception as exc:  # pragma: no cover - construction guard
            result.exit_status = ExitStatus.ERROR
            result.errors.append(f"failed to build subagent: {exc}")
            return result

        try:
            agent.set_system_prompt(system_prompt())
        except Exception:
            pass

        file_listing = ""
        if self.config.include_file_listing:
            try:
                file_listing = env.list_files(self.config.max_listed_files)
            except Exception as exc:
                logger.debug("file listing failed: %s", exc)

        prompt = build_instance_prompt(
            request.task,
            root=str(env.root),
            system_info=env.system_info(),
            file_listing=file_listing,
            test_command=request.test_command,
            tools_summary=env.tools_summary(),
        )

        steps: list[StepRecord] = []
        self._loop(agent, env, prompt, steps, result)

        result.steps = len(steps)
        try:
            result.diff = env.get_diff()
        except Exception as exc:
            logger.debug("diff capture failed: %s", exc)

        if (
            self.config.auto_commit
            and result.exit_status == ExitStatus.SUBMITTED
            and self.config.allow_git
        ):
            self._auto_commit(env, request.task)

        result.messages = [self._step_to_dict(s) for s in steps]
        if self.config.write_trajectory:
            result.trajectory_path = self._write_trajectory(
                request, result, steps, str(env.root)
            )
        return result

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _loop(
        self,
        agent: Any,
        env: WorkspaceEnvironment,
        first_prompt: str,
        steps: list[StepRecord],
        result: CodeTaskResult,
    ) -> None:
        start = time.time()
        consecutive_format_errors = 0
        next_prompt = first_prompt
        step_index = 0

        while True:
            if self.config.step_limit and step_index >= self.config.step_limit:
                result.exit_status = ExitStatus.STEP_LIMIT
                result.summary = result.summary or self._fallback_summary(steps)
                return
            if (
                self.config.wall_time_limit_seconds
                and (time.time() - start) >= self.config.wall_time_limit_seconds
            ):
                result.exit_status = ExitStatus.TIME_LIMIT
                result.summary = result.summary or self._fallback_summary(steps)
                return

            record = StepRecord(index=step_index)
            steps.append(record)
            step_index += 1

            try:
                reply = agent.send(next_prompt) or ""
            except Exception as exc:
                result.exit_status = ExitStatus.ERROR
                result.errors.append(f"subagent generation failed: {exc}")
                return
            record.reply = reply

            actions = parse_actions(reply)
            if not actions:
                record.format_error = True
                consecutive_format_errors += 1
                limit = self.config.max_consecutive_format_errors
                if limit and consecutive_format_errors >= limit:
                    result.exit_status = ExitStatus.FORMAT_ERROR_LIMIT
                    result.errors.append(
                        "stopped after repeated replies with no valid action"
                    )
                    result.summary = result.summary or self._fallback_summary(steps)
                    return
                next_prompt = format_error_prompt(
                    "no run, edit, or submit tool call was found",
                    consecutive_format_errors,
                    limit,
                )
                continue

            consecutive_format_errors = 0
            observations: list[str] = []
            submitted = False
            for action in actions:
                record.actions.append(action.raw.splitlines()[0] if action.raw else "")
                if action.kind == "submit":
                    result.exit_status = ExitStatus.SUBMITTED
                    result.summary = action.summary or self._fallback_summary(steps)
                    submitted = True
                    break
                obs = env.execute_action(action)
                observations.append(obs)
                record.observations.append(obs)

            if submitted:
                return

            if not observations:
                next_prompt = format_error_prompt(
                    "no executable action was produced", 0, 0
                )
                continue

            next_prompt = build_observation(observations)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_summary(steps: list[StepRecord]) -> str:
        for record in reversed(steps):
            if record.reply.strip():
                return record.reply.strip()
        return "No summary was produced."

    def _auto_commit(self, env: WorkspaceEnvironment, task: str) -> None:
        env.execute_action(_RunAction("git add -A"))
        message = f"auto: {task.strip()[:72]}"
        env.execute_action(_RunAction(f'git commit -m "{message}"'))

    def _resolve_root(self, request: CodeTaskRequest) -> str:
        """Resolve the workspace root for this run.

        Resolution order:
        1. an explicit ``--root`` / ``request.root`` (used as-is);
        2. a configured ``allowed_root`` (used as-is);
        3. otherwise a brand-new unique directory allocated under
           ``workspace_base`` (default ``~/.talos/code``), so every initial
           invocation of the tool gets its own isolated workspace.
        """
        if request.root:
            return request.root
        if self.config.allowed_root:
            return self.config.allowed_root
        return self._allocate_workspace()

    def _allocate_workspace(self) -> str:
        """Create and return a new unique workspace directory under ``workspace_base``."""
        base = Path(self.config.workspace_base).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S_")
        return tempfile.mkdtemp(prefix=prefix, dir=str(base))

    def _tool_config(self) -> dict:
        # Prefer the caller-supplied tool config so the coding subagent has the
        # same tool access (allow/deny/confirm policy) as its caller.
        if self._tool_config_override is not None:
            return dict(self._tool_config_override)
        try:
            tmp = ToolRegistry(self.config_manager, providers=[])
            return dict(tmp.config)
        except Exception as exc:
            logger.debug("tool config load failed: %s", exc)
            return {}

    @staticmethod
    def _load_config(cm: ConfigManager) -> CodeHarnessConfig:
        raw = None
        try:
            raw = cm.get_config("code_config", "tools")
        except Exception:
            raw = None
        return CodeHarnessConfig.from_dict(raw)

    def _build_agent(self) -> Any:
        if self._agent_factory is not None:
            return self._agent_factory()

        from ...agent import build_agent, build_agent_from_config

        cfg = self.config
        agent_cfg = None
        try:
            agent_cfg = self.config_manager.get_config(cfg.subagent_config_name, "agents")
        except Exception as exc:
            logger.debug("failed to load subagent config: %s", exc)

        if agent_cfg:
            agent = build_agent_from_config(agent_cfg, self.config_manager)
        else:
            model = cfg.subagent_model or "llama3.2"
            agent = build_agent(
                model=model,
                provider=cfg.subagent_provider,
                agent_name="code_harness",
                system_prompt="",
            )
        if cfg.subagent_model:
            agent.default_model = cfg.subagent_model
        if cfg.subagent_timeout and cfg.subagent_timeout > 0:
            agent.generation_timeout = cfg.subagent_timeout
        return agent

    @staticmethod
    def _step_to_dict(step: StepRecord) -> dict[str, Any]:
        return {
            "index": step.index,
            "reply": step.reply,
            "actions": step.actions,
            "observations": step.observations,
            "format_error": step.format_error,
        }

    def _write_trajectory(
        self,
        request: CodeTaskRequest,
        result: CodeTaskResult,
        steps: list[StepRecord],
        root: str,
    ) -> Optional[str]:
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self.config.output_dir, f"code_{_slugify(request.task)}_{ts}.json"
            )
            data = {
                "task": request.task,
                "root": root,
                "exit_status": result.exit_status.value,
                "summary": result.summary,
                "diff": result.diff,
                "steps": [self._step_to_dict(s) for s in steps],
                "errors": result.errors,
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            return path
        except Exception as exc:
            logger.debug("trajectory write failed: %s", exc)
            return None


class _RunAction:
    """Lightweight stand-in for an :class:`~pithos.tools.coder.actions.Action`.

    Used internally for harness-issued commands (e.g. the auto-commit) so we can
    reuse :meth:`WorkspaceEnvironment.execute_action` without constructing a full
    parsed action.
    """

    kind = "run"

    def __init__(self, command: str) -> None:
        self.command = command
        self.raw = command
