"""Agent Team Manager — coordinates multiple agents working together.

The :class:`AgentTeam` orchestrates a coordinator agent plus N worker agents.
Coordination follows a structured pattern:

1. **Plan** — :meth:`breakdown_task` asks the coordinator to emit a JSON plan
   that maps each worker agent name to a concrete subtask.
2. **Iterate** — :meth:`iterate` runs ``max_rounds`` rounds. Each round each
   worker receives its subtask plus a notes view (shared by default, or
   per-agent when ``parallel_notes`` is enabled) and appends a response to
   the team :class:`TeamLedger`.
3. **Check** — when ``completion_check`` is enabled the coordinator is asked
   after each round to mark per-agent completion (also JSON). Iteration
   stops early when every worker is marked complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Optional

from ..agent import OllamaAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NoteEntry:
    """A single note appended to the team ledger by a worker agent."""

    agent: str
    round: int
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TeamLedger:
    """Mutable record of a team's work on a task.

    Attributes:
        subtasks: Map of worker agent name → assigned subtask description.
        notes: Ordered list of agent responses across rounds.
        completed: Map of worker agent name → completion flag.
        round: Current iteration round (0 before any iteration).
    """

    subtasks: dict[str, str] = field(default_factory=dict)
    notes: list[NoteEntry] = field(default_factory=list)
    completed: dict[str, bool] = field(default_factory=dict)
    round: int = 0

    def all_complete(self) -> bool:
        """Return True iff every assigned worker is marked complete."""
        return bool(self.completed) and all(self.completed.values())

    def notes_for(self, agent: str, parallel: bool) -> list[NoteEntry]:
        """Return the notes view a given worker should see.

        Args:
            agent: Worker agent name.
            parallel: If True, return only this worker's own prior notes;
                otherwise return all notes (shared view).
        """
        if parallel:
            return [n for n in self.notes if n.agent == agent]
        return list(self.notes)


@dataclass
class TeamContext:
    """Workspace-scoped context for a team of agents working together."""

    team_task: str
    workspace: str
    started: bool = False
    parallel_notes: bool = False
    ledger: Optional[TeamLedger] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from free-form text.

    The coordinator may wrap its JSON in prose or markdown fences. We strip
    common fences and then locate the outermost ``{...}`` block.

    Returns the decoded dict, or ``None`` if no valid JSON object is found.
    """
    if not text:
        return None
    cleaned = text.strip()
    # Strip common markdown fences.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            first, rest = cleaned.split("\n", 1)
            if first.strip().lower() in {"json", ""}:
                cleaned = rest
    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


# ---------------------------------------------------------------------------
# AgentTeam
# ---------------------------------------------------------------------------


class AgentTeam:
    """Coordinates multiple agents working together on tasks."""

    DEFAULT_MAX_ROUNDS = 3

    def __init__(
        self,
        coordinator_model: str,
        init_context: str = "DEFAULT",
        team_task: Optional[str] = None,
    ) -> None:
        """Initialize the agent team.

        Args:
            coordinator_model: Model name for the coordinator agent.
            init_context: Initial context name.
            team_task: Optional initial team task.

        Raises:
            ValueError: If coordinator_model or init_context is empty.
        """
        if not coordinator_model or not coordinator_model.strip():
            raise ValueError("coordinator_model cannot be empty")
        if not init_context or not init_context.strip():
            raise ValueError("init_context cannot be empty")

        self.agents: dict[str, OllamaAgent] = {}
        self.init_coordinator(coordinator_model)
        self.workspaces: dict[str, TeamContext] = {
            init_context: TeamContext(team_task or "", "")
        }
        self.current_team_context: str = init_context

    # ------------------------------------------------------------------
    # Roster management
    # ------------------------------------------------------------------

    def init_coordinator(self, model_name: str) -> None:
        """Initialize the coordinator agent."""
        self.agents["coordinator"] = OllamaAgent(default_model=model_name)
        self.agents["coordinator"].create_context(
            "DEFAULT",
            "You are a project manager and team coordinator. You have been "
            "tasked with managing a team of agents to complete a project. "
            "When asked for plans or status checks, respond with valid JSON "
            "matching the schema in the prompt. Do not include prose outside "
            "the JSON object.",
        )

    def add_agent(self, agent_name: str, model_name: str) -> None:
        """Add a new agent to the team.

        Raises:
            ValueError: If agent with same name already exists or names are empty.
        """
        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name cannot be empty")
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if agent_name == "coordinator":
            raise ValueError("'coordinator' is reserved for the team coordinator")
        if agent_name in self.agents:
            raise ValueError(f"Agent '{agent_name}' already exists.")
        self.agents[agent_name] = OllamaAgent(default_model=model_name)

    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the team.

        Raises:
            ValueError: If agent doesn't exist or is the coordinator.
        """
        if agent_name == "coordinator":
            raise ValueError("Cannot remove the coordinator agent")
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' does not exist.")
        del self.agents[agent_name]

    def worker_names(self) -> list[str]:
        """Return the list of worker agent names (excludes the coordinator)."""
        return [name for name in self.agents if name != "coordinator"]

    # ------------------------------------------------------------------
    # Workspace / context helpers
    # ------------------------------------------------------------------

    def set_shared_workspace(
        self, workspace: str, context_name: Optional[str] = None
    ) -> None:
        """Set shared workspace for team context."""
        self.current_team_context = (
            context_name if context_name else self.current_team_context
        )
        self.workspaces[self.current_team_context].workspace = workspace

    def send_to_agent(
        self, agent_name: str, content: str, context_name: Optional[str] = None
    ) -> str:
        """Send a message to a specific agent."""
        self.current_team_context = (
            context_name if context_name else self.current_team_context
        )
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' does not exist.")
        return self.agents[agent_name].send(
            content,
            context_name,
            self.workspaces[self.current_team_context].workspace,
        )

    def set_team_task(self, task: str, context_name: Optional[str] = None) -> None:
        """Set a task for the team and break it down for individual agents.

        Args:
            task: The team task to set.
            context_name: Optional context name.

        Raises:
            ValueError: If task is empty or no workers are present.
        """
        if not task or not task.strip():
            raise ValueError("task cannot be empty")
        workers = self.worker_names()
        if not workers:
            raise ValueError("Cannot set a team task with no worker agents")

        self.team_task = task
        breakdown = self.breakdown_task(task)
        self.current_team_context = (
            context_name if context_name else self.current_team_context
        )
        ws = self.workspaces.setdefault(
            self.current_team_context, TeamContext(task, "")
        )
        ws.team_task = task
        ws.ledger = TeamLedger(
            subtasks=dict(breakdown),
            completed={name: False for name in breakdown},
            round=0,
        )

        for agent_name in workers:
            agent = self.agents[agent_name]
            subtask = breakdown.get(agent_name, f"Assist the team with: {task}")
            if self.current_team_context not in agent.list_contexts():
                agent.create_context(self.current_team_context, subtask)
            else:
                agent.switch_context(self.current_team_context)
                agent.set_system_prompt(subtask)
        ws.started = True

    def switch_team_context(self, context_name: str, team_task: str = "") -> None:
        """Switch to a different team context."""
        self.current_team_context = (
            context_name if context_name else self.current_team_context
        )
        for agent_name in self.agents:
            self.agents[agent_name].switch_context(self.current_team_context)
        if self.current_team_context not in self.workspaces:
            self.workspaces[self.current_team_context] = TeamContext(team_task, "")

    def switch_agent_context(self, agent_name: str, context_name: str) -> None:
        """Switch a specific agent to a different context."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' does not exist.")
        self.agents[agent_name].switch_context(context_name)

    def clear_agent_context(
        self, agent_name: str, context_name: Optional[str] = None
    ) -> None:
        """Clear context for a specific agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' does not exist.")
        self.agents[agent_name].clear_context(context_name)

    def clear_team_context(self, context_name: Optional[str] = None) -> None:
        """Clear context for all agents in the team."""
        if context_name in self.workspaces:
            del self.workspaces[context_name]
        if self.current_team_context == context_name:
            self.current_team_context = "DEFAULT"
        for agent_name in self.agents:
            self.agents[agent_name].clear_context(context_name)

    # ------------------------------------------------------------------
    # Coordinator-led planning
    # ------------------------------------------------------------------

    def _coordinator_json(
        self, prompt: str, retries: int = 1, context_name: Optional[str] = None
    ) -> dict:
        """Send a prompt to the coordinator and parse a JSON response.

        Performs ``retries`` additional attempts with a stricter follow-up
        prompt if the first response cannot be parsed.
        """
        coordinator = self.agents["coordinator"]
        attempt = 0
        last_response = ""
        current_prompt = prompt
        while True:
            response = coordinator.send(current_prompt, context_name)
            last_response = response
            parsed = _extract_json_object(response)
            if parsed is not None:
                return parsed
            attempt += 1
            if attempt > retries:
                break
            current_prompt = (
                "Your previous response could not be parsed as JSON. "
                "Reply ONLY with a single JSON object — no prose, no "
                "markdown fences. Re-emit the requested structure now."
            )
        raise ValueError(
            f"Coordinator did not return parseable JSON after {retries + 1} "
            f"attempt(s). Last response: {last_response!r}"
        )

    def breakdown_task(self, task: str) -> dict[str, str]:
        """Break ``task`` into per-worker subtasks via the coordinator.

        Args:
            task: The team task to break down.

        Returns:
            Mapping of worker agent name → subtask description. Every worker
            in :meth:`worker_names` is guaranteed a key (a generic fallback
            subtask is filled in if the coordinator omits one).

        Raises:
            ValueError: If ``task`` is empty, no workers are present, or the
                coordinator cannot produce parseable JSON.
        """
        if not task or not task.strip():
            raise ValueError("task cannot be empty")
        workers = self.worker_names()
        if not workers:
            raise ValueError("Cannot break down a task with no worker agents")

        roster = ", ".join(workers)
        prompt = (
            "Break the following team task into one concrete subtask per "
            "worker agent. Reply with a single JSON object matching this "
            'schema: {"subtasks": [{"agent": "<name>", "goal": "<short '
            'goal>", "constraints": "<constraints or empty>", "deliverable":'
            ' "<expected output>"}], "success_criteria": "<overall criteria>"}'
            f"\n\nWorker agents: {roster}\n"
            f"Number of workers: {len(workers)}\n"
            f"Task:\n{task}"
        )
        plan = self._coordinator_json(prompt)
        raw_subtasks = plan.get("subtasks") or []
        if not isinstance(raw_subtasks, list):
            raise ValueError(
                "Coordinator plan 'subtasks' must be a list, got "
                f"{type(raw_subtasks).__name__}"
            )

        breakdown: dict[str, str] = {}
        for entry in raw_subtasks:
            if not isinstance(entry, dict):
                continue
            name = entry.get("agent")
            if not isinstance(name, str) or name not in workers:
                logger.warning(
                    "Coordinator referenced unknown agent %r; ignoring", name
                )
                continue
            description = " | ".join(
                str(entry.get(k, "")).strip()
                for k in ("goal", "constraints", "deliverable")
                if entry.get(k)
            ).strip()
            breakdown[name] = description or f"Contribute to: {task}"

        for name in workers:
            breakdown.setdefault(name, f"Contribute to the team task: {task}")
        return breakdown

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _format_notes_view(self, notes: list[NoteEntry]) -> str:
        if not notes:
            return "(no notes yet)"
        return "\n".join(f"- [{n.agent} round {n.round}] {n.content}" for n in notes)

    def _check_completion(
        self, ledger: TeamLedger, context_name: Optional[str] = None
    ) -> dict[str, bool]:
        """Ask the coordinator to mark per-agent completion."""
        workers = list(ledger.subtasks.keys())
        if not workers:
            return {}
        roster = ", ".join(workers)
        recent_notes = self._format_notes_view(ledger.notes[-len(workers) * 2 :])
        prompt = (
            "Given the latest worker notes, which workers have completed "
            "their subtask? Reply with a single JSON object whose 'completed' "
            "field maps each worker name to true or false. Do not include "
            "agents not in the worker list.\n"
            f"Workers: {roster}\n"
            f"Recent notes:\n{recent_notes}\n"
            'Schema: {"completed": {"<agent>": <bool>}}'
        )
        try:
            response = self._coordinator_json(prompt, context_name=context_name)
        except ValueError:
            logger.warning("Completion check failed to parse; assuming none complete")
            return {name: False for name in workers}
        raw = response.get("completed") or {}
        if not isinstance(raw, dict):
            return {name: False for name in workers}
        result: dict[str, bool] = {}
        for name in workers:
            value = raw.get(name)
            result[name] = bool(value) if isinstance(value, (bool, int)) else False
        return result

    def iterate(
        self,
        team_task: Optional[str] = None,
        max_rounds: Optional[int] = None,
        completion_check: bool = True,
    ) -> str:
        """Iterate workers on the current task with a shared notes ledger.

        Args:
            team_task: Optional task to set before iterating. If the workspace
                hasn't been started, this seeds the team task.
            max_rounds: Maximum number of rounds to run. Defaults to
                :attr:`DEFAULT_MAX_ROUNDS`.
            completion_check: When True the coordinator is asked after each
                round whether each worker is finished, and iteration stops
                early when all are marked complete.

        Returns:
            A formatted string concatenating every note added during this
            call, in order.

        Raises:
            ValueError: If no team task is set or there are no workers.
        """
        rounds_cap = max_rounds if max_rounds is not None else self.DEFAULT_MAX_ROUNDS
        if rounds_cap <= 0:
            raise ValueError("max_rounds must be positive")
        workers = self.worker_names()
        if not workers:
            raise ValueError("No worker agents to iterate")

        resolved_task = team_task or getattr(self, "team_task", None)
        if not resolved_task:
            raise ValueError("No team task set.")
        self.team_task = resolved_task

        ws = self.workspaces[self.current_team_context]
        if not ws.started or ws.ledger is None:
            self.set_team_task(resolved_task, self.current_team_context)
            ws = self.workspaces[self.current_team_context]
        ledger = ws.ledger
        assert ledger is not None  # narrow for type checker

        produced_this_call: list[NoteEntry] = []
        for _ in range(rounds_cap):
            if completion_check and ledger.all_complete():
                break
            ledger.round += 1
            current_round = ledger.round
            for agent_name in workers:
                if completion_check and ledger.completed.get(agent_name):
                    continue
                logger.debug("AGENT %s round %d", agent_name, current_round)
                view = ledger.notes_for(agent_name, ws.parallel_notes)
                prompt = (
                    f"Round {current_round}. Your subtask:\n"
                    f"{ledger.subtasks.get(agent_name, '')}\n\n"
                    "Team notes so far:\n"
                    f"{self._format_notes_view(view)}\n\n"
                    "Provide a concrete update. End with 'STATUS: DONE' "
                    "if your subtask is complete, otherwise 'STATUS: IN_PROGRESS'."
                )
                response = self.agents[agent_name].send(
                    prompt, self.current_team_context
                )
                entry = NoteEntry(
                    agent=agent_name, round=current_round, content=response
                )
                ledger.notes.append(entry)
                produced_this_call.append(entry)
                if "STATUS: DONE" in response.upper():
                    ledger.completed[agent_name] = True

            if completion_check:
                marks = self._check_completion(ledger, self.current_team_context)
                for name, done in marks.items():
                    if done:
                        ledger.completed[name] = True
                if ledger.all_complete():
                    break

        return "\n".join(f"{n.agent}: {n.content}" for n in produced_this_call)

    def show_team(self, context_name: Optional[str] = None) -> None:
        """Display team member contexts (debug logging)."""
        self.current_team_context = (
            context_name if context_name else self.current_team_context
        )
        for agent_name, agent in self.agents.items():
            logger.debug("Agent %s:", agent_name)
            if self.current_team_context in agent.list_contexts():
                ctx = agent.contexts[self.current_team_context]
                logger.debug("%s", ctx.message_history)
