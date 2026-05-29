"""Tests for :mod:`pithos.team.agent_manager`.

These tests exercise the AgentTeam coordination logic with the underlying
``OllamaAgent`` mocked out so no network calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pithos.team.agent_manager import (
    AgentTeam,
    NoteEntry,
    TeamLedger,
    _extract_json_object,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_mock_agent(responses=None):
    """Return a MagicMock standing in for an OllamaAgent.

    ``responses`` is a list consumed in order by ``send``. After exhaustion
    the final response is repeated.
    """
    agent = MagicMock()
    agent._contexts = {}

    def create_context(name, prompt=""):
        agent._contexts[name] = prompt

    def list_contexts():
        return list(agent._contexts.keys())

    def switch_context(name):
        agent._current = name

    def set_system_prompt(prompt):
        if hasattr(agent, "_current"):
            agent._contexts[agent._current] = prompt

    def clear_context(name=None):
        if name is None:
            agent._contexts.clear()
        else:
            agent._contexts.pop(name, None)

    agent.create_context.side_effect = create_context
    agent.list_contexts.side_effect = list_contexts
    agent.switch_context.side_effect = switch_context
    agent.set_system_prompt.side_effect = set_system_prompt
    agent.clear_context.side_effect = clear_context
    agent.contexts = agent._contexts

    rsp = list(responses or ["ok"])

    def send(content, context_name=None, *args, **kwargs):
        if len(rsp) > 1:
            return rsp.pop(0)
        return rsp[0] if rsp else "ok"

    agent.send.side_effect = send
    return agent


@pytest.fixture
def patched_ollama():
    """Patch OllamaAgent so AgentTeam constructs without network calls."""
    with patch("pithos.team.agent_manager.OllamaAgent") as cls:
        instances: list[MagicMock] = []

        def factory(default_model=None, **kwargs):
            mock = _make_mock_agent()
            instances.append(mock)
            return mock

        cls.side_effect = factory
        cls.instances = instances
        yield cls


def _build_team(patched_ollama, workers=("alice", "bob")):
    """Construct a team and return (team, coordinator_mock, worker_mocks)."""
    team = AgentTeam("test-model")
    coordinator = team.agents["coordinator"]
    worker_mocks = {}
    for name in workers:
        team.add_agent(name, "test-model")
        worker_mocks[name] = team.agents[name]
    return team, coordinator, worker_mocks


def _set_coordinator_responses(coordinator, responses):
    """Override the coordinator's send behaviour with a fixed response queue."""
    queue = list(responses)

    def send(content, context_name=None, *args, **kwargs):
        if not queue:
            raise AssertionError("coordinator.send called more times than expected")
        return queue.pop(0)

    coordinator.send.side_effect = send
    return queue


# ---------------------------------------------------------------------------
# _extract_json_object
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_plain_object(self):
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_object_with_prose(self):
        text = 'Sure! Here is the plan: {"a": 1, "b": [1,2]} hope that helps.'
        assert _extract_json_object(text) == {"a": 1, "b": [1, 2]}

    def test_markdown_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert _extract_json_object(text) == {"a": 1}

    def test_unfenced_with_language(self):
        text = '```\n{"a": 1}\n```'
        assert _extract_json_object(text) == {"a": 1}

    def test_returns_none_for_empty(self):
        assert _extract_json_object("") is None
        assert _extract_json_object("   ") is None

    def test_returns_none_for_invalid_json(self):
        assert _extract_json_object("not json {{{") is None

    def test_returns_none_for_array_only(self):
        # Top-level arrays not supported (we extract objects).
        assert _extract_json_object("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# Construction / roster
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_construction_creates_coordinator(self, patched_ollama):
        team = AgentTeam("test-model")
        assert "coordinator" in team.agents
        assert "DEFAULT" in team.workspaces
        assert team.current_team_context == "DEFAULT"

    def test_empty_model_rejected(self, patched_ollama):
        with pytest.raises(ValueError, match="coordinator_model"):
            AgentTeam("")

    def test_empty_init_context_rejected(self, patched_ollama):
        with pytest.raises(ValueError, match="init_context"):
            AgentTeam("test-model", init_context="")

    def test_add_and_remove_worker(self, patched_ollama):
        team = AgentTeam("test-model")
        team.add_agent("alice", "test-model")
        assert team.worker_names() == ["alice"]
        team.remove_agent("alice")
        assert team.worker_names() == []

    def test_cannot_add_coordinator_name(self, patched_ollama):
        team = AgentTeam("test-model")
        with pytest.raises(ValueError, match="reserved"):
            team.add_agent("coordinator", "test-model")

    def test_cannot_remove_coordinator(self, patched_ollama):
        team = AgentTeam("test-model")
        with pytest.raises(ValueError, match="Cannot remove"):
            team.remove_agent("coordinator")

    def test_duplicate_agent_rejected(self, patched_ollama):
        team = AgentTeam("test-model")
        team.add_agent("alice", "test-model")
        with pytest.raises(ValueError, match="already exists"):
            team.add_agent("alice", "test-model")


# ---------------------------------------------------------------------------
# breakdown_task
# ---------------------------------------------------------------------------


class TestBreakdownTask:
    def test_happy_path(self, patched_ollama):
        team, coordinator, _ = _build_team(patched_ollama, ("alice", "bob"))
        plan = {
            "subtasks": [
                {
                    "agent": "alice",
                    "goal": "design wing",
                    "constraints": "lightweight",
                    "deliverable": "sketch",
                },
                {
                    "agent": "bob",
                    "goal": "design tail",
                    "constraints": "",
                    "deliverable": "sketch",
                },
            ],
            "success_criteria": "complete sketches",
        }
        _set_coordinator_responses(coordinator, [json.dumps(plan)])
        result = team.breakdown_task("Build an airplane")
        assert set(result.keys()) == {"alice", "bob"}
        assert "design wing" in result["alice"]
        assert "lightweight" in result["alice"]
        assert "design tail" in result["bob"]

    def test_malformed_json_retry_succeeds(self, patched_ollama):
        team, coordinator, _ = _build_team(patched_ollama, ("alice",))
        plan = {"subtasks": [{"agent": "alice", "goal": "g", "deliverable": "d"}]}
        _set_coordinator_responses(
            coordinator,
            ["this is not json at all", json.dumps(plan)],
        )
        result = team.breakdown_task("Task")
        assert "alice" in result

    def test_malformed_json_retry_exhausted(self, patched_ollama):
        team, coordinator, _ = _build_team(patched_ollama, ("alice",))
        _set_coordinator_responses(coordinator, ["nope", "still nope"])
        with pytest.raises(ValueError, match="parseable JSON"):
            team.breakdown_task("Task")

    def test_unknown_agent_in_plan_ignored(self, patched_ollama):
        team, coordinator, _ = _build_team(patched_ollama, ("alice", "bob"))
        plan = {
            "subtasks": [
                {"agent": "alice", "goal": "g", "deliverable": "d"},
                {"agent": "ghost", "goal": "x", "deliverable": "y"},
            ]
        }
        _set_coordinator_responses(coordinator, [json.dumps(plan)])
        result = team.breakdown_task("Task")
        assert set(result.keys()) == {"alice", "bob"}
        # bob got the fallback subtask.
        assert "Contribute" in result["bob"]

    def test_empty_task_rejected(self, patched_ollama):
        team, _, _ = _build_team(patched_ollama, ("alice",))
        with pytest.raises(ValueError, match="task cannot be empty"):
            team.breakdown_task("")

    def test_no_workers_rejected(self, patched_ollama):
        team = AgentTeam("test-model")
        with pytest.raises(ValueError, match="no worker agents"):
            team.breakdown_task("Task")


# ---------------------------------------------------------------------------
# iterate
# ---------------------------------------------------------------------------


class TestIterate:
    def _seed_plan(self, coordinator, workers):
        plan = {
            "subtasks": [
                {"agent": w, "goal": f"{w} goal", "deliverable": "d"} for w in workers
            ]
        }
        return json.dumps(plan)

    def test_single_round(self, patched_ollama):
        team, coordinator, workers = _build_team(patched_ollama, ("alice",))
        # plan + completion check (alice not done) -> 2 coordinator messages
        _set_coordinator_responses(
            coordinator,
            [
                self._seed_plan(coordinator, ["alice"]),
                json.dumps({"completed": {"alice": False}}),
            ],
        )
        result = team.iterate("Task", max_rounds=1)
        ledger = team.workspaces["DEFAULT"].ledger
        assert ledger is not None
        assert ledger.round == 1
        assert len(ledger.notes) == 1
        assert "alice:" in result

    def test_completes_via_status_done(self, patched_ollama):
        team, coordinator, workers = _build_team(patched_ollama, ("alice",))
        # Worker says STATUS: DONE -> no completion check needed afterwards
        # because all_complete short-circuits the next round.
        workers["alice"].send.side_effect = lambda *a, **k: "Built it. STATUS: DONE"
        _set_coordinator_responses(
            coordinator,
            [
                self._seed_plan(coordinator, ["alice"]),
                # Completion check still runs once at end of round 1.
                json.dumps({"completed": {"alice": True}}),
            ],
        )
        team.iterate("Task", max_rounds=5)
        ledger = team.workspaces["DEFAULT"].ledger
        assert ledger.round == 1
        assert ledger.completed == {"alice": True}

    def test_multi_round_with_completion(self, patched_ollama):
        team, coordinator, workers = _build_team(patched_ollama, ("alice", "bob"))
        # Coordinator: plan, then per-round completion checks
        _set_coordinator_responses(
            coordinator,
            [
                self._seed_plan(coordinator, ["alice", "bob"]),
                json.dumps({"completed": {"alice": False, "bob": False}}),
                json.dumps({"completed": {"alice": True, "bob": False}}),
                json.dumps({"completed": {"alice": True, "bob": True}}),
            ],
        )
        team.iterate("Task", max_rounds=5)
        ledger = team.workspaces["DEFAULT"].ledger
        assert ledger.round == 3
        assert ledger.all_complete()

    def test_max_rounds_cap(self, patched_ollama):
        team, coordinator, workers = _build_team(patched_ollama, ("alice",))
        responses = [self._seed_plan(coordinator, ["alice"])]
        # never marks complete
        responses.extend(json.dumps({"completed": {"alice": False}}) for _ in range(5))
        _set_coordinator_responses(coordinator, responses)
        team.iterate("Task", max_rounds=2)
        ledger = team.workspaces["DEFAULT"].ledger
        assert ledger.round == 2

    def test_no_team_task_raises(self, patched_ollama):
        team, _, _ = _build_team(patched_ollama, ("alice",))
        with pytest.raises(ValueError, match="No team task set"):
            team.iterate()

    def test_no_workers_raises(self, patched_ollama):
        team = AgentTeam("test-model")
        with pytest.raises(ValueError, match="No worker agents"):
            team.iterate("Task")

    def test_invalid_max_rounds(self, patched_ollama):
        team, _, _ = _build_team(patched_ollama, ("alice",))
        with pytest.raises(ValueError, match="max_rounds must be positive"):
            team.iterate("Task", max_rounds=0)

    def test_completion_check_disabled_runs_all_rounds(self, patched_ollama):
        team, coordinator, workers = _build_team(patched_ollama, ("alice",))
        _set_coordinator_responses(
            coordinator, [self._seed_plan(coordinator, ["alice"])]
        )
        team.iterate("Task", max_rounds=3, completion_check=False)
        ledger = team.workspaces["DEFAULT"].ledger
        assert ledger.round == 3
        assert len(ledger.notes) == 3

    def test_parallel_notes_view(self, patched_ollama):
        team, coordinator, workers = _build_team(patched_ollama, ("alice", "bob"))
        team.workspaces["DEFAULT"].parallel_notes = True
        _set_coordinator_responses(
            coordinator,
            [
                self._seed_plan(coordinator, ["alice", "bob"]),
                json.dumps({"completed": {"alice": False, "bob": False}}),
                json.dumps({"completed": {"alice": True, "bob": True}}),
            ],
        )
        # Track the prompts seen by alice in round 2 — should NOT include bob's notes.
        alice_prompts: list[str] = []

        def alice_send(content, context_name=None, *args, **kwargs):
            alice_prompts.append(content)
            return "alice update"

        workers["alice"].send.side_effect = alice_send
        workers["bob"].send.side_effect = lambda *a, **k: "bob update"
        team.iterate("Task", max_rounds=2)
        # Round 2 prompt for alice should reference her round-1 note but not bob's.
        round2 = alice_prompts[1]
        assert "alice round 1" in round2
        assert "bob round 1" not in round2


# ---------------------------------------------------------------------------
# TeamLedger
# ---------------------------------------------------------------------------


class TestTeamLedger:
    def test_all_complete_empty(self):
        ledger = TeamLedger()
        assert not ledger.all_complete()

    def test_all_complete_partial(self):
        ledger = TeamLedger(completed={"a": True, "b": False})
        assert not ledger.all_complete()

    def test_all_complete_true(self):
        ledger = TeamLedger(completed={"a": True, "b": True})
        assert ledger.all_complete()

    def test_notes_for_shared(self):
        ledger = TeamLedger(notes=[NoteEntry("a", 1, "x"), NoteEntry("b", 1, "y")])
        assert len(ledger.notes_for("a", parallel=False)) == 2

    def test_notes_for_parallel(self):
        ledger = TeamLedger(notes=[NoteEntry("a", 1, "x"), NoteEntry("b", 1, "y")])
        own = ledger.notes_for("a", parallel=True)
        assert len(own) == 1 and own[0].agent == "a"
