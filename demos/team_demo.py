"""Smoke demo for AgentTeam coordination.

Run with:
    python -m demos.team_demo
"""

from __future__ import annotations

import logging

from pithos.team import AgentTeam


def team_demo(
    team_size: int = 3,
    coordinator_model: str = "Phi4",
    team_model: str = "Phi4",
    max_rounds: int = 2,
) -> None:
    """Spin up a small team, set a task, and run a couple of rounds."""
    logging.basicConfig(level=logging.INFO)
    team = AgentTeam(coordinator_model)
    for i in range(team_size):
        team.add_agent(f"agent{i}", team_model)
    team.set_team_task("Create a novel aircraft design.")
    print("******* TEAM ***********")
    team.show_team()
    print("****** ITERATION *******")
    print(team.iterate(max_rounds=max_rounds))
    print("******* COMPLETE *******")
    ledger = team.workspaces[team.current_team_context].ledger
    if ledger:
        print(
            f"Ledger: {ledger.round} round(s), {len(ledger.notes)} note(s), "
            f"completion={ledger.completed}"
        )


if __name__ == "__main__":
    team_demo()
