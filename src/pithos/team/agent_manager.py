"""Agent Team Manager - coordinates multiple agents working together."""

from dataclasses import dataclass
import logging
from typing import Optional

from ..agent import OllamaAgent

logger = logging.getLogger(__name__)


@dataclass
class TeamContext:
    """Context for a team of agents working together."""

    team_task: str
    workspace: str
    started: bool = False


class AgentTeam:
    """Coordinates multiple agents working together on tasks."""

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

    def init_coordinator(self, model_name: str) -> None:
        """Initialize the coordinator agent."""
        self.agents["coordinator"] = OllamaAgent(default_model=model_name)
        self.agents["coordinator"].create_context(
            "DEFAULT",
            "You are a project manager and team coordinator. You have been tasked with managing a team of agents to complete a project.",
        )

    def add_agent(self, agent_name: str, model_name: str) -> None:
        """Add a new agent to the team.

        Args:
            agent_name: Name for the agent.
            model_name: Model name for the agent.

        Raises:
            ValueError: If agent with same name already exists or if names are empty.
        """
        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name cannot be empty")
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")

        if agent_name not in self.agents:
            self.agents[agent_name] = OllamaAgent(default_model=model_name)
        else:
            raise ValueError(f"Agent '{agent_name}' already exists.")

    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the team.

        Args:
            agent_name: Name of the agent to remove.

        Raises:
            ValueError: If agent doesn't exist.
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
        else:
            raise ValueError(f"Agent '{agent_name}' does not exist.")

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
        if agent_name in self.agents:
            return self.agents[agent_name].send(
                content,
                context_name,
                self.workspaces[self.current_team_context].workspace,
            )
        else:
            raise ValueError(f"Agent '{agent_name}' does not exist.")

    def set_team_task(self, task: str, context_name: Optional[str] = None) -> None:
        """Set a task for the team and break it down for individual agents.

        Args:
            task: The team task to set.
            context_name: Optional context name.

        Raises:
            ValueError: If task is empty.
        """
        if not task or not task.strip():
            raise ValueError("task cannot be empty")

        self.team_task = task
        task_breakdown = self.breakdown_task(task)
        self.current_team_context = (
            context_name if context_name else self.current_team_context
        )
        self.workspaces[self.current_team_context].team_task = task
        for agent_name, agent in self.agents.items():
            if agent_name == "coordinator":
                continue
            if self.current_team_context not in agent.list_contexts():
                agent.create_context(self.current_team_context, task_breakdown.pop())
            else:
                agent.switch_context(self.current_team_context)
                agent.set_system_prompt(task_breakdown.pop())
        self.workspaces[self.current_team_context].started = True

    def switch_team_context(self, context_name: str, team_task: str = "") -> None:
        """Switch to a different team context.

        Args:
            context_name: Name of the context to switch to.
            team_task: Optional task for the new context.
        """
        self.current_team_context = (
            context_name if context_name else self.current_team_context
        )
        for agent_name in self.agents:
            self.agents[agent_name].switch_context(self.current_team_context)
        if self.current_team_context not in self.workspaces:
            self.workspaces[self.current_team_context] = TeamContext(team_task, "")

    def switch_agent_context(self, agent_name: str, context_name: str) -> None:
        """Switch a specific agent to a different context.

        Args:
            agent_name: Name of the agent.
            context_name: Name of the context to switch to.

        Raises:
            ValueError: If agent doesn't exist.
        """
        if agent_name in self.agents:
            self.agents[agent_name].switch_context(context_name)
        else:
            raise ValueError(f"Agent '{agent_name}' does not exist.")

    def clear_agent_context(
        self, agent_name: str, context_name: Optional[str] = None
    ) -> None:
        """Clear context for a specific agent."""
        if agent_name in self.agents:
            self.agents[agent_name].clear_context(context_name)
        else:
            raise ValueError(f"Agent '{agent_name}' does not exist.")

    def clear_team_context(self, context_name: Optional[str] = None) -> None:
        """Clear context for all agents in the team."""
        if context_name in self.workspaces:
            del self.workspaces[context_name]
        if self.current_team_context == context_name:
            self.current_team_context = "DEFAULT"
        for agent_name in self.agents:
            self.agents[agent_name].clear_context(context_name)

    def breakdown_task(self, task: str) -> list[str]:
        """Break down a task into subtasks for individual agents.

        Args:
            task: The team task to break down.

        Returns:
            List of subtask descriptions.

        Raises:
            ValueError: If task is empty.
        """
        if not task or not task.strip():
            raise ValueError("task cannot be empty")

        # TODO: implement task breakdown
        self.agents["coordinator"].send(
            f"examine the following task. How should this task be broken down into discrete tasking for {len(self.agents) - 1} team members? Exclude the project manager.\n\n"
            + task
        )
        task_texts = []
        agent_idx = 1
        for agent_name, agent in self.agents.items():
            if agent_name == "coordinator":
                continue
            task_texts.append(
                self.agents["coordinator"].send(
                    f"Give detailed directions for team member #{agent_idx}. Be as detailed as you can in your direction and give explicit requirements and constraints. Do not include any additional information that is not relevant to the tasking such as framing and conclusions.",
                )
            )
            agent_idx += 1
        return task_texts

    def iterate(self, team_task: Optional[str] = None) -> str:
        """Iterate team members on the current task.

        Args:
            team_task: Optional task to set before iterating.

        Returns:
            Accumulated notes from all agents.

        Raises:
            ValueError: If no team task is set.
        """
        # TODO: implement proper task iteration
        self.team_task = team_task if team_task else getattr(self, "team_task", None)
        if not self.team_task:
            raise ValueError("No team task set.")
        if not self.workspaces[self.current_team_context].started:
            self.workspaces[self.current_team_context].team_task = self.team_task
            self.workspaces[self.current_team_context].started = True
            self.set_team_task(self.team_task, self.current_team_context)
        # TODO: implement task completion tracking
        # TODO: implement parallel notes vs shared notes
        notes = ""
        for agent_name, agent in self.agents.items():
            if agent_name == "coordinator":
                continue
            logger.debug("AGENT %s", agent_name)
            # TODO: implement proper iteration mechanism
            response = agent.send(f"Task update: {notes}", self.current_team_context)
            notes += f"\n{agent_name}: {response}"
        return notes

    def show_team(self, context_name: Optional[str] = None) -> None:
        """Display team member contexts."""
        self.current_team_context = (
            context_name if context_name else self.current_team_context
        )
        for agent_name, agent in self.agents.items():
            logger.debug("Agent %s:", agent_name)
            if self.current_team_context in agent.list_contexts():
                ctx = agent.contexts[self.current_team_context]
                logger.debug("%s", ctx.message_history)


def team_test(
    team_size: int = 5, coordinator_model: str = "Phi4", team_model: str = "Phi4"
) -> None:
    """Test the AgentTeam functionality.

    Args:
        team_size: Number of agents to create.
        coordinator_model: Model name for coordinator.
        team_model: Model name for team members.
    """
    team = AgentTeam(coordinator_model)
    for i in range(team_size):
        team.add_agent(f"agent{i}", team_model)
    team.set_team_task("create a novel aircraft design.")
    print("*******TEAM***********")
    team.show_team()
    print("******ITERATION*******")
    print(team.iterate())
    print("*******COMPLETE*******")


if __name__ == "__main__":
    team_test()
