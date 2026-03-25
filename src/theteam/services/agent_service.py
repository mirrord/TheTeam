"""
Agent service - manages agent configurations and operations.
"""

import logging
from pathlib import Path
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing agents."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize agent service.

        Args:
            config_dir: Directory containing agent configuration files.
                       Defaults to configs/agents in current working directory.
        """
        if config_dir is None:
            # Default to configs/agents directory
            config_dir = Path.cwd() / "configs" / "agents"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Runtime agents storage (for dynamically created agents)
        self.runtime_agents: dict[str, dict] = {}

    def list_agents(self) -> list[dict]:
        """List all available agents.

        Returns:
            List of agent dictionaries with id, name, model, and source.
        """
        agents = []

        # Load from YAML files
        if self.config_dir.exists():
            for yaml_file in self.config_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, "r") as f:
                        config = yaml.safe_load(f)

                    agent_id = yaml_file.stem
                    agents.append(
                        {
                            "id": agent_id,
                            "name": config.get("name", agent_id),
                            "model": config.get("model", "unknown"),
                            "source": "file",
                            "path": str(yaml_file),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error loading agent from {yaml_file}: {e}")

        # Add runtime agents
        for agent_id, config in self.runtime_agents.items():
            agents.append(
                {
                    "id": agent_id,
                    "name": config.get("name", agent_id),
                    "model": config.get("model", "unknown"),
                    "source": "runtime",
                }
            )

        return agents

    def get_agent(self, agent_id: str) -> Optional[dict]:
        """Get a specific agent's configuration.

        Args:
            agent_id: Agent identifier.

        Returns:
            Agent configuration dictionary or None if not found.
        """
        # Check runtime agents first
        if agent_id in self.runtime_agents:
            return {
                "id": agent_id,
                "config": self.runtime_agents[agent_id],
                "source": "runtime",
            }

        # Check file-based agents
        yaml_file = self.config_dir / f"{agent_id}.yaml"
        if yaml_file.exists():
            try:
                with open(yaml_file, "r") as f:
                    config = yaml.safe_load(f)
                return {
                    "id": agent_id,
                    "config": config,
                    "source": "file",
                    "path": str(yaml_file),
                }
            except Exception as e:
                logger.error(f"Error loading agent {agent_id}: {e}")
                return None

        return None

    def create_agent(self, config: dict) -> str:
        """Create a new agent.

        Args:
            config: Agent configuration dictionary.

        Returns:
            Created agent's ID.

        Raises:
            ValueError: If required 'model' field is missing.
        """
        agent_id = config.get("id")
        if not agent_id:
            # Generate ID from name
            name = config.get("name", "unnamed")
            agent_id = name.lower().replace(" ", "-")

        # Validate required fields
        if "model" not in config:
            raise ValueError("Agent configuration must include 'model'")

        # Determine if it should be saved to file or runtime
        save_to_file = config.pop("save_to_file", False)

        if save_to_file:
            yaml_file = self.config_dir / f"{agent_id}.yaml"
            try:
                with open(yaml_file, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                logger.info(f"Created agent {agent_id} in {yaml_file}")
            except Exception as e:
                logger.error(f"Error saving agent {agent_id}: {e}")
                raise
        else:
            self.runtime_agents[agent_id] = config
            logger.info(f"Created runtime agent {agent_id}")

        return agent_id

    def update_agent(self, agent_id: str, config: dict) -> bool:
        """Update an existing agent.

        Args:
            agent_id: Agent identifier.
            config: Updated configuration dictionary.

        Returns:
            True if update successful, False if agent not found.
        """
        agent = self.get_agent(agent_id)
        if not agent:
            return False

        if agent["source"] == "runtime":
            self.runtime_agents[agent_id] = config
            logger.info(f"Updated runtime agent {agent_id}")
            return True
        elif agent["source"] == "file":
            yaml_file = Path(agent["path"])
            try:
                with open(yaml_file, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                logger.info(f"Updated agent {agent_id} in {yaml_file}")
                return True
            except Exception as e:
                logger.error(f"Error updating agent {agent_id}: {e}")
                raise

        return False

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            True if deletion successful, False if agent not found.
        """
        agent = self.get_agent(agent_id)
        if not agent:
            return False

        if agent["source"] == "runtime":
            del self.runtime_agents[agent_id]
            logger.info(f"Deleted runtime agent {agent_id}")
            return True
        elif agent["source"] == "file":
            yaml_file = Path(agent["path"])
            try:
                yaml_file.unlink()
                logger.info(f"Deleted agent {agent_id} file {yaml_file}")
                return True
            except Exception as e:
                logger.error(f"Error deleting agent {agent_id}: {e}")
                raise

        return False

    def test_agent(self, agent_id: str, prompt: str) -> dict:
        """Test an agent with a prompt.

        Args:
            agent_id: Agent identifier.
            prompt: Test prompt to send to the agent.

        Returns:
            Dictionary with prompt, response, and agent_id.

        Raises:
            ValueError: If agent not found.
        """
        from pithos.agent import OllamaAgent

        agent_config = self.get_agent(agent_id)
        if not agent_config:
            raise ValueError(f"Agent {agent_id} not found")

        try:
            # Create temporary agent instance
            config = agent_config["config"]
            agent = OllamaAgent(
                default_model=config.get("model", "llama3.2:latest"),
                agent_name=config.get("name", agent_id),
                system_prompt=config.get("system_prompt", ""),
            )

            # Generate response
            response = agent.send(prompt)

            return {"prompt": prompt, "response": response, "agent_id": agent_id}
        except Exception as e:
            logger.error(f"Error testing agent {agent_id}: {e}", exc_info=True)
            raise
