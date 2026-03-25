# Multi-Agent Architecture Refactor

## Overview

This document describes the refactored relationship between agents, models, and flowcharts in the pithos/TheTeam framework.

## Key Changes

### 1. Agents as Roles

**Before:** Agents were tightly coupled to a specific model.

**After:** Agents represent roles with:
- **System prompt** - Defines the agent's behavior and expertise
- **Model settings** - Temperature, max_tokens, etc.
- **Conversation history** - Maintained across model swaps
- **Knowledge sources** - Tools and memory access
- **Optional CoT flowchart** - Internal reasoning process
- **Default model** - Can be overridden per call

### 2. Model Flexibility

Models are now swappable on demand:

```python
# Agent has a default model
agent = OllamaAgent(
    default_model="llama3.2",
    agent_name="assistant",
    system_prompt="You are helpful.",
)

# Use default model
response = agent.send("Hello")

# Override model for specific call
response = agent.send("Hello", model="phi4")

# History persists regardless of model used
```

### 3. Two Types of Flowcharts

#### Chain-of-Thought (CoT) Flowcharts
- **Purpose:** Single agent's internal reasoning process
- **Execution:** Agent executes via `follow_flowchart()`
- **Nodes:** Use `PromptNode` which generates prompts for the agent
- **Use case:** Multi-step reasoning, reflection, systematic problem-solving

Example:
```python
agent = OllamaAgent(default_model="llama3.2")
flowchart = Flowchart.from_registered("simple_reflect", config_manager)

result = agent.follow_flowchart(
    flowchart,
    initial_prompt="What is the capital of France?",
)
```

#### Team Flowcharts
- **Purpose:** Multi-agent coordination and collaboration
- **Execution:** Flowchart coordinates multiple agents
- **Nodes:** Use `AgentPromptNode` which specifies which agent to use
- **Use case:** Multi-agent workflows, role-based collaboration

Example:
```python
# Create agents with different roles
researcher = OllamaAgent(
    default_model="llama3.2",
    system_prompt="You are a thorough researcher.",
)

writer = OllamaAgent(
    default_model="llama3.2",
    system_prompt="You are a skilled writer.",
)

# Load team flowchart
flowchart = Flowchart.from_registered("multi_agent_research", config_manager)

# Execute with all agents
result = flowchart.run_team_flowchart(
    agents={"researcher": researcher, "writer": writer},
    initial_input="Research AI in healthcare",
)
```

## New Node Types

### AgentPromptNode

Executes a prompt using a specific agent in team flowcharts.

**YAML Configuration:**
```yaml
nodes:
  Research:
    type: agentprompt
    agent: researcher  # Which agent to use
    prompt: "Research the topic: {topic}"
    model: llama3.2  # Optional: override agent's default model
    context_name: research_context  # Optional: specific context
    extraction: {}
    inputs: [default]
    outputs: [default]
```

**Python:**
```python
from pithos.flownode import AgentPromptNode

node = AgentPromptNode(
    agent="researcher",
    prompt="Research {topic}",
    model="llama3.2",  # Optional override
)
```

### GetHistoryNode

Extracts an agent's conversation history for sharing or inspection.

**YAML Configuration:**
```yaml
nodes:
  ExtractHistory:
    type: gethistory
    agent: teacher
    save_to: teacher_history  # State variable to save to
    context_name: default  # Optional: specific context
    extraction: {}
```

**Python:**
```python
from pithos.flownode import GetHistoryNode

node = GetHistoryNode(
    agent="teacher",
    save_to="teacher_history",
)

# In flowchart execution
state = {"agents": {"teacher": teacher_agent}}
result = node.do(state)
# state["teacher_history"] now contains the conversation history
```

### SetHistoryNode

Injects conversation history into an agent.

**YAML Configuration:**
```yaml
nodes:
  ShareContext:
    type: sethistory
    agent: student
    history_from: teacher_history  # State variable with history
    mode: replace  # 'replace' or 'append'
    context_name: default  # Optional: specific context
    extraction: {}
```

**Python:**
```python
from pithos.flownode import SetHistoryNode

node = SetHistoryNode(
    agent="student",
    history_from="teacher_history",
    mode="replace",  # or "append"
)
```

## Agent Configuration Changes

### Old Format (Legacy - Still Supported)
```yaml
model: llama3.2
name: my-agent
system_prompt: "You are helpful."
temperature: 0.7
```

### New Format (Recommended)
```yaml
default_model: llama3.2
name: my-agent
system_prompt: "You are helpful."
temperature: 0.7
```

**Backward Compatibility:** The system automatically supports both `model` and `default_model` keys when loading configurations.

## Flowchart Detection

Flowcharts automatically detect their type based on node contents:

```python
flowchart = Flowchart.from_registered("my_flowchart", config_manager)

# Check flowchart type
if flowchart.is_team_flowchart():
    # Contains AgentPromptNode, GetHistoryNode, or SetHistoryNode
    # Execute as team flowchart
    result = flowchart.run_team_flowchart(agents=my_agents, initial_input=input)
else:
    # Contains only PromptNode, CustomNode, etc.
    # Execute as CoT flowchart
    result = agent.follow_flowchart(flowchart, initial_prompt=input)
```

## Migration Guide

### Updating Existing Code

1. **Agent Creation:**
   ```python
   # Old
   agent = OllamaAgent(model_name="llama3.2")
   
   # New
   agent = OllamaAgent(default_model="llama3.2")
   ```

2. **Agent Configurations:**
   - Update YAML files to use `default_model` instead of `model`
   - Legacy `model` key still works for backward compatibility

3. **Using Different Models:**
   ```python
   # Old: Had to create new agent
   agent_llama = OllamaAgent(model_name="llama3.2")
   agent_phi = OllamaAgent(model_name="phi4")
   
   # New: Same agent, different models
   agent = OllamaAgent(default_model="llama3.2")
   response1 = agent.send("Query", model="llama3.2")
   response2 = agent.send("Query", model="phi4")
   # History maintained across both calls
   ```

4. **Multi-Agent Workflows:**
   ```python
   # Old: Manual coordination
   researcher_response = researcher.send(query)
   writer_response = writer.send(f"Write about: {researcher_response}")
   
   # New: Team flowchart
   flowchart = Flowchart.from_registered("research_team", config_manager)
   result = flowchart.run_team_flowchart(
       agents={"researcher": researcher, "writer": writer},
       initial_input=query,
   )
   ```

## Best Practices

### 1. Agent Design
- **Define clear roles:** Each agent should have a specific purpose
- **Use descriptive system prompts:** Guide agent behavior through prompts
- **Set appropriate temperature:** Lower for consistency, higher for creativity
- **Choose meaningful agent names:** They're used as identifiers in team flowcharts

### 2. Model Selection
- **Default model:** Choose based on agent's primary use case
- **Model overrides:** Use for specific tasks (e.g., use larger model for complex reasoning)
- **Consistency:** Keep the same model within a logical workflow when possible

### 3. Team Flowcharts
- **Start simple:** Begin with 2-3 agents before scaling up
- **Clear data flow:** Use descriptive variable names in `set` operations
- **Context sharing:** Use GetHistoryNode/SetHistoryNode sparingly and purposefully
- **Error handling:** Ensure all required agents are provided

### 4. History Management
- **Independent histories:** Each agent maintains its own conversation history
- **Context isolation:** Use separate context names for different tasks
- **History transfer:** Only share history when agents truly need each other's context

## Examples

See the following files for complete examples:

- **Multi-Agent Research:** `configs/flowcharts/multi_agent_research.yaml`
- **Context Sharing:** `configs/flowcharts/agent_context_sharing.yaml`
- **Python Examples:** `src/pithos/examples/multi_agent_example.py`

## Testing

All changes are covered by unit tests:

- **Agent tests:** `tests/test_agent.py`
- **Node tests:** `tests/test_flownode.py`
- **Full test suite:** Run `pytest` from project root

## Summary

This refactor provides:

✅ **Flexible model usage** - Swap models while maintaining agent identity
✅ **Clear role definition** - Agents represent roles, not just model wrappers
✅ **Multi-agent coordination** - Team flowcharts enable complex collaboration
✅ **History independence** - Each agent tracks its own conversation
✅ **Backward compatibility** - Legacy configurations still work
✅ **Type safety** - Flowcharts self-identify as CoT or Team
✅ **Comprehensive testing** - All functionality covered by unit tests
