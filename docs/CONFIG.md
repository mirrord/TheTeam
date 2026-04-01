# Configuration Guide

This guide explains how to configure agents, flowcharts, tools, and conditions using YAML files.

## Overview

pithos uses a configuration directory structure to organize settings:

```
configs/
├── agents/              # Agent configurations
├── flowcharts/          # Flowchart definitions
├── tools/               # Tool settings
└── conditions/          # Custom conditions (future)
```

All configuration files use YAML format and are automatically discovered by the `ConfigManager`.

## Configuration Manager

The `ConfigManager` class handles loading and registration of all configurations:

```python
from pithos import ConfigManager

# Initialize with default location (configs/)
config_manager = ConfigManager()

# Or specify custom directory
config_manager = ConfigManager(config_dir="path/to/configs")

# List available configurations
agent_names = config_manager.get_registered_agent_names()
flowchart_names = config_manager.get_registered_flowchart_names()
```

## Agent Configuration

Agent configs define model settings, system prompts, and flowcharts.

### Basic Agent Config

**File:** `configs/agents/my-agent.yaml`

```yaml
model: glm-4.7-flash:latest
system_prompt: "You are a helpful assistant."
temperature: 0.7  # Optional: Controls randomness (0.0-1.0). Default is 0.7.
```

### Agent with Flowchart

An agent can have an optional **inference flowchart** (chain-of-thought) that runs at each step of agent inference instead of a single LLM round-trip. This is useful for structured reasoning patterns like reflection, iterative refinement, or multi-step analysis.

The `inference` field accepts either a registered flowchart name or an inline flowchart definition:

**By registered name:**
```yaml
model: glm-4.7-flash:latest
system_prompt: "You are a reflective reasoner."
inference: simple_reflect
```

**Inline definition:**
```yaml
model: glm-4.7-flash:latest
name: structured-reflect
inference:
  start_node: Generate
  nodes:
    Generate:
      type: prompt
      prompt: "{current_input}"
      extraction: {}
      set:
        original_question: "{current_input}"
    Reflect:
      type: prompt
      prompt: "Reflect on your previous answer and decide if you want to change it."
      extraction: {}
      set:
        original_answer: "{current_input}"
    Regenerate:
      type: prompt
      prompt: |
        Answer the following question using the thought process below.
        **QUESTION** {original_question}
        <think>{original_answer} {current_input}</think>
        Be sure to answer the question directly and concisely.
      extraction: {}
  edges:
    - from: Generate
      to: Reflect
      condition: { type: AlwaysCondition }
    - from: Reflect
      to: Regenerate
      condition: { type: AlwaysCondition }
```

When using an inference flowchart, PromptNodes inside the flowchart call the agent's underlying LLM automatically. The intermediate reasoning steps happen in a disposable context while only the final output is recorded in the main conversation history.

### Structured Output Agent

```yaml
model: glm-4.7-flash:latest
system_prompt: "Provide structured analysis."
structured_with_format: |
  {
    "analysis": "string",
    "confidence": "number",
    "reasoning": "array of strings"
  }
```

### Agent with Tools

```yaml
model: glm-4.7-flash:latest
system_prompt: "You are an assistant that can execute commands."
enable_tools: true
```

### Advanced Agent Config

```yaml
model: phi3:mini
temperature: 0.3  # Lower temperature for more focused, deterministic responses
system_prompt: |
  You are a careful reasoning assistant.
  Think step-by-step before answering.
inference: refined_reflect
enable_tools: true
contexts:
  - name: default
    system_prompt: "Default conversation mode"
  - name: coding
    system_prompt: "Code analysis and generation mode"
```

### Model Parameters

#### Temperature Parameter

The `temperature` parameter controls the randomness of model responses:

- **0.0**: Completely deterministic, always selects the most likely token
- **0.3-0.5**: More focused and consistent responses
- **0.7** (default): Balanced creativity and consistency
- **0.9-1.0**: More creative and varied responses

Use lower temperatures for tasks requiring consistency (e.g., code generation, factual Q&A).
Use higher temperatures for creative tasks (e.g., brainstorming, creative writing).

#### Max Tokens

The `max_tokens` value is fixed at `-1` (unlimited). The `num_predict` parameter is not sent to the Ollama API, allowing the model to generate as many tokens as it needs.

### Loading Agent Configs

```python
from pithos import OllamaAgent, ConfigManager

config_manager = ConfigManager()

# Load from registered config
agent = OllamaAgent.from_config("my-agent", config_manager)

# Or load manually
config = config_manager.get_config("my-agent", "agents")
agent = OllamaAgent.from_dict(config, config_manager)
```

## Flowchart Configuration

Flowcharts define reasoning workflows as directed graphs.

### Simple Flowchart

**File:** `configs/flowcharts/simple-flow.yaml`

```yaml
start_node: start

nodes:
  start:
    type: prompt
    prompt: "Think about the problem."
  
  reflect:
    type: prompt
    prompt: "Review your answer. Is it correct?"
  
  finish:
    type: chatoutput
    source: current_input

edges:
  - from: start
    to: reflect
    condition:
      type: AlwaysCondition
    priority: 1
  
  - from: reflect
    to: finish
    condition:
      type: AlwaysCondition
    priority: 1
```

### Conditional Branching

```yaml
start_node: analyze

nodes:
  analyze:
    type: prompt
    prompt: "Analyze the input."
  
  check_confidence:
    type: custom
    custom_code: |
      # Extract confidence from agent output
      output = context.get("last_output", "")
      if "high confidence" in output.lower():
          context["confident"] = True
      else:
          context["confident"] = False
  
  refine:
    type: prompt
    prompt: "You seem uncertain. Think more carefully."
  
  complete:
    type: chatoutput
    source: current_input

edges:
  - from: analyze
    to: check_confidence
    condition:
      type: AlwaysCondition
    priority: 1
  
  # High confidence -> complete
  - from: check_confidence
    to: complete
    condition:
      type: RegexCondition
      regex: "True"
      matchtype: search
    priority: 1
  
  # Low confidence -> refine
  - from: check_confidence
    to: refine
    condition:
      type: RegexCondition
      regex: "False"
      matchtype: search
    priority: 2
  
  - from: refine
    to: complete
    condition:
      type: AlwaysCondition
    priority: 1
```

### Loop with Count Limit

```yaml
start_node: initial

nodes:
  initial:
    type: prompt
    prompt: "Provide your initial answer."
  
  review:
    type: prompt
    prompt: "Review and improve your answer."
  
  done:
    type: chatoutput
    source: current_input

edges:
  - from: initial
    to: review
    condition:
      type: AlwaysCondition
    priority: 1
  
  # Loop up to 3 times
  - from: review
    to: review
    condition:
      type: CountCondition
      limit: 3
    priority: 1
  
  # Then finish
  - from: review
    to: done
    condition:
      type: AlwaysCondition
    priority: 2
```

### Node Types

#### Prompt Node
Standard LLM prompting:
```yaml
type: prompt
prompt: "Your instruction here"
```

#### Custom Code Node
Execute Python code:
```yaml
type: custom
custom_code: |
  # Access context and modify it
  last_output = context.get("last_output", "")
  context["processed"] = last_output.upper()
```

**Available in code:**
- `context`: The execution state dictionary
- Standard Python libraries

#### Input Node
Get user input:
```yaml
type: chatinput
prompt_message: "Enter your response:"
save_to: user_input
```

#### Output Node
Display a message:
```yaml
type: chatoutput
source: current_input
```

#### Tool Call Node
Execute a command-line tool:
```yaml
type: toolcall
command: "python --version"
save_to: version_result
```

### Condition Types

#### Always Condition
Always allows traversal:
```yaml
condition:
  type: AlwaysCondition
```

#### Count Condition
Traverse N times, then block:
```yaml
condition:
  type: CountCondition
  limit: 5
```

#### Regex Condition
Match a regex pattern against the current input:
```yaml
condition:
  type: RegexCondition
  regex: "error"
  matchtype: search
```

### Loading Flowcharts

```python
from pithos import Flowchart, ConfigManager

config_manager = ConfigManager()

# Load from registered config
flowchart = Flowchart.from_registered("simple-flow", config_manager)

# Or from file
flowchart = Flowchart.from_yaml("path/to/flow.yaml", config_manager)

# Execute with message-based routing
result = flowchart.run_message_based(
    initial_data="input data",
    max_steps=100
)
print(f"Output: {result['messages']}")
```

## Tool Configuration

Tools are automatically discovered from the system PATH. Configure filtering:

**File:** `configs/tools/tool_config.yaml`

```yaml
# Only allow these tools (if specified)
include:
  - python
  - git
  - ls
  - dir

# Block these tools (takes precedence over include)
exclude:
  - rm
  - del
  - format
  - shutdown

# Execution limits
timeout: 30  # seconds
max_output_size: 10240  # bytes
```

### Tool Discovery

```python
from pithos import ToolRegistry, ConfigManager

config_manager = ConfigManager()
tool_registry = ToolRegistry(config_manager)

# Refresh tool registry
tool_registry.refresh()

# List all tools (returns list of tool name strings)
tools = tool_registry.list_tools()
for name in tools:
    print(name)

# Get specific tool
tool = tool_registry.get_tool("python")
```

### Using Tools with Agents

```python
from pithos import OllamaAgent, ConfigManager

config_manager = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")

# Enable tools
agent.enable_tools(config_manager)

# Agent will now auto-execute tool calls
response = agent.send("What version of Python is installed?")
# Agent responds: runcommand("python --version")
# Tool executes automatically
# Agent sees result: Python 3.11.0
# Agent responds: "You have Python 3.11.0 installed."
```

## Programmatic Configuration

You can also create configurations dynamically:

### Create Agent Programmatically

```python
from pithos import OllamaAgent, AgentContext

agent = OllamaAgent("glm-4.7-flash:latest")
agent.create_context("my-context", system_prompt="You are helpful.")
```

### Create Flowchart Programmatically

```python
from pithos import Flowchart, ConfigManager
from pithos.conditions import AlwaysCondition

config_manager = ConfigManager()
flowchart = Flowchart(config_manager)

# Add nodes
flowchart.add_node("start", type="prompt", prompt="Begin analysis")
flowchart.add_node("finish", type="output", message="Done")

# Add edge
flowchart.add_edge("start", "finish", AlwaysCondition, priority=1)

# Set start
flowchart.set_start_node("start")

# Save to config
flowchart.register("my-flowchart")
```

## Best Practices

### Agent Configuration

1. **Use descriptive names**: `llama-coder` instead of `agent1`
2. **Version your prompts**: Keep old configs when experimenting
3. **Separate concerns**: Different configs for different tasks
4. **Document prompts**: Add comments in YAML for complex setups

### Flowchart Design

1. **Keep nodes focused**: Each node should do one thing
2. **Use priorities**: Lower number = higher priority
3. **Limit loops**: Use count conditions to prevent infinite loops
4. **Test incrementally**: Start simple, add complexity gradually
5. **Name meaningfully**: Node names should indicate their purpose

### Tool Configuration

1. **Principle of least privilege**: Only include necessary tools
2. **Test tools**: Use `pithos-tools test` before enabling
3. **Set appropriate limits**: Adjust timeout based on expected usage
4. **Review regularly**: Remove unused tools from include list

### Security

1. **Never use `eval()` on untrusted flowchart YAML**
2. **Review custom code nodes carefully**
3. **Restrict tool access** via include/exclude lists
4. **Use timeouts** to prevent hung processes
5. **Validate inputs** in custom code nodes

## Configuration Examples

See the `configs/` directory for real-world examples:

- `configs/agents/llama-structured-reflect.yaml` - Structured reasoning
- `configs/flowcharts/refined_reflect.yaml` - Multi-stage reflection
- `configs/flowcharts/teacher_student.yaml` - Two-agent interaction
- `configs/tools/tool_config.yaml` - Tool security settings

## Troubleshooting

### Config Not Found

```python
# Check available configs
config_manager = ConfigManager()
print(config_manager.get_registered_agent_names())
print(config_manager.get_registered_flowchart_names())
```

### Flowchart Won't Execute

- Verify `start_node` is set and valid
- Check all edges have valid `from` and `to` nodes
- Ensure at least one condition can be satisfied

### Tool Not Available

```bash
# List available tools
pithos-tools list

# Refresh cache
pithos-tools refresh

# Check tool config
cat configs/tools/tool_config.yaml
```

### Invalid YAML Syntax

Use a YAML validator or Python to check:

```python
import yaml
with open("configs/agents/my-agent.yaml") as f:
    config = yaml.safe_load(f)  # Raises exception if invalid
```
