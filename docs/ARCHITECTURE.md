# TheTeam Architecture

## Overview

TheTeam is a local-first LLM agent coordination suite consisting of two main components:

1. **pithos**: A Python-based agent framework for managing LLM interactions, contexts, and flowchart-driven workflows
2. **TheTeam**: A modern React-based web interface for visual agent coordination, workflow execution, and real-time chat

This document provides a high-level overview of the architecture, key components, and design patterns.

## Project Structure

```
theteam/
├── src/
│   ├── pithos/              # Core agent framework
│   │   ├── agent/             # Agent class and core LLM interaction logic
│   │   │   └── agent.py       # Agent, OllamaAgent, EXLAgent, LlamacppAgent
│   │   ├── team/              # Multi-agent coordination
│   │   │   └── agent_manager.py  # AgentTeam, TeamContext
│   │   ├── tools/             # Structured tool calling system
│   │   │   ├── registry.py    # ToolRegistry — discovers and caches tools
│   │   │   ├── executor.py    # ToolExecutor — runs tool commands
│   │   │   ├── extractor.py   # Parses tool calls from agent output
│   │   │   ├── memory_ops.py  # Memory operation helpers
│   │   │   ├── memory_tool.py # ChromaDB-backed vector memory
│   │   │   ├── models.py      # Tool data models
│   │   │   └── cli.py         # CLI entry point for tools
│   │   ├── conditions.py      # Flowchart edge condition evaluation
│   │   ├── config_manager.py  # YAML config loading with env-var support
│   │   ├── context.py         # Agent context/conversation management
│   │   ├── flowchart.py       # Flowchart execution engine (message-based)
│   │   ├── flownode.py        # FlowNode types
│   │   ├── message.py         # Message and MessageRouter
│   │   ├── validation.py      # Flowchart/node config validation
│   │   ├── utils.py           # Shared utility functions
│   │   └── __main__.py        # CLI entry point
│   └── theteam/               # Web interface (Flask + React)
│       ├── server.py          # Flask application & WebSocket server
│       ├── api/               # REST API blueprints
│       ├── services/          # Business logic layer
│       └── static/            # Built frontend assets
├── frontend/               # React + TypeScript frontend
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Main page components
│   │   ├── store/         # Zustand state stores
│   │   └── lib/           # Utility functions
│   └── package.json
├── configs/                # YAML configurations
│   ├── agents/            # Agent configs
│   ├── flowcharts/        # Flowchart definitions
│   └── tools/             # Tool configurations
├── tests/                 # Test suite (mirrors src/)
├── benchmarks/            # Benchmarking tools
└── docs/                  # Documentation

```

## Core Design Principles

1. **Local-First**: All operations run locally; no cloud dependencies
2. **Configuration-Driven**: Agents, flowcharts, and tools configured via YAML
3. **Modular**: Components are loosely coupled and independently testable
4. **Extensible**: Easy to add new node types, conditions, and tools
5. **Type-Safe**: Type hints throughout for better IDE support and reliability
6. **Observable**: Structured logging via Python's `logging` module throughout all library modules; `DEBUG` level activated when `--debug` is passed

## System Architecture

### Layer 1: LLM Interface

The foundation is the `OllamaAgent` class that wraps the Ollama Python client:

```
┌─────────────────────────────┐
│     OllamaAgent             │
│  - default_model            │
│  - send()                   │
│  - enable_tools()           │
└─────────────────────────────┘
```

**Key Responsibilities:**
- Communicate with local Ollama models
- Manage multiple conversation contexts
- Handle tool calling integration
- Serialize/deserialize agent state

### Layer 2: Context and History Management

Each agent can maintain multiple independent `AgentContext` instances:

```
┌─────────────────────────────────────────┐
│           AgentContext                  │
│  - name                                 │
│  - system_prompt                        │
│  - message_history                      │
│  - flowchart (optional)                 │
│  - copy() / share semantics             │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         ConversationStore               │
│  - SQLite FTS5 full-text search         │
│  - ChromaDB semantic/vector search      │
│  - Per-session scoping                  │
│  - Tag-based filtering                  │
└─────────────────────────────────────────┘
```

**Context Capabilities:**
- Independent conversation histories
- Per-context system prompts
- Optional flowchart attachment
- Copy (deep) or share (reference) semantics

**History Capabilities (`enable_history`):**
- Every sent/received message persisted to SQLite
- Full-text search via SQLite FTS5 (always available)
- Semantic/RAG search via ChromaDB (optional, degrades gracefully)
- Tag messages with `tag_current_message(["important", "bug-fix"])`
- Search current session or all sessions for an agent

### Layer 3: Flowchart Execution

Flowcharts provide structured reasoning paths using a directed graph:

```
┌──────────────────────────────────────┐
│         Flowchart                    │
│  - graph: MultiDiGraph               │
│  - start_node                        │
│  - run_message_based()               │
│  - step_message_based()              │
└──────────────────────────────────────┘
              │
              ├──────────────┬──────────────┬──────────────┐
              │              │              │              │
        PromptNode      CustomNode   ChatInputNode  ChatOutputNode
```

**Node Types:**
- `PromptNode`: Standard LLM prompting
- `CustomNode`: Execute arbitrary Python code (type key: `"custom"`)
- `ToolCallNode`: Execute command-line tools (type key: `"toolcall"`)
- `TextParseNode`: Extract/set state variables (type key: `"textparse"`)
- `AgentPromptNode`: Prompt a named agent (type key: `"agentprompt"`)
- `GetHistoryNode`: Read agent conversation history (type key: `"gethistory"`)
- `SetHistoryNode`: Write agent conversation history (type key: `"sethistory"`)
- `ChatInputNode`: Accept interactive user input (type key: `"chatinput"`)
- `ChatOutputNode`: Display results to user (type key: `"chatoutput"`)
- `FileInputNode`: Read data from a file (type key: `"fileinput"`)
- `FileOutputNode`: Write data to a file (type key: `"fileoutput"`)

**Conditional Transitions:**
- `AlwaysCondition`: Always proceeds
- `CountCondition`: Traverse after N times
- `RegexCondition`: Match regex pattern against state
- Custom conditions via registration

### Layer 4: Tool Calling System

Agents can discover and execute command-line tools dynamically:

```
┌─────────────────────────────────────────┐
│       ToolRegistry                      │
│  - discover_tools()                     │
│  - get_tool(name)                       │
│  - filter by include/exclude lists      │
└─────────────────────────────────────────┘
              │
              v
┌─────────────────────────────────────────┐
│       ToolExecutor                      │
│  - execute(tool, args)                  │
│  - timeout enforcement                  │
│  - output size limits                   │
└─────────────────────────────────────────┘
```

**Key Features:**
- Automatic tool discovery from system PATH
- Configurable security filters
- Description extraction from `--help` output
- Safe execution with timeouts and output limits

### Layer 5: Configuration Management

The `ConfigManager` provides centralized YAML-based configuration:

```
configs/
├── agents/              # Agent definitions
├── flowcharts/          # Workflow definitions
├── tools/               # Tool settings
└── conditions/          # Custom condition configs (future)
```

**Loading Pattern:**
```python
config_manager = ConfigManager()
agent = OllamaAgent.from_config("llama-structured-reflect", config_manager)
flowchart = Flowchart.from_registered("simple_reflect", config_manager)
```

## Data Flow

### Basic Conversation Flow

```
User Input
    │
    v
┌─────────────────┐
│  AgentContext   │
│ add_message()   │
└─────────────────┘
    │
    v
┌─────────────────┐
│  OllamaAgent    │
│   send()        │
└─────────────────┘
    │
    v
┌─────────────────┐
│  Ollama API     │
│  chat()         │
└─────────────────┘
    │
    v
┌─────────────────┐
│  AgentContext   │
│ add_message()   │
└─────────────────┘
    │
    v
Response to User
```

### Flowchart-Guided Conversation

```
User Input
    │
    v
┌──────────────────────────────┐
│   Flowchart                  │
│   run_message_based()        │
│   step_message_based()       │
└──────────────────────────────┘
    │
    v
┌──────────────────────────────┐
│   node.execute_with_         │
│   messages(state, router)    │
└──────────────────────────────┘
    │
    v
┌──────────────────┐
│  Agent.send()    │
│  with prompt     │
└──────────────────┘
    │
    v
┌──────────────────┐
│  Condition Check │
│  on all edges    │
└──────────────────┘
    │
    v
┌──────────────────┐
│  Next Node       │
│  or Finish       │
└──────────────────┘
```

### Tool Calling Flow

```
User: "What version of Python is installed?"
    │
    v
Agent Response: runcommand("python --version")
    │
    v
┌──────────────────┐
│  Tool Parsing    │
│  Extract command │
└──────────────────┘
    │
    v
┌──────────────────┐
│ ToolExecutor     │
│ execute()        │
└──────────────────┘
    │
    v
┌──────────────────┐
│ System Execution │
│ (subprocess)     │
└──────────────────┘
    │
    v
Result added to conversation
    │
    v
Agent sees result & responds naturally
```

## Key Design Patterns

### 1. Factory Pattern (FlowNode Creation)

```python
def create_node(node_type: str, config: dict) -> Optional[FlowNode]:
    """Factory for creating flow nodes by type."""
    node_classes = {
        "prompt": PromptNode,
        "custom": CustomNode,
        "toolcall": ToolCallNode,
        "textparse": TextParseNode,
        "agentprompt": AgentPromptNode,
        "gethistory": GetHistoryNode,
        "sethistory": SetHistoryNode,
        "chatinput": ChatInputNode,
        "chatoutput": ChatOutputNode,
        "fileinput": FileInputNode,
        "fileoutput": FileOutputNode,
    }
    return node_classes.get(node_type, lambda x: None)(config)
```

### 2. Strategy Pattern (Conditions)

Each condition implements a common interface:

```python
class Condition:
    def is_open(self, state: dict) -> bool:
        """Check if condition allows traversal."""
        ...
    
    def traverse(self, state: dict) -> None:
        """Update state when condition is traversed."""
        ...

# Pre-built instance — use without parentheses:
# AlwaysCondition (not AlwaysCondition())
AlwaysCondition = Condition(condition=lambda x: True, ...)
```

### 3. Registry Pattern (Tools, Conditions, Agents)

Central registries provide lookup and validation:

```python
# Tool Registry
tool_registry = ToolRegistry(config_manager)
tool_registry.refresh()           # re-discovers tools in memory
available_tools = tool_registry.list_tools()  # returns list[str] of tool names

# Agent Registry (via ConfigManager)
agent_names = config_manager.get_registered_agent_names()
agent = OllamaAgent.from_config(agent_names[0], config_manager)
```

### 4. State Machine Pattern (Flowchart Execution)

Flowcharts use message-based execution:

```python
# Execute flowchart with message routing
result = flowchart.run_message_based(
    initial_data="input",
    max_steps=100,
    history_window=0,       # rolling window size; 0 = unlimited
    on_progress=None,       # optional ProgressEvent callback
)
print(f"Output: {result['messages']}")
```

## Extension Points

### Adding New Node Types

1. Subclass `FlowNode` in `flownode.py`
2. Implement `_execute(context)` method; the `message_router` is passed to the higher-level `execute_with_messages()` wrapper, not directly to `_execute()`
3. Add to factory in `create_node()`

### Adding New Conditions

1. Subclass `Condition` in `conditions.py`
2. Implement `is_open()` and `traverse()`
3. Register in `ConditionManager`

### Adding Custom Tools

Tools are discovered automatically from PATH. To restrict:
1. Edit `configs/tools/tool_config.yaml`
2. Add to `include` or `exclude` lists
3. Run `pithos-tools refresh`

## Security Considerations

### Tool Execution

- Tools filtered by include/exclude lists
- Execution timeout (default: 30 seconds)
- Output size limits (default: 10KB)
- No shell expansion or piping (direct execution only)

### Custom Code Nodes

- Execute arbitrary Python via `exec()`
- **WARNING**: No sandboxing currently implemented
- Use only with trusted flowcharts
- Future: Consider `RestrictedPython` or similar

## Performance Characteristics

### Memory Usage

- Message histories grow linearly with conversation length
- Flowchart graphs stored in memory (typically small)
- Tool registry cached in memory (no persistent disk cache)

### Computation

- Network I/O to local Ollama server dominates latency
- Condition checking is O(1) per condition
- Graph traversal is O(edges) per step

## Testing Strategy

Tests are organized to mirror the source structure:

```
tests/
├── test_agent.py              # Agent and context tests
├── test_agent_history.py      # Conversation history storage and search
├── test_agent_memory.py       # Memory system tests
├── test_agent_tools.py        # Agent tool integration tests
├── test_flowchart.py          # Flowchart execution tests
├── test_flownode.py           # Node type tests
├── test_io_nodes.py           # I/O node (file/chat) tests
├── test_conditions.py         # Condition logic tests
├── test_tools.py              # Tool calling tests
├── test_tool_extractors.py    # Tool call parsing tests
├── test_message_routing.py    # Message routing tests
├── test_config_manager.py     # Configuration tests
├── test_memory_tool.py        # Memory tool tests
├── test_validation.py         # Flowchart validation tests
├── test_server.py             # Web server and API tests
├── test_flowchart_service.py  # Flowchart service tests
└── test_benchmark_*.py        # Benchmark harness tests
```

**Test Categories:**
- Unit tests: Individual components in isolation
- Integration tests: Multi-component interactions
- Benchmarks: Performance and regression testing

## Future Enhancements

See the [GitHub Issues tracker](https://github.com/mirrord/theteam/issues) for the full roadmap. Key planned features:

1. **Enhanced Multi-Agent Teams**: Advanced coordination patterns between multiple agents
2. **RAG Integration**: Retrieval-augmented generation support
3. **Advanced Security**: Sandboxed code execution for custom nodes
4. **Distributed Systems**: Network protocols for agent communication across machines
5. **Plugin System**: External plugin architecture for custom node types

## Related Documentation

- [Configuration Guide](CONFIG.md) - How to configure agents, flowcharts, and tools
- [Tool Calling Guide](TOOL_CALLING.md) - Using the tool calling system
- [Flowchart Design](FLOWCHARTS.md) - Creating effective flowcharts
- [Web Interface Guide](WEB_INTERFACE.md) - Using the web-based interface
- [Contributing](../README.md) - Development setup and guidelines
