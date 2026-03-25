# TheTeam

TheTeam is a local-first LLM agent coordination and development suite. It includes:
- **pithos**: An agentic LLM interaction framework for managing models, contexts, and flowchart-driven workflows
- **TheTeam**: Modern web interface for agent coordination, drag-and-drop flowchart interaction, and real-time workflow execution

TODO:
[![PyPI - Version](https://img.shields.io/pypi/v/theteam.svg)](https://pypi.org/project/theteam)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/theteam.svg)](https://pypi.org/project/theteam)

-----

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Testing](#testing)
- [CLI Commands](#cli-commands)
- [License](#license)

## Installation

### From Source (Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dane Howard/theteam.git
   cd theteam
   ```

2. **Create and activate virtual environment:**
   
   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   
   **Linux/macOS:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Install test dependencies (optional):**
   ```bash
   pip install -e ".[test]"
   ```

5. **Install benchmark dependencies (optional):**
   ```bash
   pip install -e ".[benchmark]"
   ```

### From PyPI (Coming Soon)

```bash
pip install theteam
```

## Quick Start

### Interactive Demo

Run the interactive demo to chat with an LLM agent:

```bash
pithos-demo
```

This provides a simple interface to:
- Chat with a default model (llama3.2:3b)
- Select from registered agent configurations
- Use flowcharts to guide agent reasoning

### Chat with an Agent

```bash
pithos-agent chat llama3.2:3b
```

Or use a registered agent config:

```bash
pithos-agent chat llama-structured-reflect
```

With a flowchart:

```bash
pithos-agent chat llama3.2:3b --flowchart simple_reflect
```

## Features

### pithos Framework

- **Agent Management**: Create, configure, and manage LLM agents
- **Context Management**: Multiple contexts per agent with copy/share capabilities
- **Flowchart Execution**: Guide agent reasoning through configurable flowcharts
- **Message-Based Routing**: Advanced data flow with explicit message passing between nodes
- **Tool Calling**: Enable agents to execute CLI commands dynamically
- **Conditions**: Define conditional logic for flowchart branching
- **Configuration**: YAML-based configuration for agents, flowcharts, and conditions
- **Serialization**: Save and load agent states, contexts, and flowcharts
- **Memory System**: Vector database for persistent knowledge storage and retrieval
- **Conversation History**: Persistent per-agent conversation logging with full-text and semantic (RAG) search
- **Structured Logging**: Runtime diagnostics via Python's `logging` module throughout all library modules

### Conversation History

pithos persists every message exchanged between an agent and users, enabling later retrieval by **full-text search** (SQLite FTS5, always available) or **semantic/RAG search** (ChromaDB, when installed). Messages can be annotated with arbitrary string tags for filtered retrieval.

```python
from pithos import OllamaAgent

agent = OllamaAgent("llama3.2")
agent.enable_history("./data/conversations")

response = agent.send("Fix the authentication error")
agent.tag_current_message(["important", "bug-fix"])

# Later: search across stored history
results = agent.search_history("authentication error", tags=["important"])
for r in results:
    print(f"[{r.match_type}] {r.message.role}: {r.message.content}")
```

**Features:**
- SQLite FTS5 full-text search (no extra dependencies)
- Optional ChromaDB semantic/vector search (falls back gracefully when unavailable)
- Tag messages for filtered retrieval (`important`, `bug-fix`, etc.)
- Search within current session or across all sessions for an agent
- Configurable persist directory; sessions identified by UUID

### Memory System

pithos includes a vector database-based memory system for building persistent knowledge bases organized by categories. Agents can store and retrieve information across conversations, enabling them to build domain knowledge over time.

**Features:**
- Vector database using ChromaDB for semantic search
- Organize knowledge into categories (like individual RAG databases)
- Store and retrieve with relevance scoring
- Persistent local storage
- Export/import for backup and migration
- Metadata filtering for precise queries

**Quick Start:**

```python
from pithos import OllamaAgent, ConfigManager

# Create agent and enable memory
config_manager = ConfigManager()
agent = OllamaAgent("llama3.2")
agent.enable_memory(config_manager)

# Agent can now use memory operations
agent.send("Remember that Python was created by Guido van Rossum")
# Agent: storemem(facts, "Python was created by Guido van Rossum")

agent.send("Who created Python?")
# Agent: retrievemem(facts, "Python creator")
# System provides stored knowledge
# Agent responds with the answer
```

**CLI Usage:**

```bash
# Store knowledge
pithos-memory store facts "Python is a high-level language"

# Retrieve knowledge
pithos-memory retrieve facts "What is Python?"

# List categories
pithos-memory list

# Get category info
pithos-memory info facts

# Export/import for backup
pithos-memory export facts backup.json
pithos-memory import backup.json
```

**Memory Operations:**
- `storemem(category, "content")` - Store knowledge in a category
- `retrievemem(category, "query")` - Search for relevant knowledge

**Use Cases:**
- Build domain knowledge bases
- Remember user preferences and context
- Cache frequently accessed information
- Create searchable code snippet libraries
- Store conversation history semantically

**See:** [Memory Tool Guide](docs/MEMORY.md) for detailed documentation.

### Message-Based Routing

pithos uses **message-based routing** as the default execution model. Nodes communicate by passing explicit messages, enabling sophisticated data flow patterns.

**Features:**
- Nodes produce messages as output instead of modifying shared state
- Messages routed separately along edges to target nodes
- Nodes wait for all required inputs before executing
- Support for merge/join patterns with multiple inputs
- Explicit data dependencies and provenance tracking

**Quick Start:**

```python
from pithos import Flowchart, ConfigManager, AlwaysCondition

config_manager = ConfigManager()
flow = Flowchart(config_manager)

# Add nodes with explicit inputs/outputs
flow.add_node("start", type="prompt", 
              prompt="Start: {default}",
              inputs=["default"], 
              outputs=["default"])

flow.add_node("process", type="prompt",
              prompt="Process {input1} and {input2}",
              inputs=["input1", "input2"],  # Requires both inputs
              outputs=["result"])

# Add edges with message routing
flow.add_edge("start", "process", AlwaysCondition,
              output_key="default", input_key="input1")

# Run flowchart (message-based routing is the default)
result = flow.run_message_based(initial_data="Hello")
print(f"Steps: {result['steps']}, Messages: {len(result['messages'])}")
```
```

**Patterns:**
- **Linear Pipeline**: Sequential processing with explicit data flow
- **Merge/Join**: Multiple inputs converging to single node
- **Conditional Branching**: Route messages based on content
- **Fan-out**: Single source to multiple targets

**See:** [Message Routing Guide](docs/MESSAGE_ROUTING.md) for detailed documentation.

### Input and Output Nodes

pithos flowcharts now support dedicated **Input and Output nodes** for handling data flow into and out of flowcharts. Every flowchart requires at least one input node and one output node.

**Features:**
- **ChatInputNode**: Interactive user input via chat interface (default)
- **ChatOutputNode**: Display results to users via chat (default)
- **FileInputNode**: Read data from files with dynamic paths
- **FileOutputNode**: Write results to files (write or append mode)
- **Automatic insertion**: Missing I/O nodes are added automatically
- **Template support**: Use `{placeholders}` in paths and formatting

**Quick Start:**

```python
from pithos import Flowchart, ConfigManager

config_manager = ConfigManager()

# Flowchart with explicit I/O nodes
flowchart_data = {
    "nodes": {
        "file_in": {
            "type": "fileinput",
            "file_path": "data/input.txt",
            "save_to": "content"
        },
        "analyze": {
            "type": "prompt",
            "prompt": "Analyze: {content}"
        },
        "file_out": {
            "type": "fileoutput",
            "file_path": "data/output.txt",
            "source": "current_input"
        }
    },
    "edges": [
        {"from": "file_in", "to": "analyze"},
        {"from": "analyze", "to": "file_out"}
    ],
    "start_node": "file_in"
}

flowchart = Flowchart.from_dict(flowchart_data, config_manager)
```

**Automatic I/O Insertion:**

If you don't specify I/O nodes, they're added automatically:

```yaml
# Simple flowchart without explicit I/O
nodes:
  process:
    type: prompt
    prompt: "Process: {current_input}"
```

Becomes:

```yaml
# With automatic I/O nodes
nodes:
  __auto_chat_input__:
    type: chatinput
  process:
    type: prompt
    prompt: "Process: {current_input}"
  __auto_chat_output__:
    type: chatoutput
```

**See:** [Flowchart Guide](docs/FLOWCHARTS.md) for detailed I/O node documentation.

### Tool Calling

pithos agents can execute command-line tools to interact with the system. Tools are discovered from the system PATH and filtered via configuration.

**Features:**
- Dynamic CLI tool discovery from system PATH
- Configurable include/exclude lists for security
- Automatic description extraction from `--help` output
- Timeout and output size limits
- Integration with agent conversations and flowcharts

**Quick Start:**

```python
from pithos import OllamaAgent, ConfigManager

# Create agent and enable tools
config_manager = ConfigManager()
agent = OllamaAgent("llama3.2")
agent.enable_tools(config_manager)

# Agent can now use tools in responses
response = agent.send("What version of Python is installed?")
# Agent will respond with: runcommand("python --version")
# Tool executes automatically and result is added to conversation
```

**CLI Usage:**

```bash
# List available tools
pithos-tools list

# Show tool details
pithos-tools show python

# Test tool execution
pithos-tools test python --version

# Refresh tool cache
pithos-tools refresh
```

**Configuration:**

Tools are configured in `configs/tools/tool_config.yaml`:

```yaml
# Tool filtering mode: 'include', 'exclude', or 'all'
mode: include

# List of allowed tools
include:
  - python
  - git
  - curl
  
# List of forbidden tools
exclude:
  - rm
  - del
  - shutdown

# Manual tool descriptions
descriptions:
  python: "Python interpreter for running scripts"
  git: "Version control system"
```

**Flowchart Integration:**

Use `ToolCallNode` in flowcharts for explicit tool execution:

```yaml
nodes:
  CheckPython:
    type: toolcall
    command: "python --version"
    save_to: "python_version"
    error_handling: "continue"
  
  AnalyzeVersion:
    type: prompt
    prompt: "Python version: {python_version[stdout]}"
```

**Security:**
- Only whitelisted tools can be executed
- Configurable timeouts prevent hanging
- Output size limits prevent memory exhaustion
- Command injection protection via subprocess

### Core Components

- **OllamaAgent**: Local LLM agent using Ollama
- **Flowchart**: Graph-based workflow execution engine
- **FlowNode**: Customizable nodes for prompts and custom logic
- **Conditions**: Configurable conditions (Always, Count, Regex, Custom)
- **ConfigManager**: Centralized configuration management

## Testing

All tests use pytest and are located in the `tests/` directory.

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=pithos --cov-report=html
```

### Run Specific Test Module

```bash
pytest tests/test_agent.py -v
```

### Test Statistics

- **601 tests** covering all core modules
- All tests passing
- Comprehensive coverage of:
  - Agent context management (copy vs share)
  - Flowchart execution and branching
  - Configuration loading/saving
  - Conversation history storage and search
  - Edge cases and error handling

## CLI Commands

### pithos-demo

Interactive demo for exploring pithos capabilities.

```bash
pithos-demo
```

### pithos-agent

Agent management and interactive chat.

```bash
# Chat with an agent
pithos-agent chat <agent_config>

# Register an agent configuration
pithos-agent register <config_file> --name <name>
```

### pithos-config

List and manage configurations.

```bash
pithos-config
```

Displays:
- Registered namespaces
- Registered agents
- Registered conditions
- Registered flowcharts

### pithos-conditions

Condition management CLI.

```bash
# Register a condition
pithos-conditions register <config_file> --name <name>

# Show a condition
pithos-conditions show <condition_name>

# Test a condition
pithos-conditions test <condition_name> <test_string>
```

### pithos-tools

Tool calling management CLI.

```bash
# List available tools
pithos-tools list

# Show tool details
pithos-tools show <tool_name>

# Test tool execution
pithos-tools test <tool_name> <args>

# Refresh tool cache
pithos-tools refresh
```

### pithos-memory

Memory system management CLI for vector database knowledge storage.

```bash
# Store knowledge in a category
pithos-memory store <category> <content>

# Retrieve relevant knowledge
pithos-memory retrieve <category> <query> [--limit N]

# List all categories
pithos-memory list

# Get category information
pithos-memory info <category>

# Delete a specific entry
pithos-memory delete <category> <entry_id>

# Export category to JSON
pithos-memory export <category> <output_file>

# Import category from JSON
pithos-memory import <input_file> [--category <name>]
```

**Examples:**
```bash
# Store a fact
pithos-memory store python "Python uses significant whitespace"

# Search for information
pithos-memory retrieve python "What is Python's syntax?"

# Backup a category
pithos-memory export python python_knowledge.json
```

**See [MEMORY.md](docs/MEMORY.md) for detailed documentation.**

### pithos-benchmark

Run benchmarks to evaluate and compare LLM agents and workflows.

```bash
# Run with default configuration
pithos-benchmark

# Run with custom config
pithos-benchmark --config my_benchmark.yaml

# Run specific models only
pithos-benchmark --models model1 model2

# Run with custom number of rounds
pithos-benchmark --rounds 5

# Generate report from existing results
pithos-benchmark report --path ./results/2024-03-12-Multi-Benchmark

# List available configs
pithos-benchmark list-configs

# List available datasets
pithos-benchmark list-datasets
```

**Features:**
- YAML-based benchmark configuration
- Support for multiple dataset types (multiple choice, free-form)
- Compare different models and flowchart workflows
- Generate performance charts and statistics
- CLI overrides for quick iterations

**See [BENCHMARKS.md](docs/BENCHMARKS.md) for detailed documentation.**

## Configuration

Configurations are stored as YAML files in the `configs/` directory:

```
configs/
  agents/          # Agent configurations
  flowcharts/      # Flowchart definitions
  conditions/      # Custom conditions (optional)
```

### Example Agent Config

```yaml
# configs/agents/my-agent.yaml
default_model: llama3.2:3b
system_prompt: "You are a helpful assistant."
```

### Example Flowchart Config

```yaml
# configs/flowcharts/simple.yaml
start_node: prompt1
nodes:
  prompt1:
    type: prompt
    prompt: "What is your question?"
    extraction:
      answer: ".*"
  prompt2:
    type: prompt
    prompt: "Follow-up: {answer}"
edges:
  - from: prompt1
    to: prompt2
    condition:
      type: always
```

## Web Interface (TheTeam)

TheTeam provides a modern web interface for managing agents, creating flowcharts visually, and interacting with LLMs in real-time.

### Features

- **Visual Flowchart Editor**: Interactive drag-and-drop interface using ReactFlow
  - Create and edit workflows visually with modern node styling
  - 5 edge types with color coding (default, conditional, loop, error, success)
  - Real-time execution visualization with node highlighting
  - Import/export YAML configurations
  - Live node output inspection
  - Edit buttons on nodes (visible on hover)
  - Variable name display on hover next to connection points
  - Interactive edge type switching by clicking edge labels

- **Chat Interface**: Real-time conversations with agents
  - Multiple conversation management
  - Dynamic agent switching mid-conversation
  - Base Model mode for direct model access without agent configuration
  - Persistent conversation history
  - WebSocket-based real-time streaming responses

- **Agent Configuration**: Comprehensive agent management
  - Create and edit agents with full parameter control in the UI
  - Configure all generation parameters (temperature, max tokens, top k, top p, repeat penalty)
  - Multi-select tool assignment interface
  - Chain-of-thought flowchart selection
  - Tool auto-loop and iteration settings
  - Memory system enable/disable
  - Test agents with sample prompts
  - Save configurations persistently to file or runtime

- **Robust Networking**: Production-ready WebSocket implementation
  - Automatic reconnection with exponential backoff
  - Connection health monitoring with ping/pong
  - Seamless error recovery
  - Real-time state synchronization

### Starting the Web Server

TheTeam provides multiple ways to start the web interface:

#### Option 1: WebGUI Mode (Recommended for Development)

Start both backend and frontend with a single command:

```bash
theteam webgui
```

This will:
- ✅ Start the backend server at `http://localhost:5000`
- ✅ Start the frontend dev server at `http://localhost:3000` (with hot-reload)
- ✅ Display live status updates for both processes (refreshes every 5 seconds)
- ✅ Handle Ctrl+C gracefully to shut down both processes

The status display automatically refreshes in place, showing only the current status to keep your terminal clean.

**Debug Mode:** Enable verbose error output and detailed process information:

```bash
theteam webgui --debug
```

In debug mode, you'll see:
- Command line arguments for both processes
- Process IDs (PIDs)
- Full tracebacks for any errors
- All output from both backend and frontend (not just important messages)
- Exit codes and remaining output when processes fail
- Status updates will accumulate (not refresh in place) for easier debugging

**Note:** Press Ctrl+C once to initiate graceful shutdown, or twice to force stop.

#### Option 2: Backend Server Only

Run just the backend server (requires frontend to be built):

```bash
# Start with default settings
theteam server

# Or with custom settings
theteam server --host 127.0.0.1 --port 5000 --debug
```

#### Option 3: Legacy Command (Backward Compatible)

```bash
# Start the server with old syntax
theteam --host 127.0.0.1 --port 5000
```

Then open `http://localhost:5000` in your browser.

### Development Mode (Manual Control)

For granular control during development:

```bash
# Terminal 1: Start backend
theteam server --debug

# Terminal 2: Start frontend dev server
cd frontend
npm install
npm run dev
```

The frontend dev server runs on `http://localhost:3000` with hot-reload enabled.

### Building the Frontend

```bash
cd frontend
npm install
npm run build
```

This builds the optimized static files into `src/theteam/static/` which are served by the Flask backend.

**See [frontend/README.md](frontend/README.md) for detailed frontend documentation.**

## Development

### Project Structure

```
src/
  pithos/              # LLM agent framework
    agent/               # Agent class and core LLM interaction logic
    team/                # AgentTeam multi-agent coordination
    tools/               # Structured tool calling system
    conditions.py        # Flowchart edge condition evaluation
    config_manager.py    # YAML config loading with env-var support
    context.py           # Agent context/conversation management
    flowchart.py         # Flowchart execution engine
    flownode.py          # FlowNode types
    message.py           # Message and MessageRouter
    validation.py        # Flowchart/node config validation
  theteam/               # Web interface
    server.py            # Flask application & WebSocket server
    api/                 # REST API blueprints
    services/            # Business logic layer
    static/              # Built frontend assets
tests/                   # Comprehensive test suite
configs/                 # YAML configurations
benchmarks/              # Benchmark suites
frontend/                # React web interface
  src/                   # Frontend source code
    components/          # UI components
    pages/               # Main pages
    store/               # State management
```

### Running Tests

Before running tests, ensure the virtual environment is activated:

```bash
# Windows
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Then run pytest:

```bash
pytest -v
```

## License

`theteam` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
