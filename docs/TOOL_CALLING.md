# Tool Calling Guide

This guide explains how to use pithos's tool calling system to enable agents to execute command-line tools dynamically.

## Overview

Tool calling allows LLM agents to discover and execute command-line tools from your system PATH. Agents can use tools to:

- Check system information (versions, environment)
- Execute scripts and programs
- Interact with development tools (git, npm, pip)
- Query system state
- Automate workflows

**Key Features:**
- Multiple syntax formats for reliability
- Clear error feedback for agents
- No system crashes on tool failures
- Automatic recovery and continuation

## Quick Start

### Enable Tools for an Agent

```python
from pithos import OllamaAgent, ConfigManager

config_manager = ConfigManager()
agent = OllamaAgent("llama3.2")

# Enable tool calling
agent.enable_tools(config_manager)

# Now the agent can execute tools
response = agent.send("What version of Python is installed?")
```

### Agent Tool Workflow

When tools are enabled, the agent follows this pattern:

1. **User request**: "What version of Python is installed?"
2. **Agent response**: Uses any supported format (see below)
3. **System executes**: Tool runs automatically with error handling
4. **Result added**: Output or error feedback injected into conversation
5. **Agent continues**: Even if tool fails, agent receives clear feedback and can retry

## Tool Call Syntax

pithos supports multiple syntax formats to improve reliability. Agents can use any format:

### 1. CLI-Style (Simplest)
```
RUN: python --version
EXEC: git status
TOOL: npm list
```

### 2. Function-Style
```
run(python --version)
tool(git status)
execute(npm list)
```

### 3. Bracket-Style
```
[RUN]python --version[/RUN]
<RUN>git status</RUN>
[EXEC]npm list[/EXEC]
```

### 4. Legacy (Still Supported)
```
runcommand("python --version")
runcommand('git status')
```

**All formats work identically** - use whichever is most natural for your agent or workflow.

## Tool Discovery

Tools are automatically discovered from your system PATH.

### List Available Tools

```bash
# List all discovered tools
pithos-tools list

# Show detailed info for a specific tool
pithos-tools show python
```

### Refresh Tool Cache

The tool registry is cached for performance. Refresh when you install new tools or update your PATH:

```bash
pithos-tools refresh
```

### Programmatic Discovery

```python
from pithos import ToolRegistry, ConfigManager

config_manager = ConfigManager()
tool_registry = ToolRegistry(config_manager)

# Refresh tool registry (in memory — no disk cache)
tool_registry.refresh()

# List all tools (returns list of tool name strings)
tools = tool_registry.list_tools()
for name in tools:
    print(name)

# Get specific tool metadata
tool = tool_registry.get_tool("python")
print(tool.name)
print(tool.description)
print(tool.path)
```

## Tool Configuration

Configure tool access and security settings in `configs/tools/tool_config.yaml`.

### Default Configuration

```yaml
# Only allow these tools (leave empty to allow all)
include: []

# Block these tools (takes precedence over include)
exclude:
  - rm
  - del
  - format
  - shutdown
  - poweroff
  - reboot

# Execution limits
timeout: 30  # seconds
max_output_size: 10240  # bytes (10 KB)
```

### Security Best Practices

#### Allowlist Approach (Recommended)

Only permit specific tools:

```yaml
include:
  - python
  - git
  - npm
  - pip
  - ls
  - cat
  - grep

exclude: []
```

#### Blocklist Approach

Allow most tools but block dangerous ones:

```yaml
include: []

exclude:
  # File operations
  - rm
  - del
  - format
  
  # System control
  - shutdown
  - poweroff
  - reboot
  
  # Network (optional)
  - curl
  - wget
  - nc
  
  # Compilation (optional)
  - gcc
  - make
```

#### Execution Limits

Adjust timeouts and output limits based on expected tool usage:

```yaml
# For long-running tools (e.g., builds)
timeout: 60

# For tools with verbose output
max_output_size: 102400  # 100 KB
```

## Tool Execution

### How Tools Are Executed

When an agent outputs a tool call in any supported format, pithos:

1. **Extracts** tool calls using multi-pattern matching
2. **Parses** the command string
3. **Validates** the tool is in the registry and not blocked
4. **Executes** via subprocess with timeout
5. **Captures** stdout and stderr
6. **Generates feedback** with clear error hints if failures occur
7. **Injects** result into conversation
8. **Continues** agent processing with result (even on failure)

### Execution Environment

- **Subprocess**: Tools run in isolated subprocesses
- **No shell**: Direct execution (no shell expansion or piping)
- **Current directory**: Workspace root
- **Environment**: Inherits parent process environment
- **Timeout**: Enforced per tool_config.yaml
- **Output**: Captured and truncated if too large

### Tool Result Format

Results are added to the conversation as system messages:

**Success:**
```
Tool execution: python --version
Status: ✓ Success
Exit code: 0

Output:
Python 3.11.0
```

**Failure with feedback:**
```
Tool execution: badcommand
Status: ✗ Failed
Exit code: 127

Stderr:
Command not found

💡 Hint: Tool 'badcommand' not found or not allowed.
Available tools include: python, git, npm, pip, echo, cat...
Use exact tool names from the available list.
```

### Error Handling

Tool failures **never crash the system**. Instead:

1. **Clear error messages** explain what went wrong
2. **Helpful hints** guide the agent on how to fix the issue
3. **Execution continues** allowing the agent to retry or adapt
4. **Context preserved** so agents understand what happened

Common error scenarios and feedback:

- **Tool not found**: Lists available tools
- **Invalid syntax**: Shows correct command format
- **Timeout**: Suggests simpler operations
- **Permission error**: Explains access restrictions
- **Non-zero exit**: Includes stderr output for debugging

### Error Handling

If tool execution fails:

```
Tool: nonexistent_tool --flag
Error: Tool not found in registry
```

```
Tool: python infinite_loop.py
Error: Command timed out after 10 seconds
```

## Advanced Usage

### Test Tools Before Enabling

```bash
# Test a tool execution
pithos-tools test python --version

# Test with arguments
pithos-tools test git status
```

### Tool Results in Flowcharts

Use tool calls in custom code nodes:

```yaml
nodes:
  check_env:
    type: custom
    custom_code: |
      # Store tool results in context
      import subprocess
      result = subprocess.run(
          ["python", "--version"],
          capture_output=True,
          text=True,
          timeout=5
      )
      context["python_version"] = result.stdout.strip()
  
  analyze:
    type: prompt
    prompt: "Analyze the Python version in context['python_version']"
```

Or use dedicated tool call nodes:

```yaml
nodes:
  version_check:
    type: toolcall
    command: "python --version"
    save_to: version_result
```

### Conditional Logic Based on Tool Output

```yaml
nodes:
  run_git:
    type: toolcall
    command: "git status"
    save_to: git_output
  
  check_status:
    type: custom
    custom_code: |
      output = context.get("last_output", "")
      if "nothing to commit" in output:
          context["git_clean"] = True
      else:
          context["git_clean"] = False
  
  commit_changes:
    type: prompt
    prompt: "There are uncommitted changes. Review them."
  
  continue:
    type: prompt
    prompt: "Repository is clean. Proceed."

edges:
  - from: run_git
    to: check_status
    condition: {type: AlwaysCondition}
  
  - from: check_status
    to: commit_changes
    condition:
      type: RegexCondition
      regex: "False"
      matchtype: search
  
  - from: check_status
    to: continue
    condition:
      type: RegexCondition
      regex: "True"
      matchtype: search
```

### Multiple Tool Calls

Agents can chain tool calls:

```python
agent.send("""
Check the following:
1. Python version
2. Git status
3. Current directory contents
""")

# Agent may respond with multiple runcommand() calls:
# runcommand("python --version")
# runcommand("git status")
# runcommand("ls")
```

Each tool executes in sequence and results are added to conversation.

### Programmatic Execution

Execute tools directly without agents:

```python
from pithos.tools import ToolExecutor, ToolRegistry, ConfigManager

config_manager = ConfigManager()
tool_registry = ToolRegistry(config_manager)
tool = tool_registry.get_tool("python")

executor = ToolExecutor(
    timeout=30,
    max_output_size=10240
)

# Run a complete command string
result = executor.run("python --version", tool_registry)
print(result.stdout)
print(result.stderr)
print(result.exit_code)
```

## CLI Commands

### List Tools

```bash
pithos-tools list
```

Output:
```
Available tools:
  python: Python interpreter
  git: Git version control
  npm: Node package manager
  pip: Python package installer
  ...
```

### Show Tool Details

```bash
pithos-tools show python
```

Output:
```
Tool: python
Path: /usr/bin/python3
Description: Python interpreter
Arguments: [command] [options]
```

### Test Tool Execution

```bash
pithos-tools test python --version
```

Output:
```
Executing: python --version
Output: Python 3.11.0
Return code: 0
Execution time: 0.12s
```

### Refresh Cache

```bash
pithos-tools refresh
```

Output:
```
Scanning system PATH...
Found 47 tools
Tool registry refreshed.
```

### Execute Tools with Agent-Formatted Output

The `pithos tool` command allows you to execute tools directly from the command line and see the same formatted output that would be provided to an agent. This is useful for testing tool calls and understanding how agents perceive tool results.

#### Basic Usage (Agent Format)

```bash
pithos tool python --version
```

Output:
```
Tool execution: python --version
Status: ✓ Success
Exit code: 0

Output:
Python 3.11.0
```

#### JSON Output Format

Get structured JSON output with all execution details:

```bash
pithos tool --format json python --version
```

Output:
```json
{
  "command": "python --version",
  "success": true,
  "exit_code": 0,
  "execution_time": 0.007,
  "stdout": "Python 3.11.0\n",
  "stderr": ""
}
```

#### Simple Output Format

Get only the raw stdout/stderr (useful for scripting):

```bash
pithos tool --format simple python --version
```

Output:
```
Python 3.11.0
```

**Note:** In simple format, the command exits with the same exit code as the tool.

#### Error Handling Example

When a tool is not allowed or fails, you get clear feedback:

```bash
pithos tool invalid-command --test
```

Output:
```
Tool execution: invalid-command --test
Status: ✗ Failed
Exit code: -1

Stderr:
Tool 'invalid-command' is not available or not allowed

💡 Hint: Tool 'invalid-command' not found or not allowed.
Available tools include: curl, git, node, npm, pip, python...
Use exact tool names from the available list.
```

The CLI follows the same error handling rules as agent tool calling - it never crashes and always provides actionable feedback.


## Security Considerations

### Input Validation

- **No shell expansion**: Commands are executed directly, not via shell
- **No piping**: Cannot chain commands with `|`, `>`, etc.
- **No variable expansion**: `$VAR` is treated as literal string
- **Argument splitting**: Basic whitespace splitting only

### Sandboxing

Currently, tools run with the same permissions as the pithos process. Future improvements:

- User permission restrictions
- Filesystem access controls
- Network access restrictions
- Resource usage limits (CPU, memory)

### Recommended Practices

1. **Use allowlists**: Only enable necessary tools
2. **Limit permissions**: Run pithos with restricted user account
3. **Review outputs**: Check tool results before acting on them
4. **Set tight timeouts**: Prevent long-running or hung processes
5. **Monitor usage**: Log tool executions for audit trails

## Troubleshooting

### Tool Not Found

**Problem**: Agent tries to use a tool but it's not in the registry.

**Solution**:
```bash
# Check if tool is discoverable
which python  # Linux/Mac
where python  # Windows

# Refresh cache
pithos-tools refresh

# Verify tool appears
pithos-tools list | grep python
```

### Tool Blocked by Config

**Problem**: Tool execution fails with "Tool not in allowed list"

**Solution**: Edit `configs/tools/tool_config.yaml`:
```yaml
include:
  - python  # Add your tool here
```

### Command Timeout

**Problem**: Tool execution times out

**Solution**: Increase timeout in config:
```yaml
timeout: 60  # Increase from default 10s
```

### Output Truncated

**Problem**: Tool output is cut off

**Solution**: Increase output size limit:
```yaml
max_output_size: 102400  # Increase from default 10KB
```

### Agent Doesn't Use Tools

**Problem**: Agent doesn't call tools even when enabled

**Solution**:
1. Verify tools are enabled: `agent.tools_enabled == True`
2. Check model capability (some models are better at tool use)
3. Give explicit instructions: "Use the python tool to check the version"
4. Try a more capable model (e.g., llama3.2:3b or larger)

### Tool Execution Error

**Problem**: Tool returns error or non-zero exit code

**Solution**:
```bash
# Test tool manually
pithos-tools test python --invalid-flag

# Check tool help
python --help

# Verify permissions
ls -l $(which python)
```

## Examples

### Check System Information

```python
agent = OllamaAgent("llama3.2")
agent.enable_tools(config_manager)

response = agent.send("""
Please check:
1. Python version
2. Operating system
3. Current user
""")
```

### Development Workflow

```python
agent.send("""
Check Git status and tell me if there are uncommitted changes.
If there are changes, list them.
""")
```

### Validation and Testing

```python
agent.send("""
Run the tests using: python -m pytest tests/
Tell me if they pass or fail.
""")
```

### Environment Setup Check

```python
agent.send("""
Verify the development environment:
1. Check Python version (should be 3.11+)
2. Check if git is installed
3. Check if pip is available
4. List installed packages
""")
```

## API Reference

### ToolRegistry

```python
class ToolRegistry:
    def __init__(self, config_manager: ConfigManager): ...
    def refresh(self) -> None: ...
    def list_tools(self) -> list[str]: ...
    def get_tool(self, name: str) -> Optional[ToolMetadata]: ...
```

### ToolExecutor

```python
class ToolExecutor:
    def __init__(self, timeout: int = 30, max_output_size: int = 10000): ...
    def run(self, command: str, tool_registry: ToolRegistry) -> ToolResult: ...
```

### ToolResult

```python
@dataclass
class ToolResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    command: str
    error_hint: Optional[str] = None
```

## Related Documentation

- [Configuration Guide](CONFIG.md) - Tool configuration details
- [Architecture](ARCHITECTURE.md) - How tool calling fits into pithos
- [Security](SECURITY.md) - Security considerations and best practices
