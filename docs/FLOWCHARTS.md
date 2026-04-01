# Flowchart Design Guide

This guide explains how to design effective flowcharts for guiding LLM agent reasoning in pithos.

## What Are Flowcharts?

Flowcharts in pithos are directed graphs that guide agent reasoning through structured workflows. Each node represents a step (like prompting the agent, executing code, or gathering input), and edges define transitions between steps based on conditions.

## Why Use Flowcharts?

Flowcharts provide several benefits over simple prompting:

1. **Structured Reasoning**: Break complex problems into discrete steps
2. **Reflection**: Enable agents to review and refine their outputs
3. **Conditional Logic**: Branch based on agent output or state
4. **Iterative Improvement**: Loop through refinement cycles
5. **Reproducibility**: Save and reuse proven reasoning patterns
6. **Transparency**: Visualize agent decision-making process

## Basic Structure

### Minimal Flowchart

```yaml
start_node: begin

nodes:
  begin:
    type: prompt
    prompt: "Analyze the problem."
  
  finish:
    type: chatoutput
    source: current_input

edges:
  - from: begin
    to: finish
    condition:
      type: AlwaysCondition
    priority: 1
```

### Key Components

1. **start_node**: Entry point for execution
2. **nodes**: Dictionary of node definitions
3. **edges**: List of transitions between nodes

### Input and Output Requirements

**Every flowchart must have at least one input node and one output node.** This ensures data flows into and out of the flowchart properly.

- **If no input node exists**: A `ChatInputNode` is automatically added at the beginning of the flowchart
- **If no output node exists**: A `ChatOutputNode` is automatically added at the end of the flowchart

This automatic insertion means you can focus on the core logic of your flowchart without always needing to explicitly define I/O nodes. However, for production flowcharts or specific use cases (like file I/O), you should explicitly define your input and output nodes.

**Example - Automatic I/O**:
```yaml
# No explicit I/O nodes - they're added automatically
start_node: process

nodes:
  process:
    type: prompt
    prompt: "Analyze: {current_input}"
```

This is equivalent to:
```yaml
start_node: __auto_chat_input__

nodes:
  __auto_chat_input__:
    type: chatinput
    prompt_message: "Enter your input:"
    save_to: user_input
  
  process:
    type: prompt
    prompt: "Analyze: {current_input}"
  
  __auto_chat_output__:
    type: chatoutput
    source: current_input

edges:
  - from: __auto_chat_input__
    to: process
  - from: process
    to: __auto_chat_output__
```

## Node Types

### Prompt Node

Sends a prompt to the LLM agent:

```yaml
my_node:
  type: prompt
  prompt: "Your instruction here"
```

**Use cases**:
- Primary reasoning step
- Request analysis or generation
- Ask for self-critique
- Guide reflection

**Example**:
```yaml
analyze:
  type: prompt
  prompt: |
    Analyze the following problem carefully.
    Break it down into smaller components.
    Identify assumptions and constraints.
```

### Text Parse Node

Extracts information from text input and sets variables in the execution state:

```yaml
capture_question:
  type: textparse
  set:
    original_question: '{current_input}'
    context: 'initial'
```

**Use cases**:
- Capture input values into named variables
- Extract patterns using regex
- Parse and store intermediate results
- Separate data extraction from prompt logic

**Configuration**:
- `set`: Dictionary mapping variable names to values (can reference `{current_input}` and other state variables)
- `extraction`: Dictionary mapping variable names to regex patterns for extracting values
- `transform`: Optional transformation instruction

**Example - Capture Variables**:
```yaml
CaptureQuestion:
  type: textparse
  set:
    original_question: '{current_input}'
    timestamp: '{current_timestamp}'
```

**Example - Regex Extraction**:
```yaml
ExtractAnswer:
  type: textparse
  extraction:
    answer_letter: "Answer:\s*([A-D])"
    confidence: "Confidence:\s*(\d+)%"
```

**Design Pattern**: Use TextParse nodes to separate concerns:
- **Prompt nodes** focus on generating prompts and invoking the LLM
- **TextParse nodes** focus on extracting, storing, and organizing data

This separation makes flowcharts more maintainable and easier to understand.

### Custom Code Node

Executes Python code to manipulate state:

```yaml
process:
  type: custom
  custom_code: |
    # Access context and modify it
    output = context.get("last_output", "")
    context["word_count"] = len(output.split())
    context["contains_error"] = "error" in output.lower()
```

**Use cases**:
- Parse agent outputs
- Extract information
- Calculate metrics
- Set flags for conditional branching

**Available variables**:
- `context`: Dictionary with execution state
- Standard Python libraries

**Security note**: Code runs without sandboxing—only use trusted flowcharts.

### Input Nodes

Input nodes bring data into a flowchart from external sources. **Every flowchart must have at least one input node.** If no input node is explicitly defined, a `ChatInputNode` is automatically added at the beginning.

#### Chat Input Node

Receives input from the user via the chat interface (default input type):

```yaml
get_user_input:
  type: chatinput
  prompt_message: "Enter your query:"
  save_to: user_input
```

**Configuration**:
- `prompt_message`: Message displayed when requesting input (default: "Enter your input:")
- `save_to`: State variable to store the input (default: "user_input")

**Use cases**:
- Interactive user input
- Starting a conversation chain
- Gathering requirements

#### File Input Node

Reads data from a file:

```yaml
load_data:
  type: fileinput
  file_path: "data/input.txt"
  save_to: file_content
  encoding: utf-8
```

**Configuration**:
- `file_path`: Path to the file to read (can use `{placeholders}` for dynamic paths)
- `save_to`: State variable to store file content (default: "file_content")
- `encoding`: File encoding (default: "utf-8")

**Use cases**:
- Load documents for analysis
- Read configuration files
- Import datasets

**Example with dynamic path**:
```yaml
load_user_file:
  type: fileinput
  file_path: "data/{username}/profile.txt"
  save_to: user_profile
```

### Output Nodes

Output nodes write data to external destinations. **Every flowchart must have at least one output node.** If no output node is explicitly defined, a `ChatOutputNode` is automatically added at the end.

#### Chat Output Node

Displays output in the chat interface (default output type):

```yaml
show_result:
  type: chatoutput
  source: current_input
  format_template: "Result: {current_input}"
```

**Configuration**:
- `source`: State variable to output (default: "current_input")
- `format_template`: Optional template for formatting output

**Use cases**:
- Display final results
- Show intermediate outputs
- Provide user feedback

**Example with formatting**:
```yaml
display_analysis:
  type: chatoutput
  source: analysis_result
  format_template: |
    Analysis Complete:
    ---
    {analysis_result}
    ---
    Confidence: {confidence_score}%
```

#### File Output Node

Writes data to a file:

```yaml
save_results:
  type: fileoutput
  file_path: "output/results.txt"
  source: current_input
  mode: w
  encoding: utf-8
```

**Configuration**:
- `file_path`: Path to write to (can use `{placeholders}` for dynamic paths)
- `source`: State variable to write (default: "current_input")
- `mode`: File mode - "w" for write, "a" for append (default: "w")
- `encoding`: File encoding (default: "utf-8")

**Use cases**:
- Save analysis results
- Log outputs
- Export processed data

**Example with append mode**:
```yaml
log_result:
  type: fileoutput
  file_path: "logs/{session_id}.log"
  source: current_input
  mode: a
```

### Tool Call Node

Executes a command-line tool:

```yaml
check_version:
  type: toolcall
  command: "python --version"
  save_to: version_result
```

**Use cases**:
- Verify environment
- Run tests or validation
- Execute scripts
- Query system state

## Edges and Conditions

Edges define transitions between nodes. Each edge has:

- **from**: Source node
- **to**: Target node
- **condition**: Condition object that determines if transition is allowed
- **priority**: Lower number = higher priority

### Condition Types

#### Always Condition

Always allows traversal:

```yaml
- from: start
  to: next
  condition:
    type: AlwaysCondition
  priority: 1
```

**Use**: Linear progressions

#### Count Condition

Allows N traversals, then blocks:

```yaml
- from: review
  to: review  # Self-loop
  condition:
    type: CountCondition
    limit: 3
  priority: 1
```

**Use**: Iteration limits, preventing infinite loops

#### Regex Condition

Checks the current input against a regular expression:

```yaml
- from: check
  to: success
  condition:
    type: RegexCondition
    regex: "passed"
    matchtype: search
  priority: 1
```

**Use**: Branching based on agent output content

### Edge Priority

When multiple edges from a node have open conditions, the lowest priority value wins:

```yaml
edges:
  # Try this first
  - from: check
    to: retry
    condition:
      type: CountCondition
      limit: 2
    priority: 1
  
  # Then fall back to this
  - from: check
    to: finish
    condition:
      type: AlwaysCondition
    priority: 2
```

This allows "try X times, then give up" patterns.

## Design Patterns

### Linear Reasoning

Simple sequential steps:

```yaml
start_node: step1

nodes:
  step1:
    type: prompt
    prompt: "Step 1: Identify the problem."
  
  step2:
    type: prompt
    prompt: "Step 2: Propose solutions."
  
  step3:
    type: prompt
    prompt: "Step 3: Evaluate solutions."
  
  done:
    type: chatoutput
    source: current_input

edges:
  - {from: step1, to: step2, condition: {type: AlwaysCondition}, priority: 1}
  - {from: step2, to: step3, condition: {type: AlwaysCondition}, priority: 1}
  - {from: step3, to: done, condition: {type: AlwaysCondition}, priority: 1}
```

### Reflection Loop

Agent reviews and refines its output:

```yaml
start_node: initial

nodes:
  initial:
    type: prompt
    prompt: "Provide your answer."
  
  reflect:
    type: prompt
    prompt: |
      Review your previous answer.
      Identify any errors or areas for improvement.
      Provide a refined answer.
  
  done:
    type: chatoutput
    source: current_input

edges:
  - from: initial
    to: reflect
    condition: {type: AlwaysCondition}
    priority: 1
  
  # Reflect up to 2 times
  - from: reflect
    to: reflect
    condition: {type: CountCondition, limit: 2}
    priority: 1
  
  # Then finish
  - from: reflect
    to: done
    condition: {type: AlwaysCondition}
    priority: 2
```

### Conditional Branching

Different paths based on output:

```yaml
start_node: analyze

nodes:
  analyze:
    type: prompt
    prompt: "Analyze the input and rate your confidence (high/low)."
  
  extract_confidence:
    type: custom
    custom_code: |
      output = context.get("last_output", "").lower()
      context["high_confidence"] = "high confidence" in output
  
  proceed:
    type: prompt
    prompt: "Proceed with high confidence answer."
  
  refine:
    type: prompt
    prompt: "Low confidence detected. Think more carefully."
  
  done:
    type: chatoutput
    source: current_input

edges:
  - from: analyze
    to: extract_confidence
    condition: {type: AlwaysCondition}
  
  - from: extract_confidence
    to: proceed
    condition:
      type: RegexCondition
      regex: "True"
      matchtype: search
    priority: 1
  
  - from: extract_confidence
    to: refine
    condition: {type: AlwaysCondition}
    priority: 2
  
  - from: proceed
    to: done
    condition: {type: AlwaysCondition}
  
  - from: refine
    to: done
    condition: {type: AlwaysCondition}
```

### Teacher-Student Pattern

Two-agent interaction (requires multi-agent support):

```yaml
start_node: student_answer

nodes:
  student_answer:
    type: prompt
    prompt: "Provide your answer to the problem."
  
  teacher_review:
    type: prompt
    prompt: |
      Review the student's answer.
      Provide constructive feedback.
      Identify errors and suggest improvements.
  
  student_revise:
    type: prompt
    prompt: |
      Based on the teacher's feedback, revise your answer.
  
  done:
    type: chatoutput
    source: current_input

edges:
  - from: student_answer
    to: teacher_review
    condition: {type: AlwaysCondition}
  
  - from: teacher_review
    to: student_revise
    condition: {type: AlwaysCondition}
  
  # Can iterate multiple times
  - from: student_revise
    to: teacher_review
    condition: {type: CountCondition, limit: 2}
    priority: 1
  
  - from: student_revise
    to: done
    condition: {type: AlwaysCondition}
    priority: 2
```

### Systematic Backwards Reasoning

Work backwards from goal to solution:

```yaml
start_node: define_goal

nodes:
  define_goal:
    type: prompt
    prompt: "What is the desired end state?"
  
  identify_prerequisites:
    type: prompt
    prompt: "What conditions must be true to reach that goal?"
  
  plan_steps:
    type: prompt
    prompt: "What steps achieve those prerequisites?"
  
  execute:
    type: prompt
    prompt: "Execute the plan step-by-step."
  
  verify:
    type: prompt
    prompt: "Verify the goal is achieved."
  
  done:
    type: chatoutput
    source: current_input

edges:
  - {from: define_goal, to: identify_prerequisites, condition: {type: AlwaysCondition}}
  - {from: identify_prerequisites, to: plan_steps, condition: {type: AlwaysCondition}}
  - {from: plan_steps, to: execute, condition: {type: AlwaysCondition}}
  - {from: execute, to: verify, condition: {type: AlwaysCondition}}
  - {from: verify, to: done, condition: {type: AlwaysCondition}}
```

## Best Practices

### Prompt Design

1. **Be specific**: Clear instructions produce better results
2. **Provide context**: Reference previous steps when needed
3. **Ask for reasoning**: Request step-by-step thinking
4. **Set expectations**: Specify output format or criteria

**Good**:
```yaml
prompt: |
  Review your previous answer for errors.
  Check for: logical inconsistencies, false assumptions, missing steps.
  Provide a revised answer that addresses any issues found.
```

**Avoid**:
```yaml
prompt: "Review your answer."
```

### Loop Control

1. **Always use count limits**: Prevent infinite loops
2. **Set reasonable limits**: 2-5 iterations typically sufficient
3. **Provide escape conditions**: Use priority to ensure exit path

**Good**:
```yaml
edges:
  - from: refine
    to: refine
    condition: {type: CountCondition, limit: 3}
    priority: 1  # Try this first
  
  - from: refine
    to: done
    condition: {type: AlwaysCondition}
    priority: 2  # Then exit
```

**Avoid**:
```yaml
edges:
  - from: refine
    to: refine
    condition: {type: AlwaysCondition}  # Infinite loop!
```

### State Management

1. **Use meaningful keys**: `context["confidence_level"]` not `context["x"]`
2. **Document state**: Comment what custom code sets/reads
3. **Validate state**: Check for key existence before reading

**Good**:
```yaml
type: custom
custom_code: |
  # Extract confidence rating from output
  output = context.get("last_output", "")
  if "high" in output:
      context["confidence"] = "high"
  elif "low" in output:
      context["confidence"] = "low"
  else:
      context["confidence"] = "unknown"
```

### Node Naming

1. **Use descriptive names**: Indicate purpose at a glance
2. **Follow conventions**: `analyze`, `reflect`, `check_`, `extract_`
3. **Be consistent**: Same patterns across flowcharts

**Good**: `extract_confidence`, `refine_answer`, `verify_solution`
**Avoid**: `node1`, `step`, `x`

### Modularity

1. **Keep flowcharts focused**: One reasoning pattern per flowchart
2. **Reuse patterns**: Create library of proven flowcharts
3. **Compose workflows**: Chain multiple flowcharts for complex tasks

## Testing Flowcharts

### Manual Testing

```python
from pithos import Flowchart, OllamaAgent, ConfigManager

config_manager = ConfigManager()
flowchart = Flowchart.from_yaml("my_flow.yaml", config_manager)

agent = OllamaAgent("glm-4.7-flash")
context = agent.create_context("test", flowchart=flowchart)

# Execute flowchart
result = flowchart.run_message_based(
    initial_data="test problem",
    max_steps=100,
)
print(f"Completed: {result['completed']}")
print(f"Output messages: {result['messages']}")
```

### Unit Testing

```python
def test_flowchart_structure():
    config_manager = ConfigManager()
    flowchart = Flowchart.from_registered("my_flow", config_manager)
    
    # Verify structure
    assert flowchart.start_node == "analyze"
    assert "reflect" in flowchart.graph.nodes
    assert flowchart.graph.has_edge("analyze", "reflect")

def test_condition_logic():
    from pithos.conditions import RegexCondition
    condition = RegexCondition(regex="high", matchtype="search")
    state = {"current_input": "high confidence"}
    
    assert condition.is_open(state) == True
```

## Performance Optimization

### Minimize Prompts

Each prompt node calls the LLM, which is slow. Combine related steps:

**Inefficient**:
```yaml
nodes:
  step1:
    type: prompt
    prompt: "Identify the problem."
  step2:
    type: prompt
    prompt: "List the constraints."
  step3:
    type: prompt
    prompt: "Propose a solution."
```

**Better**:
```yaml
nodes:
  analyze:
    type: prompt
    prompt: |
      1. Identify the problem
      2. List the constraints
      3. Propose a solution
```

### Cache Results

Store expensive computations in state:

```yaml
expensive_check:
  type: custom
  custom_code: |
    # Only compute once
    if "cached_result" not in context:
        # Expensive operation
        context["cached_result"] = complex_computation()
```

### Limit Iterations

More iterations = more LLM calls. Start with 2-3 and increase only if needed.

## Debugging

### Progress Callback

The most powerful way to trace execution is the `on_progress` callback, which is
called **before each node runs** and receives a `ProgressEvent` with full context
about what is about to happen:

```python
from pithos import ProgressEvent

def on_progress(event: ProgressEvent):
    edge_label = (
        f"via {event.edge.from_node} --[{event.edge.condition_type}]--> "
        if event.edge else "start"
    )
    print(f"[step {event.step}] {edge_label}{event.node_id}")
    print(f"  inputs : {event.inputs}")
    if event.previous_results:
        print(f"  prev   : {[m.data for m in event.previous_results]}")

result = flowchart.run_message_based(
    initial_data="test input",
    max_steps=100,
    on_progress=on_progress,
)
```

**`ProgressEvent` fields:**

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Zero-based step counter |
| `node_id` | `str` | ID of the node about to execute |
| `inputs` | `dict[str, Any]` | Input data keyed by port name |
| `edge` | `EdgeInfo \| None` | Edge that routed execution here (`None` for the start node) |
| `previous_results` | `list[Message]` | Output messages from the previous node |

**`EdgeInfo` fields:** `from_node`, `to_node`, `condition_type`, `priority`, `output_key`, `input_key`.

### Inspect Messages

```python
result = flowchart.run_message_based(
    initial_data="test input",
    max_steps=100,
)
print(f"Steps executed: {result['steps']}")
print(f"Output messages: {result['messages']}")
for msg in result['message_history']:
    print(f"Message: {msg.data} -> {msg.target_node}/{msg.input_key}")
```

### Trace Execution Path

Inspect the message history to see execution flow:

```python
result = flowchart.run_message_based(
    initial_data="test input",
    max_steps=100,
)
execution_path = [msg.target_node for msg in result['message_history']]
print(f"Path taken: {' -> '.join(str(n) for n in execution_path)}")
```

### Limit Message History Size

Long-running flowcharts can accumulate a large `message_history`. Use
`history_window` to keep only the most recent *N* messages, reducing memory
usage without losing the current run's output:

```python
result = flowchart.run_message_based(
    initial_data="test input",
    max_steps=500,
    history_window=20,   # Keep only the last 20 messages
)
# result['message_history'] will have at most 20 entries
```

Set `history_window=0` (the default) for an unlimited rolling buffer.

## Common Pitfalls

### Infinite Loops

**Problem**: Flowchart never finishes

**Solution**: Add count limits to all loops

### Unreachable Nodes

**Problem**: Some nodes never execute

**Solution**: Verify all nodes have incoming edges (except start)

### Condition Never Satisfied

**Problem**: Flowchart gets stuck

**Solution**: Add lower-priority "always" condition as fallback

### State Key Errors

**Problem**: Custom code fails on missing keys

**Solution**: Use `context.get("key", default)` instead of `context["key"]`

## Examples

See the `configs/flowcharts/` directory for real examples:

- `simple_reflect.yaml` - Basic reflection pattern
- `refined_reflect.yaml` - Multi-stage reflection
- `systematic_backwards.yaml` - Backwards reasoning
- `teacher_student.yaml` - Two-agent interaction
- `stepwise_reflect.yaml` - Incremental refinement

## Flowcharts as Agent Inference Engines

In addition to standalone execution, flowcharts can be attached directly to an agent as an **inference flowchart**. When set, every call to `agent.send()` is automatically routed through the flowchart, enabling structured chain-of-thought reasoning on every response.

### How It Works

1. User calls `agent.send("question")`
2. The agent detects an attached inference flowchart and delegates to `_inference_send()`
3. A disposable temporary context is created for the flowchart run
4. The agent is injected into the flowchart's `shared_context` so that `PromptNode` nodes can call `agent.send()` directly
5. A `_running_inference` flag prevents recursive re-entry — nested `send()` calls from PromptNodes go straight to the LLM
6. The flowchart executes (e.g., Generate → Reflect → Refine)
7. Only the **final output** is recorded in the agent's main conversation context
8. Post-processing (tool calling, memory, compaction) runs on the final output as usual

### Setting an Inference Flowchart

```python
from pithos import OllamaAgent, ConfigManager

config_manager = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")

# By registered name
agent.set_inference_flowchart("simple_reflect", config_manager)

# By inline dict
agent.set_inference_flowchart({
    "start_node": "generate",
    "nodes": {
        "generate": {"type": "prompt", "prompt": "Answer: {current_input}"},
        "reflect": {"type": "prompt", "prompt": "Review and improve your answer."},
    },
    "edges": [
        {"from": "generate", "to": "reflect", "condition": {"type": "AlwaysCondition"}},
    ]
}, config_manager)

# By Flowchart instance
flowchart = Flowchart.from_registered("simple_reflect", config_manager)
agent.set_inference_flowchart(flowchart)

# Remove it
agent.clear_inference_flowchart()
```

### YAML Configuration

Inference flowcharts can be configured directly in agent YAML configs:

```yaml
# configs/agents/structured-reflect.yaml
default_model: glm-4.7-flash:latest
system_prompt: "You are a careful, reflective assistant."

inference:
  start_node: Generate
  nodes:
    Generate:
      type: prompt
      prompt: "{current_input}"
    Reflect:
      type: prompt
      prompt: |
        Review your previous answer. Identify errors or improvements.
        Provide a refined answer.
  edges:
    - from: Generate
      to: Reflect
      condition: { type: AlwaysCondition }
      priority: 1
```

### Design Considerations

- **Temporary context**: The flowchart runs in a disposable context (`_cot_<uuid>`) that is deleted after each call, so intermediate reasoning steps don't pollute the main conversation
- **Recursion guard**: The `_running_inference` flag ensures PromptNodes inside the flowchart call the LLM directly without re-entering the flowchart
- **Transparency**: The user sees only the final refined output; the intermediate reasoning steps are internal
- **Composability**: Inference flowcharts work alongside tools, memory, compaction, and all other agent features

## Related Documentation

- [Configuration Guide](CONFIG.md) - Flowchart YAML syntax
- [Architecture](ARCHITECTURE.md) - How flowcharts fit into pithos
- [Tool Calling](TOOL_CALLING.md) - Using tools in flowcharts
