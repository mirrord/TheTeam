# Message-Based Routing

## Overview

TheTeam uses **message-based routing** as the default execution model. In message-based mode:

- **Nodes produce messages** instead of modifying a shared state dictionary
- **Messages are routed separately** along edges to target nodes
- **Nodes wait for all required inputs** before executing
- **Data flow is explicit** through message passing

This enables more sophisticated workflows including:
- Nodes with multiple inputs (merge/join patterns)
- Parallel execution paths
- Explicit data dependencies
- Better tracking of data provenance

## Key Concepts

### Messages

A `Message` carries data from one node to another:

```python
from pithos import Message

msg = Message(
    data="Hello, world!",
    source_node="node1",
    target_node="node2",
    input_key="input1"
)
```

**Fields:**
- `data`: The message payload (any Python object)
- `source_node`: ID of the node that produced this message
- `target_node`: ID of the node that should receive it
- `input_key`: Which input port/key this message is for (default: "default")
- `message_id`: Unique identifier
- `timestamp`: When the message was created
- `metadata`: Additional metadata dict

### Node Input Requirements

Nodes declare their required inputs and outputs:

```python
# Node with single default input/output
flowchart.add_node("simple", 
    type="prompt",
    prompt="Process: {default}",
    inputs=["default"],  # Required inputs
    outputs=["default"]  # Output keys
)

# Node with multiple inputs
flowchart.add_node("merge",
    type="prompt", 
    prompt="Merge {input1} and {input2}",
    inputs=["input1", "input2"],  # Both required
    outputs=["result"]
)
```

**Execution Rule:** A node only executes when **all** required inputs have received messages.

### Message Router

The `MessageRouter` manages message flow and tracks node readiness:

```python
from pithos.message import MessageRouter

# Unlimited history (default)
router = MessageRouter()

# Rolling window — keep only the last 50 messages
router = MessageRouter(max_history=50)

# Register nodes with their input requirements
router.register_node("node1", required_inputs=["default"])
router.register_node("node2", required_inputs=["input1", "input2"])

# Send messages
router.send_message(msg)

# Check which nodes are ready
ready = router.get_ready_nodes()  # Returns list of node IDs
```

**`max_history`** controls how many messages are retained in `message_history`.
Older messages are evicted as new ones arrive.  `0` (the default) means
unlimited.

## Usage

### Default Behavior

**Message-based routing is enabled by default.** New flowcharts automatically use message passing:

```python
from pithos import Flowchart, ConfigManager

config_manager = ConfigManager()
flowchart = Flowchart(config_manager)

# Add nodes and edges...
flowchart.add_node("start", type="prompt", prompt="Start", 
                   inputs=["default"], outputs=["default"])
flowchart.add_node("end", type="prompt", prompt="End",
                   inputs=["default"], outputs=["default"])
flowchart.add_edge("start", "end", AlwaysCondition)

# Execute with message-based routing
result = flowchart.run_message_based(
    initial_data="input data",
    max_steps=100
)
```

### Running Message-Based Flowcharts

#### Full Execution

Run the entire flowchart:

```python
result = flowchart.run_message_based(
    initial_data="Initial input",
    max_steps=100,
    history_window=0,       # 0 = unlimited (default)
    on_progress=None,       # optional callback — see below
)

print(f"Completed: {result['completed']}")
print(f"Steps: {result['steps']}")
print(f"Messages: {len(result['messages'])}")
```

**Returns:**
- `completed`: Whether the flowchart finished
- `steps`: Number of execution steps
- `messages`: List of output messages
- `message_history`: All messages sent during execution

#### Step-by-Step Execution

Execute one step at a time:

```python
flowchart.reset()

# Send initial message
initial_msg = Message(data="Start", input_key="default")
messages = flowchart.step_message_based(initial_msg)

# Continue stepping
while not flowchart.finished:
    messages = flowchart.step_message_based()
    print(f"Step produced {len(messages)} messages")
```

### Edge Configuration

Edges can specify which outputs connect to which inputs:

```python
# Connect specific output to specific input
flowchart.add_edge(
    "source_node",
    "target_node", 
    condition=AlwaysCondition,
    output_key="result",  # From source node's "result" output
    input_key="input1"    # To target node's "input1" input
)
```

## Rolling Message History

By default `MessageRouter` keeps every message that passes through it in
`message_history`, which can grow large for long-running flowcharts.  A
**rolling window** limits this to the *N* most recently sent messages:

```python
# Keep only the last 10 messages
result = flowchart.run_message_based(
    initial_data="start",
    max_steps=200,
    history_window=10,
)
print(len(result["message_history"]))  # <= 10
```

You can also change the window on an existing `MessageRouter`:

```python
flowchart.message_router._max_history = 10
```

**Rules:**
- `history_window=0` (default) — keep all messages.
- The window is enforced per-message: when `len > max_history` the oldest
  entries are removed first (FIFO eviction).
- `reset()` clears `message_history` but **preserves** the `_max_history`
  setting so it applies to the next run automatically.

## Progress Callback

Pass `on_progress` to `run_message_based` (or `run_team_flowchart`) to receive
a `ProgressEvent` **before each node executes**:

```python
from pithos import ProgressEvent, EdgeInfo

def my_callback(event: ProgressEvent) -> None:
    if event.edge:
        print(
            f"[{event.step}] {event.edge.from_node}"
            f" --[{event.edge.condition_type}]--> {event.node_id}"
        )
    else:
        print(f"[{event.step}] START -> {event.node_id}")
    print(f"  inputs          : {event.inputs}")
    print(f"  previous outputs: {[m.data for m in event.previous_results]}")

result = flowchart.run_message_based(
    initial_data="hello",
    on_progress=my_callback,
)
```

### `ProgressEvent` Reference

```python
@dataclass
class ProgressEvent:
    step: int                    # 0-based counter; increments after each node
    node_id: str                 # ID of the node about to run
    inputs: dict[str, Any]       # Input data keyed by port name
    edge: Optional[EdgeInfo]     # None for the start node
    previous_results: list[Message]  # Outputs of the previous node
```

### `EdgeInfo` Reference

```python
@dataclass
class EdgeInfo:
    from_node: str        # Source node of the traversed edge
    to_node: str          # Destination node (equals event.node_id)
    condition_type: str   # Class name of the condition, e.g. "AlwaysCondition"
    priority: int         # Edge priority value
    output_key: str       # Output port on the source node
    input_key: str        # Input port on the destination node
```

### Use Cases

- **Live progress reporting** in UIs or CLIs
- **Structured logging** of every node with timing
- **Early termination** — raise an exception inside the callback to abort
- **Collecting per-node metrics** for analysis

```python
import time

timings: list[tuple[str, float]] = []
start_times: dict[str, float] = {}

def timing_callback(event: ProgressEvent) -> None:
    if event.step > 0:
        elapsed = time.monotonic() - start_times.get(event.step - 1, 0)
        timings.append((event.node_id, elapsed))
    start_times[event.step] = time.monotonic()

flowchart.run_message_based(initial_data="in", on_progress=timing_callback)
```

## Patterns

### Linear Pipeline

Simple sequential processing:

```python
flow.add_node("step1", type="prompt", prompt="Step 1: {default}", 
              inputs=["default"], outputs=["default"])
flow.add_node("step2", type="prompt", prompt="Step 2: {default}",
              inputs=["default"], outputs=["default"])
flow.add_node("step3", type="prompt", prompt="Step 3: {default}",
              inputs=["default"], outputs=["default"])

flow.add_edge("step1", "step2", AlwaysCondition)
flow.add_edge("step2", "step3", AlwaysCondition)
```

### Merge/Join Pattern

Multiple inputs converging to one node:

```python
# Two source nodes
flow.add_node("source_a", type="prompt", prompt="Source A",
              inputs=["default"], outputs=["data_a"])
flow.add_node("source_b", type="prompt", prompt="Source B",
              inputs=["default"], outputs=["data_b"])

# Merge node waits for both inputs
flow.add_node("merge", type="prompt", 
              prompt="Combine {input_a} and {input_b}",
              inputs=["input_a", "input_b"],
              outputs=["combined"])

# Connect sources to merge node
flow.add_edge("source_a", "merge", AlwaysCondition,
              output_key="data_a", input_key="input_a")
flow.add_edge("source_b", "merge", AlwaysCondition,
              output_key="data_b", input_key="input_b")
```

The merge node will **only execute** after both `source_a` and `source_b` have produced their outputs.

### Conditional Branching

Route messages based on conditions:

```python
flow.add_node("check", type="prompt", prompt="Check input",
              inputs=["default"], outputs=["default"])
flow.add_node("path_a", type="prompt", prompt="Path A",
              inputs=["default"], outputs=["default"])
flow.add_node("path_b", type="prompt", prompt="Path B",
              inputs=["default"], outputs=["default"])

# Conditional edges
cond_a = Condition(lambda s: "A" in str(s.get("default", "")))
cond_b = Condition(lambda s: "B" in str(s.get("default", "")))

flow.add_edge("check", "path_a", cond_a, priority=1)
flow.add_edge("check", "path_b", cond_b, priority=2)
```

## Serialization

Flowcharts can be saved and loaded with message routing:

```python
# Save flowchart
flow.to_yaml("my_flow.yaml")

# Load flowchart
loaded_flow = Flowchart.from_yaml("my_flow.yaml", config_manager)

# Execute
result = loaded_flow.run_message_based(
    initial_data="test input",
    max_steps=100
)
```

## YAML Configuration

Flowcharts use message-based execution by default:

```yaml
start_node: start

nodes:
  start:
    type: prompt
    prompt: "Start: {default}"
    inputs: [default]
    outputs: [default]
  
  process:
    type: prompt
    prompt: "Process: {input1} and {input2}"
    inputs: [input1, input2]
    outputs: [result]
  
  end:
    type: prompt
    prompt: "End: {default}"
    inputs: [default]
    outputs: [default]

edges:
  - from: start
    to: process
    condition: {type: AlwaysCondition}
    output_key: default
    input_key: input1
  
  - from: process
    to: end
    condition: {type: AlwaysCondition}
    output_key: result
    input_key: default
```

## Implementation Details

### Node Execution

When a node executes in message mode:

1. **Build context** from input messages
2. **Apply extractions** (regex patterns)
3. **Execute** node logic
4. **Create output messages** from results
5. **Route messages** to downstream nodes

### Message Routing

The flowchart routes messages along edges:

1. **Evaluate conditions** on each edge
2. **Filter by priority** (lower = higher priority)
3. **Create routed messages** to target nodes
4. **Track edge traversal** (for count conditions)

### Node Readiness

A node becomes "ready" when:
- All required input keys have received messages
- The node hasn't executed yet (inputs cleared after execution)

## Testing

Test message-based execution:

```python
def test_message_flow():
    flow = Flowchart(config_manager)
    flow.add_node("node1", type="prompt", prompt="Test", extraction={})
    
    result = flow.run_message_based(initial_data="test")
    
    assert result["completed"]
    assert result["steps"] > 0
```

See `tests/test_message_routing.py` for comprehensive examples.

## Message-Based Execution Principles

pithos flowcharts use a message-based execution model:

- **Isolated messages** instead of shared state
- **Nodes produce messages** as outputs
- **Nodes wait for all inputs** before executing
- **Explicit message routing** via edges
- **Multiple named inputs** for complex merge patterns

## Best Practices

1. **Declare inputs/outputs explicitly** for clarity
2. **Use meaningful input_key names** for multi-input nodes
3. **Set max_steps** to prevent infinite loops
4. **Use history_window** for long-running flowcharts to bound memory usage
5. **Use on_progress** for debugging, logging, and UI feedback
6. **Check message_history** for post-run debugging
7. **Test readiness logic** for complex merge patterns
8. **Use conditions** for dynamic routing

## Limitations

- **Start node** must have a default input or receive an initial message
- **Parallel execution** of ready nodes not yet implemented (sequential)
- **Message filtering** on edges not yet supported
- **Stateful nodes** may need adaptation for message mode

## Future Enhancements

- Parallel execution of ready nodes
- Message transformation on edges
- Message priorities and queuing
- Timeout handling for blocked nodes
- Visual message flow debugging
- Streaming message processing