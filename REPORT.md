This file is an artifact of planning. Do not read or use it.
# pithos Modules Review Report

## 1. Obviously Unaddressed TODO Items

### agent_manager.py (5 TODOs - Critical)
The `AgentTeam` class has several unimplemented core features:

- **Line 178**: `breakdown_task()` - Task breakdown is not actually implemented. Currently asks coordinator to break down tasks but lacks structured decomposition logic.
- **Line 208**: `iterate()` - Proper task iteration mechanism missing.
- **Line 216**: Task completion tracking is not implemented.
- **Line 217**: Parallel notes vs shared notes distinction is not implemented.
- **Line 223**: Proper iteration mechanism within the iterate() method is missing.

**Impact**: The AgentTeam multi-agent coordination feature is essentially a stub. It won't work reliably for real team-based workflows.

---

## 2. Clear Paths to Design Improvement

### A. Architecture & Organization [COMPLETE]

**1. Message Routing vs State-Based Execution Duality** [DONE]
- The `Flowchart` class supports both message-based and state-based execution modes
- This creates confusion and complexity (see error messages warning about mixing modes)
- **Recommendation**: Deprecate state-based execution entirely or separate into distinct classes (`StateFlowchart` and `MessageFlowchart`)

**2. Tool Calling Pattern is Fragile** [DONE]
- Relies on regex parsing of agent output (`runcommand("...")`)
- Pattern: `r'runcommand\(["\']([^"\'\'\"]+)["\']\)'` in agent.py line 522
- **Issue**: Easily broken by LLM output variations, quotes within commands, etc.
- **Recommendation**: Implement proper structured tool calling using Ollama's native tool support (if available) or JSON-based function calling

**3. Memory Operations Similar Fragility** [DONE]
- Same regex-based parsing for `storemem()` and `retrievemem()` operations
- **Recommendation**: Consolidate tool and memory operations into a unified function-calling system

**4. FlowNode Message/State Hybrid** [DONE]
- FlowNode has both `do()` (state-based) and `execute_with_messages()` (message-based)
- Output message creation logic in `_create_output_messages()` tries to map state dict to messages
- **Recommendation**: Clean separation or complete migration to message-based system

### B. Code Quality & Maintainability [COMPLETE]

**1. Error Handling Gaps** [DONE]
- `ToolExecutor.run()` catches broad exceptions but error reporting could be clearer
- ✅ **Validation of node configurations before flowchart execution** - COMPLETED
  - Created comprehensive validation module (`src/pithos/validation.py`)
  - Validates node types, required parameters, edge configurations
  - Checks for unreachable nodes, cycles, and security concerns
  - Integrated into `Flowchart.from_dict()` with automatic validation
  - Added `Flowchart.validate()` method for manual validation
  - 29 comprehensive validation tests plus integration tests
- Missing input validation in many places (e.g., empty strings, None values)

**2. Type Consistency** [DONE]
- Mix of `list[str]` (Python 3.9+) and `list[str]` (typing module) throughout codebase
- **Recommendation**: Pick one style and standardize

**3. Config Manager Path Resolution** [DONE]
- Uses `Path(__file__).parent.parent.resolve()` to find configs directory
- **Issue**: Fragile when package structure changes or running from different locations
- **Recommendation**: Make config_dir configurable with environment variable support

**4. CustomNode Security** [DONE]
- Uses `exec()` with global `__builtins__` exposed (flownode.py line 391)
- **Warning**: Major security risk if executing untrusted YAML configs
- **Recommendation**: Implement sandboxing or restricted execution environment

### C. Performance Considerations [COMPLETE]

**1. Tool Discovery on Every Init** [DONE]
- `ToolRegistry.__init__()` scans entire PATH every time
- **Impact**: Slow initialization, especially on Windows
- **Current**: Has config-based caching but discovery still runs
- **Recommendation**: Implement proper persistent cache with invalidation

**2. Memory Store Collection Caching** [DONE]
- Collections are cached in `_collections` dict
- **Good**: But no cache invalidation strategy
- **Recommendation**: Add TTL or size-based eviction

**3. Message Router History** [DONE]
- Keeps unlimited `message_history` (message.py line 94)
- **Recommendation**: Add max size with rolling window

---

## 3. Features to Significantly Enhance Usefulness & Usability

### A. High-Impact Features

**1. Streaming Response Support** [DONE]
- Current `agent.send()` waits for complete response
- **Feature**: Add streaming support for real-time token-by-token output
- **Benefit**: Critical for web UI responsiveness and better UX
```python
def send_streaming(self, content: str, callback: Callable[[str], None]) -> str:
    # Stream tokens to callback as they arrive
```

**2. Flowchart Debugging & Visualization** [DONE]
- No built-in way to trace flowchart execution
- **Feature**: Add execution tracing with node timing, state snapshots, decision paths
```python
flowchart.enable_trace()
result = flowchart.run_message_based(data)
print(flowchart.get_execution_trace())  # Shows node sequence, timing, conditions
```

**3. Agent Conversation Persistence** [DONE]
- Can serialize to dict but no built-in save/load to files
- **Feature**: Add conversation save/resume functionality
```python
agent.save_conversation("conversations/session_123.json")
agent.load_conversation("conversations/session_123.json")
```

**4. Flowchart Validation** [DONE]
- No validation that flowcharts are executable before running
- **Feature**: Add pre-execution validation
  - Check all nodes have required inputs
  - Detect unreachable nodes
  - Verify at least one path to output nodes
  - Validate condition syntax

### B. Developer Experience Improvements

**6. Better Configuration Validation**
- YAML configs have no schema validation
- **Feature**: Add JSON Schema or Pydantic models for all config types
- **Benefit**: Catch errors early with clear messages

**7. Built-in Prompt Templates**
- Users must write all prompts from scratch
- **Feature**: Add template library with variables
```python
from pithos.templates import PromptTemplate

template = PromptTemplate.get("code_review")
prompt = template.render(code=code_snippet, language="python")
```

**8. Context Tagging & Search** [DONE]
- No way to tag or search through conversation history
- **Feature**: Add metadata tags and search capabilities
```python
agent.tag_current_message(["important", "bug-fix"])
results = agent.search_history("authentication error")
```

**9. Flowchart Composition** [DONE]
- Can't easily combine/reuse flowchart components
- **Feature**: Add sub-flowchart support
  - Define reusable flowchart modules
  - Import and nest them in larger flows

**10. Progress Callbacks for Long Operations** [DONE]
- No feedback during long flowchart executions
- **Feature**: Add progress reporting
```python
def progress_callback(node_id: str, step: int, total: int):
    print(f"Executing {node_id} ({step}/{total})")

flowchart.run_message_based(data, on_progress=progress_callback)
```

### C. Advanced Features for Power Users

**11. Conditional Flowchart Branches Based on Agent Analysis**
- Current conditions are state-based only
- **Feature**: Add LLM-evaluated conditions
```yaml
edges:
  - from: analyze
    to: deep_dive
    condition:
      type: llm_condition
      prompt: "Does the analysis suggest we need more detail?"
      threshold: 0.7  # Confidence threshold
```

**12. Agent Personality Profiles**
- System prompts are ad-hoc
- **Feature**: Pre-configured personality/role templates
```python
agent = OllamaAgent.from_role(
    "software_architect",
    model="glm-4.7-flash",
    expertise_level="senior"
)
```

**13. Memory Auto-Categorization**
- Users must manually specify categories
- **Feature**: LLM-based automatic category suggestion
```python
agent.enable_memory(auto_categorize=True)
# Agent automatically determines best category for stored knowledge
```

**14. Flowchart Hot-Reload** [DONE]
- Changes to flowchart YAML require restart
- **Feature**: Watch for file changes and reload
```python
flowchart = Flowchart.from_registered("my_flow", watch=True)
# Automatically reloads when YAML changes
```

**15. Multi-Agent Negotiation Patterns**
- AgentTeam is incomplete
- **Feature**: Implement common multi-agent patterns:
  - Debate (agents argue positions)
  - Consensus (agents vote on solutions)
  - Chain-of-custody (sequential agent refinement)
  - Parallel-merge (agents work independently, merge results)

### D. Observability & Monitoring

**16. Built-in Metrics Collection** [DONE]
- No tracking of performance metrics
- **Feature**: Add metrics tracking
  - Token usage per model
  - Response times
  - Tool call success rates
  - Memory hit rates
  - Flowchart execution paths

**17. Logging Infrastructure** [DONE]
- Print statements throughout (e.g., agent_manager.py line 222)
- **Feature**: Replace with proper logging
```python
import logging
logger = logging.getLogger("pithos.agent_manager")
logger.info("Agent processing step", extra={"agent": agent_name})
```

**18. Error Recovery Strategies**
- Flowchart failures abort entire execution
- **Feature**: Add retry logic and error handlers
```yaml
nodes:
  api_call:
    type: toolcall
    retry: 3
    on_error: fallback_node
```

---

## 4. Priority Recommendations

### Immediate (Critical for Stability)
1. ✅ **Complete AgentTeam implementation** - It's advertised but non-functional
2. ✅ **Fix security issue in CustomNode** - `exec()` with unrestricted builtins
3. ✅ **Add configuration validation** - Prevent runtime errors from bad configs

### Short-term (Usability)
4. ✅ **Implement streaming responses** - Essential for web UI
5. ✅ **Add flowchart validation** - Catch errors before execution
6. ✅ **Improve error messages** - Clearer guidance for users
7. ✅ **Add conversation persistence** - Save/load sessions

### Medium-term (Enhancement)
8. ✅ **Migrate to unified function calling** - Replace regex parsing
9. ✅ **Add execution tracing/debugging** - Understand flowchart behavior
10. ✅ **Implement prompt templates** - Reduce boilerplate

### Long-term (Advanced Capabilities)
11. ✅ **Multi-model flowchart nodes** - Optimize cost/performance
12. ✅ **LLM-evaluated conditions** - More intelligent branching
13. ✅ **Advanced multi-agent patterns** - Rich collaboration models
14. ✅ **Comprehensive observability** - Production-ready monitoring

---

## Summary

The pithos framework has a solid foundation with good architectural concepts (contexts, flowcharts, message routing). However, it suffers from:

- **Incomplete features** (AgentTeam)
- **Dual execution models** causing complexity
- **Fragile parsing patterns** for tools/memory
- **Security concerns** with code execution
- **Limited observability** into execution

The highest-value improvements would be:
1. Completing the multi-agent system
2. Adding streaming and tracing for better UX
3. Implementing proper validation and error handling
4. Migrating to structured function calling

These changes would transform pithos from an interesting prototype into a production-ready agent framework.