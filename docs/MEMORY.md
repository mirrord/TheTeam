# Memory Tool Guide

The pithos Memory Tool provides vector database-based knowledge storage and retrieval for agents. It enables agents to build up domain knowledge over time, store important facts, and retrieve relevant context from previous interactions. Two automatic memory management features sit on top of the core store: **context compaction** trims long conversation histories into summaries, and **automatic recall** surfaces relevant past knowledge before each response.

## Overview

The memory tool organizes knowledge into **categories** (like individual RAG databases), where each category contains related information. Under the hood, it uses [ChromaDB](https://www.trychroma.com/), a lightweight vector database that:

- **Stores text with embeddings** for semantic search
- **Persists data locally** for offline use
- **Supports metadata filtering** for precise queries
- **Calculates relevance scores** automatically

## Quick Start

### Installation

ChromaDB is included as a core dependency. It is installed automatically when you install the project:

```bash
pip install -e .
```

### Basic Usage (CLI)

```bash
# Store knowledge in a category
pithos-memory store facts "Python is a high-level programming language"

# Retrieve knowledge from a category
pithos-memory retrieve facts "What is Python?"

# List all categories
pithos-memory list

# Get category information
pithos-memory info facts

# Export a category to JSON
pithos-memory export facts facts_backup.json

# Import from JSON
pithos-memory import facts_backup.json
```

### Agent Integration

Enable memory for an agent to allow it to store and retrieve knowledge during conversations:

```python
from pithos import OllamaAgent, ConfigManager

config_manager = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")

# Enable memory
agent.enable_memory(config_manager)

# Now the agent can use memory operations
response = agent.send("Remember that Python was created by Guido van Rossum")
# Agent outputs: storemem(facts, "Python was created by Guido van Rossum")

# Later, retrieve the information
response = agent.send("Who created Python?")
# Agent outputs: retrievemem(facts, "Python creator")
# System provides: Retrieved results from facts...
# Agent responds: "Python was created by Guido van Rossum"
```

## Memory Operations

When memory is enabled for an agent, it can use two operations with multiple syntax formats for reliability:

### Store Operation

pithos supports multiple syntax formats:

**CLI-Style (Simplest):**
```
STORE[facts]: The Eiffel Tower is 330 meters tall
STORE[notes]: Important meeting on Friday
```

**Function-Style:**
```
store(facts, The Eiffel Tower is 330 meters tall)
store(notes, Important meeting on Friday)
```

**Legacy (Still Supported):**
```
storemem(facts, "The Eiffel Tower is 330 meters tall")
storemem(notes, "Important meeting on Friday")
```

The system automatically:
- Generates a unique ID
- Creates embeddings for semantic search
- Adds timestamp metadata
- Persists to disk

### Retrieve Operation

**CLI-Style (Simplest):**
```
RETRIEVE[facts]: Eiffel Tower height
RETRIEVE[notes]: meeting schedule
```

**Function-Style:**
```
retrieve(facts, Eiffel Tower height)
retrieve(notes, meeting schedule)
```

**Legacy (Still Supported):**
```
retrievemem(facts, "Eiffel Tower height")
retrievemem(notes, "meeting schedule")
```

Searches the category for relevant content. Returns:
- Top N most relevant results (configurable)
- Relevance scores (0.0 to 1.0)
- Original content and metadata
- Filtered by similarity threshold

**All syntax formats work identically** - use whichever is most natural for your agent or workflow.

### Error Handling and Feedback

Memory operations **never crash the system**. Instead, agents receive clear feedback:

**Successful Store:**
```
✓ Stored in facts: The Eiffel Tower is 330 meters tall (ID: abc123)
```

**Successful Retrieve:**
```
✓ Retrieved 3 results from facts for query: Eiffel Tower height
  1. [Score: 0.95] The Eiffel Tower is 330 meters tall
  2. [Score: 0.82] Eiffel Tower facts and history
  3. [Score: 0.78] French landmarks and architecture
```

**No Results Found:**
```
✗ No relevant results found in facts for: nonexistent query
```

**Operation Error:**
```
✗ Error in store operation: Invalid category name
💡 Hint: Check that the category name is valid and content/query is properly formatted
```

Memory failures provide specific guidance so agents can retry or adapt their approach.

## Programmatic Usage

### Direct Memory Store API

```python
from pithos.memory_tool import MemoryStore
from pithos.config_manager import ConfigManager

config_manager = ConfigManager()
memory = MemoryStore(config_manager)

# Store single entry
entry_id = memory.store(
    category="python_docs",
    content="Python uses indentation to define code blocks",
    metadata={"source": "manual", "topic": "syntax"}
)

# Store multiple entries
contents = [
    "Python is dynamically typed",
    "Python supports multiple paradigms",
    "Python has a rich standard library"
]
metadatas = [
    {"topic": "types"},
    {"topic": "paradigms"},
    {"topic": "stdlib"}
]
entry_ids = memory.store_batch("python_docs", contents, metadatas)

# Retrieve relevant knowledge
results = memory.retrieve(
    category="python_docs",
    query="What is Python's type system?",
    n_results=5
)

for result in results:
    print(f"[{result.relevance_score:.2f}] {result.content}")
    print(f"  Metadata: {result.metadata}")

# Get category info
info = memory.get_category_info("python_docs")
print(f"Category: {info['name']}, Entries: {info['count']}")

# List all categories
categories = memory.list_categories()
print(f"Available categories: {', '.join(categories)}")

# Delete an entry
success = memory.delete("python_docs", entry_id)

# Delete entire category
success = memory.delete_category("python_docs")
```

### Advanced Features

#### Metadata Filtering

Filter search results by metadata:

```python
results = memory.retrieve(
    category="docs",
    query="syntax rules",
    where={"topic": "syntax", "source": "official"}
)
```

#### Export and Import

Backup and restore categories:

```python
# Export to JSON
memory.export_category("python_docs", "backup.json")

# Import from JSON
category_name = memory.import_category("backup.json")

# Import to a different category
category_name = memory.import_category("backup.json", category="restored_docs")
```

#### Get All Entries

Retrieve all entries without search ranking:

```python
entries = memory.get_all_entries("python_docs")
for entry in entries:
    print(f"{entry['id']}: {entry['content']}")
```

## Configuration

Memory tool configuration is in `configs/tools/memory_config.yaml`:

```yaml
# Enable or disable memory tool
enabled: true

# Directory for persistent vector database storage
persist_directory: "./data/memory"

# Embedding function to use ('default' uses ChromaDB's built-in)
embedding_function: default

# Maximum number of results to return in a single query
max_results: 10

# Similarity threshold for filtering results (0.0 to 1.0)
# Results with relevance_score below this threshold are filtered out
similarity_threshold: 0.5

# Default metadata to attach to all entries
default_metadata:
  source: "pithos"
  version: "1.0"

# Predefined categories (optional)
categories:
  docs:
    description: "Technical documentation and guides"
    max_entries: 1000
  
  code:
    description: "Code examples and snippets"
    max_entries: 500
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable memory tool |
| `persist_directory` | string | `"./data/memory"` | Directory for database files |
| `embedding_function` | string | `"default"` | Embedding model to use |
| `max_results` | integer | `10` | Maximum search results |
| `similarity_threshold` | float | `0.5` | Minimum relevance score to include |
| `default_metadata` | object | `{}` | Metadata added to all entries |

### Similarity Threshold

The `similarity_threshold` controls how strict the relevance filtering is:

- **0.0** - Return all results, even tangentially related
- **0.5** - Moderate relevance required (default)
- **0.7** - High relevance required
- **0.9** - Very high relevance (may return few results)
- **1.0** - Exact or near-exact matches only

Higher thresholds reduce noise but may miss relevant information. Lower thresholds retrieve more context but may include irrelevant results.

## Architecture

### Data Flow

```
Agent Response
    ↓
Extract memory operations (regex)
    ↓
Execute operations
    ↓
  Store: Generate ID → Create embeddings → Save to ChromaDB
  Retrieve: Query embeddings → Calculate relevance → Filter by threshold
    ↓
Format results as system message
    ↓
Add to conversation context
    ↓
Agent continues reasoning
```

### Storage Structure

```
persist_directory/
├── chroma.sqlite3          # ChromaDB metadata
└── [collection_uuid]/       # Per-category data
    ├── data_level0.bin      # Vector index
    └── ...
```

Each category is a separate ChromaDB collection with:
- **Documents**: The text content
- **Embeddings**: Vector representations
- **Metadata**: Custom key-value pairs
- **IDs**: Unique identifiers

### Embedding Generation

ChromaDB automatically generates embeddings using its default sentence transformer model. This converts text into numerical vectors that capture semantic meaning, enabling similarity search.

## Use Cases

### 1. Building Domain Knowledge

Store domain-specific information as the agent learns:

```python
# Agent learns facts during conversation
storemem(astronomy, "Mars has two moons: Phobos and Deimos")
storemem(astronomy, "Jupiter is the largest planet in our solar system")
storemem(astronomy, "Saturn's rings are made of ice and rock particles")

# Later, retrieve relevant context
retrievemem(astronomy, "Tell me about planets with moons")
```

### 2. Conversation Memory

Remember user preferences and context:

```python
# Remember user details
storemem(user_profile, "User prefers Python over JavaScript")
storemem(user_profile, "User works on machine learning projects")

# Recall when relevant
retrievemem(user_profile, "What are the user's programming preferences?")
```

### 3. Code Snippet Library

Build a searchable code knowledge base:

```python
# Store useful code patterns
storemem(code_snippets, "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)")
storemem(code_snippets, "async def fetch_data(url): return await client.get(url)")

# Find relevant patterns
retrievemem(code_snippets, "How to implement recursion?")
```

### 4. Documentation Cache

Cache frequently accessed documentation:

```python
# Cache API documentation
storemem(api_docs, "requests.get(url, params, headers) - Send GET request")
storemem(api_docs, "json.dumps(obj, indent) - Serialize object to JSON")

# Quick lookup
retrievemem(api_docs, "How to send HTTP GET request?")
```

## Best Practices

### 1. Organize by Category

Use clear, descriptive category names that reflect the knowledge domain:

✅ **Good:**
- `python_syntax`
- `user_preferences`
- `project_requirements`

❌ **Avoid:**
- `misc`
- `data`
- `temp`

### 2. Include Rich Metadata

Add metadata to improve filtering and context:

```python
memory.store(
    "documentation",
    "The Flask route decorator registers URL patterns",
    metadata={
        "framework": "flask",
        "topic": "routing",
        "difficulty": "beginner",
        "source": "official_docs"
    }
)
```

### 3. Write Complete, Self-Contained Content

Each stored entry should be understandable on its own:

✅ **Good:**
```python
storemem(facts, "Python was created by Guido van Rossum and released in 1991")
```

❌ **Avoid:**
```python
storemem(facts, "Created by Guido in 1991")  # Missing context
```

### 4. Regular Cleanup

Periodically review and clean up outdated or duplicate information:

```python
# Export for review
memory.export_category("old_notes", "review.json")

# After cleanup, delete old category
memory.delete_category("old_notes")

# Import cleaned data
memory.import_category("cleaned_notes.json", category="notes")
```

### 5. Use Appropriate Thresholds

Adjust similarity thresholds based on your use case:

- **Broad research**: 0.5-0.6 (capture more context)
- **General use**: 0.7 (balanced)
- **Fact lookup**: 0.8-0.9 (high precision)

---

## Automatic Context Compaction

As a conversation grows, the message history can exceed the model's effective context window, slowing inference and degrading response quality. **Automatic context compaction** solves this by monitoring history length and replacing the oldest messages with a concise LLM-generated summary when a threshold is reached.

### How It Works

1. After each agent response, the compactor checks whether `len(message_history) >= threshold`.
2. It identifies _compactable_ messages — everything except protected entries (auto-recall injections and prior summaries).
3. The oldest compactable messages, except the last `keep_last`, are fed to the LLM in a single summarisation call.
4. The summary is stored in the vector memory (if enabled) under `memory_category` for future recall.
5. The compacted messages are removed and replaced with a single `[CONTEXT SUMMARY]` system message that is itself protected from future compaction.

### Quick Start

```python
from pithos import OllamaAgent, ConfigManager
from pithos.agent.compaction import CompactionConfig

config_manager = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")
agent.enable_memory(config_manager)  # optional — enables archiving summaries

agent.enable_compaction(CompactionConfig(
    threshold=20,       # compact when history reaches 20 messages
    keep_last=6,        # always keep the 6 most-recent messages intact
    summary_model=None, # None = use agent's default_model
))

# Now just chat normally — compaction runs automatically
for _ in range(30):
    agent.send("Tell me something interesting about astronomy.")
```

### YAML Configuration

```yaml
# configs/agents/my-agent.yaml
default_model: glm-4.7-flash
compaction:
  enabled: true
  threshold: 20           # message count that triggers compaction
  keep_last: 6            # most-recent messages to leave untouched
  summary_model: null     # null = use default_model
  memory_category: context_summaries
  summary_max_tokens: 512
```

### CompactionConfig Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `threshold` | int | `20` | Total message count before compaction runs |
| `keep_last` | int | `6` | Most-recent compactable messages to preserve |
| `summary_model` | str \| None | `None` | Ollama model for summarisation; falls back to `default_model` |
| `memory_category` | str | `"context_summaries"` | ChromaDB category for archived summaries |
| `summary_max_tokens` | int | `512` | Max output tokens for the summary response |

### Summary Format

The compactor asks the model to respond in a structured two-line format:

```
Summary: <concise paragraph summarising the compacted turns>
Entities: <comma-separated list of important named entities not already named in the summary>
```

The resulting `[CONTEXT SUMMARY]` system message injected into history looks like:

```
[CONTEXT SUMMARY]
Discussion covered setting up a Python project, installing dependencies, and
resolving a version conflict with numpy. A fix using a virtual environment was agreed upon.

Key entities: numpy, pip, venv, requirements.txt
```

### API Reference

```python
# Enable with default settings
agent.enable_compaction()

# Enable with custom settings
from pithos.agent.compaction import CompactionConfig
agent.enable_compaction(CompactionConfig(threshold=15, keep_last=4))

# Disable
agent.disable_compaction()

# Check status
print(agent.compaction_enabled)   # True / False
print(agent._compactor.config.threshold)
```

---

## Automatic Memory Recall

**Automatic recall** retrieves relevant memories and prior conversation snippets before each user turn and injects them as context. The agent therefore "remembers" past interactions without you needing to manually manage retrieval.

### How It Works

1. Before the user message is appended to the context, the recaller makes an *ephemeral* LLM call (not stored in history) asking: "Given this conversation so far and the incoming message, what should I search for?"
2. The model returns 1–3 concise search queries.
3. These queries are run against the configured sources (vector memory and/or conversation history).
4. Snippets scoring above `min_relevance` are collected, deduplicated, and injected into the conversation as a `[RECALLED CONTEXT]` system message at position 0.
5. Any previously injected recall message is replaced, so at most one is present at any time.
6. The recall message is tagged `_pithos_no_compact` so it is never included in compaction candidates.

### Quick Start

```python
from pithos import OllamaAgent, ConfigManager
from pithos.agent.recall import RecallConfig

config_manager = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")
agent.enable_memory(config_manager)    # recall needs a source to search
agent.enable_history()                  # optional second source

agent.enable_recall(RecallConfig(
    sources=["memory", "history"],  # search both sources
    n_results=5,                    # max snippets to inject
    min_relevance=0.5,              # relevance filter (0–1)
))

# The agent now surfaces relevant memories automatically
agent.send("What did we decide about the database schema last time?")
# → recall injects matching history snippets before the LLM sees the question
```

### YAML Configuration

```yaml
# configs/agents/my-agent.yaml
default_model: glm-4.7-flash
recall:
  enabled: true
  sources:
    - memory      # ChromaDB vector memory store
    - history     # SQLite + vector conversation history
  n_results: 5
  recall_model: null   # null = use default_model
  categories: []       # empty = search all memory categories
  min_relevance: 0.5
```

### RecallConfig Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `sources` | list[str] | `["memory", "history"]` | Data sources to search (`"memory"` and/or `"history"`) |
| `n_results` | int | `5` | Maximum snippets injected per turn |
| `recall_model` | str \| None | `None` | Model for query generation; falls back to `default_model` |
| `categories` | list[str] | `[]` | Memory categories to search; empty = all |
| `min_relevance` | float | `0.5` | Minimum relevance score (0–1) to include a snippet |

### Injected Message Format

The recall message appears as the first item in `message_history` and looks like:

```
[RECALLED CONTEXT]
The following memories were automatically retrieved as relevant context:

1. [memory] The project uses PostgreSQL 15 with pgvector enabled.
2. [memory] Database migrations are managed by Alembic.
3. [history] [user] What is the schema for the conversations table?
4. [history] [assistant] The conversations table has columns: id, session_id, agent_name, ...
```

### Sources

| Source | Requires | Searches |
|---|---|---|
| `"memory"` | `agent.enable_memory(...)` | Vector memory store (ChromaDB) |
| `"history"` | `agent.enable_history(...)` | Persistent conversation log (ChromaDB + SQLite FTS5) |

If a source is configured but its backing store is not enabled on the agent, it is silently skipped.

### API Reference

```python
# Enable with default settings
agent.enable_recall()

# Enable with custom settings
from pithos.agent.recall import RecallConfig
agent.enable_recall(RecallConfig(sources=["memory"], n_results=3))

# Disable
agent.disable_recall()

# Check status
print(agent.recall_enabled)   # True / False
```

### Combining Compaction and Recall

Compaction and recall work seamlessly together. Recall injections are protected from compaction, and compaction summaries are stored in the memory store where recall can later find them:

```python
agent.enable_memory(config_manager)
agent.enable_history()
agent.enable_compaction(CompactionConfig(threshold=20, keep_last=6))
agent.enable_recall(RecallConfig(sources=["memory", "history"], n_results=4))
```

Each turn the agent:
1. Retrieves relevant prior context and injects it at the top of history
2. Responds normally
3. If history is now too long, compacts the oldest messages and archives the summary to memory — ready to be recalled on the next relevant turn

---

## Troubleshooting

### ChromaDB Not Installed

**Error:** `RuntimeError: ChromaDB is not installed`

**Solution:**
```bash
pip install chromadb
```

### Empty Search Results

**Problem:** Queries return no results even when data exists

**Possible causes:**
1. Similarity threshold too high → Lower threshold in config
2. Query too specific → Use broader query terms
3. Wrong category → Check `memory.list_categories()`

**Debug:**
```python
# Check data exists
entries = memory.get_all_entries("your_category")
print(f"Found {len(entries)} entries")

# Try with lower threshold
results = memory.retrieve("your_category", "query", n_results=10)
print(f"Found {len(results)} results")
```

### Performance Issues

**Problem:** Slow search or storage operations

**Solutions:**
1. **Reduce max_results** - Limit to top 5-10 results
2. **Use metadata filters** - Narrow search scope with `where`
3. **Batch operations** - Use `store_batch()` instead of multiple `store()` calls
4. **Smaller categories** - Split large categories into focused subcategories

### Data Migration

**Need to move data to a new location:**

```python
# Export all categories
old_store = MemoryStore(persist_directory="./old_path")
for category in old_store.list_categories():
    old_store.export_category(category, f"{category}.json")

# Import to new location
new_store = MemoryStore(persist_directory="./new_path")
for category in old_categories:
    new_store.import_category(f"{category}.json")
```

## API Reference

### MemoryStore Class

```python
class MemoryStore:
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        persist_directory: Optional[str] = None
    )
    
    def store(
        self,
        category: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> str
    
    def store_batch(
        self,
        category: str,
        contents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None
    ) -> list[str]
    
    def retrieve(
        self,
        category: str,
        query: str,
        n_results: Optional[int] = None,
        where: Optional[dict[str, Any]] = None
    ) -> list[SearchResult]
    
    def delete(self, category: str, entry_id: str) -> bool
    
    def delete_category(self, category: str) -> bool
    
    def list_categories(self) -> list[str]
    
    def get_category_info(self, category: str) -> dict[str, Any]
    
    def get_all_entries(self, category: str) -> list[dict[str, Any]]
    
    def export_category(self, category: str, output_path: str) -> None
    
    def import_category(
        self,
        input_path: str,
        category: Optional[str] = None
    ) -> str
    
    def clear_all(self) -> None
```

### OllamaAgent Memory Methods

```python
class OllamaAgent:
    def enable_memory(
        self,
        config_manager: ConfigManager,
        persist_directory: Optional[str] = None
    ) -> None
    
    # Internal methods (not typically called directly)
    def _extract_memory_ops(self, content: str) -> list[dict[str, Any]]
    def _execute_memory_ops(self, operations: list[dict[str, Any]]) -> str
```

### Data Classes

```python
@dataclass
class MemoryEntry:
    id: str
    category: str
    content: str
    metadata: dict[str, Any]
    timestamp: str
    embedding: Optional[list[float]] = None

@dataclass
class SearchResult:
    id: str
    category: str
    content: str
    metadata: dict[str, Any]
    distance: float
    relevance_score: float  # 0.0 to 1.0, higher is better
```

## Examples

### Example 1: Building a Learning Agent

```python
from pithos import OllamaAgent, ConfigManager

config_manager = ConfigManager()
agent = OllamaAgent("glm-4.7-flash")
agent.enable_memory(config_manager)

# Teach the agent facts
agent.send("Learn this: Python uses duck typing.")
agent.send("Remember: Python's GIL prevents true parallelism.")
agent.send("Store this: List comprehensions are faster than loops.")

# The agent can recall and combine knowledge
response = agent.send("What do you know about Python's performance?")
# Agent retrieves relevant facts and synthesizes an answer
```

### Example 2: Multi-Category Knowledge Base

```python
from pithos.memory_tool import MemoryStore

memory = MemoryStore()

# Organize by programming language
memory.store("python", "Python uses significant whitespace")
memory.store("python", "Python has dynamic typing")
memory.store("javascript", "JavaScript is single-threaded")
memory.store("javascript", "JavaScript uses prototype-based inheritance")
memory.store("rust", "Rust guarantees memory safety without garbage collection")

# Query specific categories
python_info = memory.retrieve("python", "type system")
js_info = memory.retrieve("javascript", "concurrency")
rust_info = memory.retrieve("rust", "memory management")
```

### Example 3: Context-Aware Assistant

```python
agent = OllamaAgent("glm-4.7-flash")
agent.enable_memory(ConfigManager())

# First interaction
agent.send("I'm working on a Flask web app")
# Agent: storemem(context, "User is working on a Flask web application")

agent.send("I need to add authentication")
# Agent: storemem(context, "User needs to implement authentication in Flask app")

# Later interaction
agent.send("How should I structure my project?")
# Agent: retrievemem(context, "user project details")
# System provides previous context
# Agent: "Based on your Flask web app with authentication needs, I recommend..."
```

## See Also

- [Tool Calling Guide](TOOL_CALLING.md) - General tool system
- [Agent Guide](../README.md) - Agent creation and management
- [Configuration Guide](CONFIG.md) - Configuration system
- [ChromaDB Documentation](https://docs.trychroma.com/) - Vector database details
