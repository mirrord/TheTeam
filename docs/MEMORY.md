# Memory Tool Guide

The pithos Memory Tool provides vector database-based knowledge storage and retrieval for agents. It enables agents to build up domain knowledge over time, store important facts, and retrieve relevant context from previous interactions.

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
agent = OllamaAgent("llama3.2")

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
agent = OllamaAgent("llama3.2")
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
agent = OllamaAgent("llama3.2")
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
