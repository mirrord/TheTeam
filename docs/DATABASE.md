# Database Management

This document describes the database management features in TheTeam/pithos, including storage, retrieval, searching, and clearing operations across all database systems.

## Overview

TheTeam uses multiple databases for different purposes:

1. **Memory Store (Vector Database)** - ChromaDB-based vector storage for knowledge categorization and semantic search
2. **Conversation History** - SQLite + ChromaDB for persistent agent conversation history with full-text and semantic search
3. **Flowchart Store** - SQLite + ChromaDB for flowchart configurations with metadata (tags, notes, descriptions)

The `DatabaseManager` class provides a unified interface for managing all databases.

## Installation

The core database functionality requires:
- SQLite (bundled with Python)
- ChromaDB (optional but recommended for semantic search)

```bash
pip install chromadb
```

## Database Manager

### Basic Usage

```python
from pithos.database_manager import DatabaseManager

# Initialize manager
db_manager = DatabaseManager(
    memory_dir="./data/memory",
    history_dir="./data/conversations",
    flowchart_dir="./data/flowcharts"
)

# Get database information
info_list = db_manager.get_database_info()
for info in info_list:
    print(f"{info.name}: {info.size_bytes} bytes")

# Close connections when done
db_manager.close()
```

### Clearing Databases

```python
# Clear individual databases
db_manager.clear_memory()
db_manager.clear_history()
db_manager.clear_flowcharts()

# Clear all databases (requires confirmation)
results = db_manager.clear_all(confirm=True)
```

### Universal Search

Search across all databases at once:

```python
# Semantic search
results = db_manager.search_all(query="authentication", semantic=True, limit=10)

for db_name, db_results in results.items():
    print(f"\n{db_name}:")
    for result in db_results:
        print(f"  - {result.content[:100]}...")
        print(f"    Relevance: {result.relevance_score:.3f}")
```

Exact text search:

```python
# Search all databases
results = db_manager.search_exact(text="specific_keyword")

# Search specific databases only
results = db_manager.search_exact(
    text="specific_keyword",
    databases=["history", "flowcharts"]
)
```

## Flowchart Store

The Flowchart Store provides persistent storage for flowchart configurations with rich metadata support.

### Storing Flowcharts

```python
from pithos.flowchart_store import FlowchartStore

store = FlowchartStore()

# Store a flowchart
config = {
    "nodes": {
        "node1": {"type": "agent", "label": "Main Agent"},
        "node2": {"type": "router", "label": "Router"}
    },
    "edges": [
        {"from": "node1", "to": "node2", "condition": {"type": "AlwaysCondition"}}
    ],
    "start_node": "node1"
}

flowchart_id = store.store_flowchart(
    name="Authentication Flow",
    config=config,
    description="Handles user authentication and authorization",
    notes="Updated to support OAuth2",
    tags=["auth", "security", "production"]
)
```

### Retrieving Flowcharts

```python
# Get by ID
flowchart = store.get_flowchart(flowchart_id)
print(f"Name: {flowchart.name}")
print(f"Description: {flowchart.description}")
print(f"Tags: {flowchart.tags}")
print(f"Config: {flowchart.config}")

# List all flowcharts
flowcharts = store.list_flowcharts()

# List with tag filter
production_flowcharts = store.list_flowcharts(tags=["production"])

# List with limit
recent_flowcharts = store.list_flowcharts(limit=10)
```

### Searching Flowcharts

```python
# Text search
results = store.search_text(query="authentication")

# Semantic search (requires ChromaDB)
results = store.search_semantic(query="user login and security")

# Automatic mode selection
results = store.search(
    query="payment processing",
    semantic=True,  # Use semantic if available
    limit=10
)

# Exact text match
results = store.search_exact(text="OAuth2")

# Filter by tags
results = store.search(
    query="data processing",
    tags=["production", "critical"],
    limit=5
)
```

### Managing Metadata

```python
# Update notes
store.update_notes(flowchart_id, "Added support for multi-factor authentication")

# Add tags
store.add_tags(flowchart_id, ["mfa", "2fa"])

# Remove tags
store.remove_tags(flowchart_id, ["deprecated"])

# List all tags with counts
tags = store.list_tags()
for tag, count in tags:
    print(f"{tag}: {count} flowcharts")
```

### Import/Export

```python
# Export to YAML
store.export_flowchart(flowchart_id, "auth_flow.yaml")

# Import from YAML
new_id = store.import_flowchart("auth_flow.yaml")

# Import with specific ID
new_id = store.import_flowchart("auth_flow.yaml", flowchart_id="custom_id")
```

## Memory Store

The Memory Store has been enhanced with universal search capabilities.

### Search All Categories

```python
from pithos.tools.memory_tool import MemoryStore

memory = MemoryStore()

# Search across all categories
results = memory.search_all_categories(
    query="machine learning",
    n_results=10,
    min_relevance=0.7
)

for category, category_results in results.items():
    print(f"\n{category}:")
    for result in category_results:
        print(f"  - {result.content[:100]}...")
        print(f"    Score: {result.relevance_score:.3f}")
```

### Exact Text Search

```python
# Search for exact text across all categories
results = memory.search_exact(text="neural network")

# Search specific categories only
results = memory.search_all_categories(
    query="deep learning",
    categories=["ml", "research", "papers"]
)
```

### Clear All Memory

```python
# Clear all knowledge categories
memory.clear_all()
```

## Conversation History

The Conversation History store now supports clearing and exact text search.

### Clear History

```python
from pithos.agent.history import ConversationStore

history = ConversationStore()

# Clear all conversation history
history.clear_all()
```

### Exact Text Search

```python
# Search for exact text in messages
messages = history.search_exact(
    text="authentication error",
    agent_name="support_agent",
    session_id="session_123"
)
```

## CLI Commands

### Database Manager CLI

```bash
# Show database information
pithos-database info

# Clear specific database
pithos-database clear memory --confirm
pithos-database clear history --confirm
pithos-database clear flowcharts --confirm

# Clear all databases
pithos-database clear all --confirm

# Search across all databases
pithos-database search "authentication"

# Exact text search
pithos-database search "specific_keyword" --exact

# Limit results
pithos-database search "machine learning" --limit 20
```

### Flowchart Store CLI

```bash
# Import flowchart
pithos-flowcharts import config.yaml

# Export flowchart
pithos-flowcharts export fc_123 output.yaml

# List flowcharts
pithos-flowcharts list
pithos-flowcharts list --tags production,critical
pithos-flowcharts list --limit 10

# Get flowchart
pithos-flowcharts get fc_123

# Search flowcharts
pithos-flowcharts search "authentication"
pithos-flowcharts search "user login" --exact
pithos-flowcharts search "payment" --tags production

# Manage tags
pithos-flowcharts tags
pithos-flowcharts add-tags fc_123 "new-tag,another-tag"

# Update notes
pithos-flowcharts notes fc_123 "Updated implementation notes"

# Delete flowchart
pithos-flowcharts delete fc_123

# Clear all flowcharts
pithos-flowcharts clear --confirm
```

## Web API Endpoints

### Database Info

```http
GET /api/database/info
```

Returns information about all databases.

### Clear Database

```http
POST /api/database/clear/{database}
Content-Type: application/json

{
  "confirm": true
}
```

Database can be: `memory`, `history`, `flowcharts`, or `all`.

### Universal Search

```http
POST /api/database/search
Content-Type: application/json

{
  "query": "authentication",
  "exact": false,
  "semantic": true,
  "limit": 10
}
```

### Memory Categories

```http
GET /api/database/memory/categories
```

### Memory Search

```http
POST /api/database/memory/search
Content-Type: application/json

{
  "query": "machine learning",
  "n_results": 10,
  "min_relevance": 0.7,
  "categories": ["ml", "research"]
}
```

### Flowchart Management

```http
# List flowcharts
GET /api/database/flowcharts?tags=production&limit=10

# Get flowchart
GET /api/database/flowcharts/{flowchart_id}

# Store flowchart
POST /api/database/flowcharts
Content-Type: application/json

{
  "name": "Authentication Flow",
  "config": {...},
  "description": "User authentication",
  "notes": "Implementation notes",
  "tags": ["auth", "security"]
}

# Update notes
PUT /api/database/flowcharts/{flowchart_id}
Content-Type: application/json

{
  "notes": "Updated notes"
}

# Add tags
POST /api/database/flowcharts/{flowchart_id}/tags
Content-Type: application/json

{
  "tags": ["new-tag", "another-tag"]
}

# Delete flowchart
DELETE /api/database/flowcharts/{flowchart_id}

# Search flowcharts
POST /api/database/flowcharts/search
Content-Type: application/json

{
  "query": "authentication",
  "exact": false,
  "semantic": true,
  "tags": ["production"],
  "limit": 10
}

# List tags
GET /api/database/flowcharts/tags
```

## Best Practices

### 1. Regular Backups

```python
# Export important flowcharts
for flowchart in store.list_flowcharts(tags=["production"]):
    store.export_flowchart(
        flowchart.id,
        f"backups/{flowchart.name}.yaml"
    )

# Export memory categories
memory.export_category("critical_knowledge", "backups/knowledge.json")
```

### 2. Use Tags Effectively

```python
# Tag by environment
store.add_tags(flowchart_id, ["production", "v2.0"])

# Tag by feature area
store.add_tags(flowchart_id, ["auth", "payments", "notifications"])

# Tag by status
store.add_tags(flowchart_id, ["active", "tested"])
```

### 3. Structured Notes

Use consistent formatting in notes:

```python
notes = """
## Implementation Details
- Added OAuth2 support (2024-03-15)
- Integrated with Auth0 (2024-03-20)

## Known Issues
- Rate limiting not implemented

## TODO
- Add refresh token rotation
- Implement session management
"""

store.update_notes(flowchart_id, notes)
```

### 4. Clean Up Regularly

```python
# Delete old test flowcharts
test_flowcharts = store.list_flowcharts(tags=["test", "temporary"])
for fc in test_flowcharts:
    if is_old(fc.updated_at):
        store.delete_flowchart(fc.id)
```

### 5. Close Connections

Always close database connections when done:

```python
try:
    db_manager = DatabaseManager()
    # ... use db_manager ...
finally:
    db_manager.close()
```

Or use context managers when available:

```python
with get_database_manager() as db_manager:
    # ... use db_manager ...
    pass  # Automatically closed
```

## Performance Considerations

### Vector Search

- Vector embeddings are computed on-demand
- First search after storage may be slower
- Results improve with more data
- Consider using `min_relevance` to filter low-quality matches

### Text Search

- FTS5 full-text search is fast and always available
- Use for exact matches or when semantic meaning isn't critical
- Falls back to LIKE queries if FTS fails

### Caching

- Memory store uses LRU caching for collections
- Default TTL: 300 seconds (5 minutes)
- Default max size: 50 collections
- Configure via `memory_config.yaml`

## Troubleshooting

### ChromaDB Not Available

If ChromaDB is not installed:
- Vector/semantic search falls back to text search
- All other features work normally
- Install with: `pip install chromadb`

### Database Locked Errors

If you see SQLite "database is locked" errors:
- Ensure only one process accesses the database
- Close connections properly with `.close()`
- Check for zombie processes holding locks

### Search Returns No Results

If searches return empty results:
- Vector embeddings may need time to process
- Try increasing `n_results` or decreasing `min_relevance`
- Use exact text search as fallback
- Verify data was actually stored

### Clear Operation Fails

If clear operations fail:
- Ensure no other processes are accessing databases
- Close all connections first
- Check file permissions
- On Windows, ChromaDB may need system.stop() first

## See Also

- [MEMORY.md](MEMORY.md) - Memory and recall system
- [FLOWCHARTS.md](FLOWCHARTS.md) - Flowchart execution
- [CONFIG.md](CONFIG.md) - Configuration system
- [TOOL_CALLING.md](TOOL_CALLING.md) - Tool system
