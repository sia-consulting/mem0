# Phase 3: Multi-Hop Graph Walking

> **Before starting:** Look up what has been implemented in the previous phases. Inspect the actual codebase — don't assume anything was implemented as originally planned.

## Goal

Enable **multi-hop traversal** of the graph so agents can "walk" from one node to its neighbors, then to their neighbors, etc. Currently, search only returns immediate (1-hop) neighbors. This phase adds methods to explore the graph structure iteratively.

## Current State (Inspect Before Starting)

Check these files:
- `mem0/memory/graph_memory.py` — `_search_graph_db()` (single-hop MATCH), `get_all()`, `search()`
- The "Me" anchor node (if implemented in Phase 2)
- Any properties support added in Phase 1

Currently, `_search_graph_db()` does:
```cypher
MATCH (n)-[r]->(m)  -- one hop only
```

## Changes Required

### 1. Add `get_node()` Method

Retrieve a specific node by name or ID with all its properties:

```python
def get_node(self, node_name, filters):
    """Get a single node and its properties."""
```

Returns:
```python
{
    "name": "supervisor",
    "entity_type": "person",
    "properties": {"email": "sia@sia-consulting.eu"},
    "edge_count": 5
}
```

### 2. Add `get_neighbors()` Method

Get all nodes directly connected to a given node (1-hop):

```python
def get_neighbors(self, node_name, filters, direction="both", relationship_type=None, limit=50):
    """Get nodes connected to the given node.
    
    Args:
        node_name: The node to start from
        filters: Scope filters (user_id, etc.)
        direction: "outgoing", "incoming", or "both"
        relationship_type: Optional filter by relationship type
        limit: Max results
    """
```

Returns list of:
```python
{
    "node": {"name": "project_alpha", "properties": {...}},
    "relationship": "supervises",
    "edge_properties": {"since": "2024-01"},
    "direction": "outgoing"
}
```

### 3. Add `walk()` Method

Walk the graph from a starting node up to N hops:

```python
def walk(self, start_node, filters, depth=2, relationship_types=None, limit=100):
    """Walk the graph from start_node up to `depth` hops.
    
    Args:
        start_node: Node name to start from
        filters: Scope filters
        depth: Maximum hops (1-5, default 2)
        relationship_types: Optional list of relationship types to follow
        limit: Max total results
    """
```

Uses Cypher variable-length paths:
```cypher
MATCH path = (start {name: $start_name, user_id: $user_id})-[*1..N]-(end)
WHERE ALL(r IN relationships(path) WHERE r.valid IS NULL OR r.valid = true)
RETURN [n IN nodes(path) | n.name] AS node_names,
       [r IN relationships(path) | type(r)] AS relationships,
       length(path) AS depth
ORDER BY depth ASC
LIMIT $limit
```

### 4. Add `find_path()` Method

Find the shortest path between two nodes:

```python
def find_path(self, from_node, to_node, filters, max_depth=5):
    """Find shortest path between two nodes."""
```

Uses:
```cypher
MATCH path = shortestPath((a {name: $from})-[*..N]-(b {name: $to}))
```

### 5. Add `get_edges()` Method

Get all edges for a node, with optional filtering:

```python
def get_edges(self, node_name, filters, relationship_type=None, direction="both"):
    """Get all edges connected to a node with their properties."""
```

### 6. Expose Through Memory Class

Add public methods to `Memory` (main.py):
```python
def graph_get_node(self, node_name, **kwargs)
def graph_get_neighbors(self, node_name, **kwargs)
def graph_walk(self, start_node, **kwargs)
def graph_find_path(self, from_node, to_node, **kwargs)
def graph_get_edges(self, node_name, **kwargs)
```

These are thin wrappers that validate filters and delegate to `self.graph.*()`.

## Testing

- Test `get_node()` returns correct properties
- Test `get_neighbors()` with direction and relationship filters
- Test `walk()` respects depth limits
- Test `walk()` doesn't follow soft-deleted edges
- Test `find_path()` finds shortest path
- Test `get_edges()` returns edge properties
- Test that all methods respect user_id/agent_id scoping

## Files Modified

- `mem0/memory/graph_memory.py`
- `mem0/memory/main.py`
- `tests/memory/test_graph_walking.py` (new)
