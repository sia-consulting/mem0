# Phase 4: Direct Node/Edge Creation API

> **Before starting:** Look up what has been implemented in the previous phases. Inspect the actual codebase — don't assume anything was implemented as originally planned.

## Goal

Allow agents to **directly create nodes, edges, and properties** without going through LLM extraction. Currently, all memory goes through an LLM pipeline that extracts entities and relationships. This phase adds a direct, structured API for precise graph manipulation.

## Current State (Inspect Before Starting)

Check these files:
- `mem0/memory/graph_memory.py` — `add()`, `_add_entities()`
- `mem0/memory/main.py` — `add()` with `infer=True/False`
- Any properties/walking methods from Phases 1-3
- The "Me" anchor node from Phase 2

Currently, `add(data, filters)` always runs LLM entity extraction. The `infer=False` flag on `Memory.add()` only affects vector store behavior.

## Changes Required

### 1. Add `add_node()` Method to MemoryGraph

```python
def add_node(self, name, filters, entity_type=None, properties=None, source_node=None, relationship=None):
    """Directly add a node to the graph.
    
    Args:
        name: Node name/identifier
        filters: Scope filters (user_id, etc.)
        entity_type: Optional type label for the node
        properties: Optional dict of key-value properties
        source_node: Required - name of existing node to connect FROM
        relationship: Required if source_node given - relationship type for the edge
    
    Returns:
        dict with created node info
    """
```

The node MUST be connected to the graph (enforced via required `source_node`). If `source_node` is not provided, connect to the "Me" anchor node (from Phase 2, if implemented).

### 2. Add `add_edge()` Method to MemoryGraph

```python
def add_edge(self, source, destination, relationship, filters, properties=None):
    """Directly add an edge between two existing nodes.
    
    Args:
        source: Source node name
        destination: Destination node name
        relationship: Relationship type
        filters: Scope filters
        properties: Optional dict of edge properties
    
    Returns:
        dict with created edge info
        
    Raises:
        ValueError if source or destination node doesn't exist
    """
```

### 3. Add `update_node_properties()` Method

```python
def update_node_properties(self, node_name, filters, properties):
    """Update properties on an existing node.
    
    Args:
        node_name: Name of the node to update
        filters: Scope filters
        properties: Dict of properties to set (merges with existing)
    """
```

### 4. Add `update_edge_properties()` Method

```python
def update_edge_properties(self, source, destination, relationship, filters, properties):
    """Update properties on an existing edge."""
```

### 5. Expose Through Memory Class

Add public methods to `Memory` (main.py):

```python
def graph_add_node(self, name, *, user_id=None, agent_id=None, run_id=None, 
                   entity_type=None, properties=None, source_node=None, relationship=None):
    """Directly add a node to the agent's graph."""

def graph_add_edge(self, source, destination, relationship, *,
                   user_id=None, agent_id=None, run_id=None, properties=None):
    """Directly add an edge between existing nodes."""

def graph_update_node(self, node_name, properties, *,
                      user_id=None, agent_id=None, run_id=None):
    """Update properties on a graph node."""

def graph_update_edge(self, source, destination, relationship, properties, *,
                      user_id=None, agent_id=None, run_id=None):
    """Update properties on a graph edge."""
```

### 6. Validation

- Node names must be non-empty strings
- Properties must be a flat dict (no nested objects) — Cypher doesn't support nested properties
- Relationship types are sanitized using `sanitize_relationship_for_cypher()`
- Source/destination nodes must exist in the graph (or be the "Me" node)

## Testing

- Test `add_node()` creates node with properties
- Test `add_node()` connects to source node
- Test `add_node()` defaults to "Me" node if no source given
- Test `add_edge()` creates edge with properties between existing nodes
- Test `add_edge()` fails if nodes don't exist
- Test `update_node_properties()` merges properties
- Test `update_edge_properties()` merges properties
- Test that direct API nodes are found by `search()` and `walk()`

## Files Modified

- `mem0/memory/graph_memory.py`
- `mem0/memory/main.py`
- `mem0/memory/utils.py` (validation helpers)
- `tests/memory/test_direct_graph_api.py` (new)
