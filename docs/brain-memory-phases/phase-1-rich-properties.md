# Phase 1: Rich Properties on Nodes and Edges

> **Before starting:** Look up what has been implemented in the previous phases. Inspect the actual codebase — don't assume anything was implemented as originally planned.

## Goal

Add support for **arbitrary key-value properties** on graph nodes and edges, beyond the current fixed set (`name`, `embedding`, `mentions`, `created` on nodes; `relationship`, `mentions`, `valid` on edges).

For example, a "Supervisor" node should be able to hold `{ "name": "Sia Ghassemi", "email": "sia@sia-consulting.eu" }`, and an edge connecting "customer" to "my-company" should carry `{ "meet_via": "email", "introduced_by": "sia_ghassemi" }`.

## Current State (Inspect Before Starting)

Check these files to understand the current node/edge structure:

- `mem0/memory/graph_memory.py` — `_add_entities()` method, Cypher MERGE queries
- `mem0/graphs/tools.py` — LLM tool schemas (`EXTRACT_ENTITIES_TOOL`, `RELATIONS_TOOL`, `ADD_MEMORY_TOOL_GRAPH`)
- `mem0/graphs/utils.py` — `EXTRACT_RELATIONS_PROMPT`
- `mem0/memory/utils.py` — `remove_spaces_from_entities()`, `format_entities()`

## Changes Required

### 1. Extend LLM Tool Schemas (`mem0/graphs/tools.py`)

**EXTRACT_ENTITIES_TOOL** — Add a `properties` field to each entity:
```json
{
  "entity": "sia_ghassemi",
  "entity_type": "person",
  "properties": {
    "email": "sia@sia-consulting.eu",
    "role": "supervisor"
  }
}
```

**RELATIONS_TOOL** — Add a `properties` field to each relationship:
```json
{
  "source": "sia_ghassemi",
  "relationship": "supervises",
  "destination": "project_alpha",
  "properties": {
    "since": "2024-01",
    "capacity": "lead"
  }
}
```

Update both regular and `_STRUCT_TOOL` variants.

### 2. Update Entity Extraction Prompts (`mem0/graphs/utils.py`)

Update `EXTRACT_RELATIONS_PROMPT` to instruct the LLM to also extract meaningful properties for entities and relationships. Properties should be key-value pairs that capture important attributes that don't fit into the entity name or relationship type.

### 3. Update Graph Memory (`mem0/memory/graph_memory.py`)

**`_retrieve_nodes_from_data()`** — Parse and return properties alongside `entity_type_map`. The return type changes from `{entity: type}` to `{entity: {"type": type, "properties": {...}}}`.

**`_establish_nodes_relations_from_data()`** — Parse and return properties on relationships. Each entity dict gains a `properties` key.

**`_add_entities()`** — In the Cypher MERGE queries, SET arbitrary properties on nodes and edges:
```cypher
MERGE (source:__Entity__ {name: $source_name, user_id: $user_id})
ON CREATE SET source.created = timestamp(), source.custom_prop1 = $prop1, ...
ON MATCH SET source.custom_prop1 = $prop1, ...
```

Use Cypher's `SET n += $props` syntax for dynamic property maps.

**`_search_graph_db()`** — Return node and edge properties in search results:
```cypher
RETURN n.name AS source, type(r) AS relationship, m.name AS destination,
       properties(n) AS source_properties, properties(r) AS edge_properties, properties(m) AS destination_properties
```

**`get_all()`** — Also return properties in results.

### 4. Update Utilities (`mem0/memory/utils.py`)

**`remove_spaces_from_entities()`** — Preserve the `properties` field when normalizing entities.

**`format_entities()`** — Include properties in the formatted string output.

### 5. Update Search Results Format

The search and get_all results should include properties:
```python
{
    "source": "sia_ghassemi",
    "source_properties": {"email": "sia@sia-consulting.eu"},
    "relationship": "supervises",
    "edge_properties": {"since": "2024-01"},
    "destination": "project_alpha",
    "destination_properties": {}
}
```

## Testing

- Unit test that entities extracted by LLM include properties
- Unit test that `_add_entities()` stores properties in the graph
- Unit test that `search()` returns properties
- Unit test that `get_all()` returns properties
- Ensure backward compatibility: calls without properties still work

## Files Modified

- `mem0/graphs/tools.py`
- `mem0/graphs/utils.py`
- `mem0/memory/graph_memory.py`
- `mem0/memory/utils.py`
- `tests/memory/test_graph_memory_properties.py` (new)
