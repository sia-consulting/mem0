# Phase 2: "Me" Anchor Node + Per-Agent Graphs

> **Before starting:** Look up what has been implemented in the previous phases. Inspect the actual codebase — don't assume anything was implemented as originally planned.

## Goal

Each named agent gets a **"Me" root node** as the anchor point for its graph. All new nodes must be connected (directly or transitively) to this root. This makes the graph navigable from a known starting point, like a brain where everything connects back to "self."

## Current State (Inspect Before Starting)

Check these files:
- `mem0/memory/graph_memory.py` — `__init__()`, `add()`, `_add_entities()`
- `mem0/memory/main.py` — `_add_to_graph()`, how `filters` are constructed
- `mem0/graphs/configs.py` — `GraphStoreConfig`

Currently, nodes are scoped by `user_id`/`agent_id`/`run_id` in their properties but there is no concept of a root node.

## Changes Required

### 1. Auto-create "Me" Node on First Use

In `MemoryGraph.__init__()` or a new `_ensure_me_node(filters)` method:

```cypher
MERGE (me:__Entity__ {name: "me", user_id: $user_id})
ON CREATE SET me.created = timestamp(), me.entity_type = "self", me.is_anchor = true
```

The "Me" node should be created lazily on first `add()` call for a given agent/user scope, not on initialization (since we may not have filters at init time).

### 2. Connect New Nodes to the Graph

When `_add_entities()` creates a new node that has no existing connection to any other node in the graph, it should be connected to the "Me" node with a relationship derived from the context. 

Specifically:
- After adding entities, check if any newly created node is an "orphan" (not connected to any other node)
- If orphan found, create a relationship from "Me" → orphan using a sensible default like `KNOWS_ABOUT` or let the LLM decide

### 3. Update Entity Extraction Prompts

Update `EXTRACT_RELATIONS_PROMPT` in `mem0/graphs/utils.py` to instruct the LLM:
- When the text mentions "I", "me", "my" → the source entity should be "me" (already partially done — currently maps to `user_id`)
- Make it explicit that "me" is the anchor node

### 4. Add `get_me_node()` Method

A public method on `MemoryGraph` that returns the "Me" node and its direct connections:

```python
def get_me_node(self, filters, depth=1):
    """Returns the Me node and its connections up to `depth` hops."""
```

This is a precursor to Phase 3's multi-hop walking.

### 5. Configuration Option

Add to `GraphStoreConfig`:
```python
anchor_node_name: str = "me"  # Customizable root node name
```

## Testing

- Test that first `add()` creates a "Me" node
- Test that subsequent `add()` calls don't duplicate the "Me" node
- Test that orphan nodes get connected to "Me"
- Test `get_me_node()` returns correct structure
- Test that the anchor node name is configurable

## Files Modified

- `mem0/memory/graph_memory.py`
- `mem0/graphs/utils.py`
- `mem0/graphs/configs.py`
- `tests/memory/test_me_anchor_node.py` (new)
