# Phase 6: Agent Tool Interface

> **Before starting:** Look up what has been implemented in the previous phases. Inspect the actual codebase — don't assume anything was implemented as originally planned.

## Goal

Expose graph walking and manipulation methods as **LLM-callable tool definitions** that can be registered with any agent framework (LangChain, AutoGen, OpenAI function calling, etc.). This enables agents to actively navigate and modify the graph during conversations.

## Current State (Inspect Before Starting)

Check these files:
- `mem0/graphs/tools.py` — existing LLM tool definitions
- `mem0/memory/graph_memory.py` — walking methods from Phase 3, direct API from Phase 4
- `mem0/memory/main.py` — public graph methods
- Any document methods from Phase 5

## Changes Required

### 1. Define Navigation Tool Schemas

Add to `mem0/graphs/tools.py`:

**SEARCH_NODES_TOOL** — Find nodes matching a query:
```json
{
    "name": "brain_search_nodes",
    "description": "Search for nodes in memory by name or semantic similarity. Use this to find a starting point for exploration.",
    "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "description": "Max results", "default": 10}
    }
}
```

**GET_NEIGHBORS_TOOL** — Get connected nodes:
```json
{
    "name": "brain_get_neighbors",
    "description": "Get all nodes directly connected to a given node. Use this to explore what's around a known node.",
    "parameters": {
        "node_name": {"type": "string"},
        "direction": {"type": "string", "enum": ["outgoing", "incoming", "both"]},
        "relationship_type": {"type": "string", "description": "Optional: filter by relationship type"}
    }
}
```

**WALK_GRAPH_TOOL** — Multi-hop exploration:
```json
{
    "name": "brain_walk",
    "description": "Walk the memory graph from a starting node, exploring connections up to N hops deep.",
    "parameters": {
        "start_node": {"type": "string"},
        "depth": {"type": "integer", "default": 2, "maximum": 5}
    }
}
```

**GET_NODE_TOOL** — Inspect a specific node:
```json
{
    "name": "brain_get_node",
    "description": "Get detailed information about a specific node including all its properties.",
    "parameters": {
        "node_name": {"type": "string"}
    }
}
```

### 2. Define Manipulation Tool Schemas

**ADD_NODE_TOOL**:
```json
{
    "name": "brain_add_node",
    "description": "Add a new node to memory, connected to an existing node.",
    "parameters": {
        "name": {"type": "string"},
        "entity_type": {"type": "string"},
        "properties": {"type": "object"},
        "source_node": {"type": "string"},
        "relationship": {"type": "string"}
    }
}
```

**ADD_EDGE_TOOL**:
```json
{
    "name": "brain_add_edge",
    "description": "Add a relationship between two existing nodes in memory.",
    "parameters": {
        "source": {"type": "string"},
        "destination": {"type": "string"},
        "relationship": {"type": "string"},
        "properties": {"type": "object"}
    }
}
```

**REMEMBER_TOOL** — High-level "store this information":
```json
{
    "name": "brain_remember",
    "description": "Store new information in memory. The system will figure out the best nodes, edges, and properties.",
    "parameters": {
        "information": {"type": "string", "description": "The information to remember"}
    }
}
```

### 3. Tool Registry Class

Create `mem0/graphs/brain_tools.py`:

```python
class BrainToolRegistry:
    """Registry of LLM-callable tools for brain-like graph memory."""
    
    def __init__(self, memory_instance):
        self.memory = memory_instance
    
    def get_tools(self, format="openai"):
        """Return tool definitions in the specified format.
        
        Args:
            format: "openai", "langchain", "anthropic"
        """
    
    def execute_tool(self, tool_name, arguments, filters):
        """Execute a tool call and return the result."""
    
    def get_openai_tools(self):
        """Return tools in OpenAI function calling format."""
    
    def get_tool_descriptions(self):
        """Return human-readable tool descriptions for system prompts."""
```

### 4. Expose Through Memory Class

```python
def get_brain_tools(self, format="openai"):
    """Get LLM tool definitions for brain-like graph navigation."""

def execute_brain_tool(self, tool_name, arguments, *, user_id=None, agent_id=None, run_id=None):
    """Execute a brain tool call."""
```

## Testing

- Test tool schemas are valid JSON Schema
- Test `BrainToolRegistry.get_tools()` returns correct format
- Test `execute_tool()` correctly dispatches to graph methods
- Test each tool works end-to-end with mocked graph
- Test tool result formats are LLM-friendly (concise, structured)

## Files Modified

- `mem0/graphs/tools.py` (add new tool definitions)
- `mem0/graphs/brain_tools.py` (new — tool registry)
- `mem0/memory/main.py`
- `tests/memory/test_brain_tools.py` (new)
