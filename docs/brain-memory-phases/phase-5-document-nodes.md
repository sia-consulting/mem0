# Phase 5: Document Nodes with Vector Store Bridge

> **Before starting:** Look up what has been implemented in the previous phases. Inspect the actual codebase — don't assume anything was implemented as originally planned.

## Goal

Handle **large documents** by creating a graph node that links to vectorized content in the vector store. When information is too large or detailed for graph properties, a `Document` node is created in the graph with a reference to chunks stored in the vector store. The agent can then load detailed content from the vector store when it reaches a Document node.

## Current State (Inspect Before Starting)

Check these files:
- `mem0/memory/graph_memory.py` — current add/search flow
- `mem0/memory/main.py` — how vector store and graph store interact
- `mem0/vector_stores/` — vector store interface
- Any direct API from Phase 4
- Any properties support from Phase 1

Currently, graph and vector stores run independently in parallel. There's no cross-reference between them.

## Changes Required

### 1. Define Document Node Type

A Document node in the graph:
```python
{
    "name": "quarterly_report_q1_2024",
    "entity_type": "document",
    "properties": {
        "title": "Q1 2024 Quarterly Report",
        "source_url": "storage://drive1/doc123",  # or a regular URL
        "vector_collection": "mem0",  # which vector store collection
        "vector_tag": "doc:quarterly_report_q1_2024",  # tag to find chunks
        "chunk_count": 15,
        "content_summary": "Financial report covering Q1 2024...",
        "created_at": "2024-04-01T00:00:00Z"
    }
}
```

### 2. Add `add_document()` Method to MemoryGraph

```python
def add_document(self, content, filters, title=None, source_url=None, 
                 source_node=None, relationship=None, properties=None,
                 vector_store=None, embedding_model=None):
    """Add a document to the graph with vectorized content.
    
    Args:
        content: The document text content
        filters: Scope filters
        title: Human-readable document title
        source_url: Where the document came from
        source_node: Node to connect the document to
        relationship: Relationship from source_node to this document
        properties: Additional properties for the document node
        vector_store: Reference to the vector store instance (for chunking/storing)
        embedding_model: Reference to the embedding model (for vectorizing chunks)
    
    Flow:
        1. Generate a document tag/ID
        2. Chunk the content
        3. Vectorize and store chunks in vector store with the tag
        4. Summarize the content using LLM
        5. Create a Document node in the graph with the vector reference
        6. Connect to source_node
    """
```

### 3. Add `load_document()` Method

```python
def load_document(self, node_name, filters, query=None, limit=5,
                  vector_store=None, embedding_model=None):
    """Load content from a Document node's linked vector store.
    
    Args:
        node_name: Name of the Document node
        filters: Scope filters
        query: Optional semantic search query within the document
        limit: Max chunks to return
        vector_store: Reference to vector store
        embedding_model: Reference to embedding model
    
    Returns:
        List of relevant text chunks from the document
    """
```

If `query` is provided, performs semantic search within the document's chunks. Otherwise, returns the first N chunks.

### 4. Simple Chunking Utility

Add a basic text chunking function (or use an existing one):

```python
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
```

### 5. Expose Through Memory Class

```python
def graph_add_document(self, content, *, user_id=None, agent_id=None, run_id=None,
                       title=None, source_url=None, source_node=None, 
                       relationship=None, properties=None):
    """Add a document to the graph with vectorized content."""

def graph_load_document(self, node_name, *, user_id=None, agent_id=None, run_id=None,
                        query=None, limit=5):
    """Load content from a Document node."""
```

### 6. Metadata Tagging in Vector Store

When storing document chunks, use metadata to tag them:
```python
{
    "data": "chunk text...",
    "document_tag": "doc:quarterly_report_q1_2024",
    "document_title": "Q1 2024 Quarterly Report",
    "chunk_index": 0,
    "user_id": "...",
    "is_document_chunk": True
}
```

This allows filtering document chunks in the vector store.

## Testing

- Test `add_document()` creates graph node with correct properties
- Test `add_document()` chunks and stores content in vector store
- Test `load_document()` retrieves relevant chunks
- Test `load_document()` with semantic query
- Test that document nodes appear in `search()` and `walk()` results
- Test that document chunks are scoped to user_id

## Files Modified

- `mem0/memory/graph_memory.py`
- `mem0/memory/main.py`
- `mem0/memory/utils.py` (chunking utility)
- `tests/memory/test_document_nodes.py` (new)
