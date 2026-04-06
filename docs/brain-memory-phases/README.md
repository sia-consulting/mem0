# Brain-like Graph Memory Structure — Implementation Phases

This document describes the phased implementation of a "Brain-like" memory structure for mem0, where the **Graph DB becomes the primary source of memory** instead of being supplementary to the vector store.

## Vision

Traditional memory uses Vector-DBs and Graph-DBs by creating key-value pairs, vectorizing those and then storing them in the DB. The new approach sees the Graph-DB as the main source of memory, where it creates nodes on new information, connects their relationships via edges and allows the Agent/LLM to actively "walk" on the graph to receive memory more precisely.

## Phases

| Phase | Title | Description |
|-------|-------|-------------|
| 1 | [Rich Properties](./phase-1-rich-properties.md) | Add support for arbitrary key-value properties on nodes and edges |
| 2 | [Me Anchor Node](./phase-2-me-anchor-node.md) | Create a "Me" root node per agent as the starting point of the graph |
| 3 | [Multi-Hop Graph Walking](./phase-3-graph-walking.md) | Enable multi-hop traversal so agents can walk the graph |
| 4 | [Direct Node/Edge API](./phase-4-direct-api.md) | Let agents directly create nodes, edges, and properties |
| 5 | [Document Nodes](./phase-5-document-nodes.md) | Bridge graph nodes to vectorized documents in the vector store |
| 6 | [Agent Tool Interface](./phase-6-agent-tools.md) | Expose graph walking as LLM-callable tools |
| 7 | [Storage Providers](./phase-7-storage-providers.md) | Support `storage://` URLs for OneDrive, Google Drive, etc. |

## Guiding Principles

1. **Each phase is self-contained** — it builds on whatever was actually implemented before, not on the original plan.
2. **Each phase starts with "look up what has been implemented"** — always inspect the actual codebase before making changes.
3. **Backward compatibility** — existing mem0 add/search/delete APIs continue to work.
4. **Neo4j first** — implement against Neo4j (MemoryGraph in graph_memory.py) first, then extend to other graph stores.
