"""Tests for Phase 3: Multi-Hop Graph Walking.

Tests for get_node(), get_neighbors(), walk(), find_path(), and get_edges()
methods on MemoryGraph.
"""

import json
from unittest.mock import Mock, MagicMock, patch, call

# Mock optional deps at module level so the import works without installing
# langchain_neo4j and rank_bm25. Matches the pattern in
# test_graph_memory_soft_delete.py.
_neo4j_mock = Mock()
patch.dict("sys.modules", {
    "langchain_neo4j": _neo4j_mock,
    "rank_bm25": Mock(),
}).start()

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(base_label=True, threshold=0.7):
    """Build a minimal MemoryConfig-like object for MemoryGraph."""
    config = MagicMock()
    config.graph_store.config.url = "bolt://localhost:7687"
    config.graph_store.config.username = "neo4j"
    config.graph_store.config.password = "password"
    config.graph_store.config.database = "neo4j"
    config.graph_store.config.base_label = base_label
    config.graph_store.threshold = threshold
    config.graph_store.custom_prompt = None
    config.graph_store.llm = None
    config.graph_store.anchor_node_name = "me"
    config.llm.provider = "openai"
    config.embedder.provider = "openai"
    config.embedder.config = {}
    config.vector_store.config = MagicMock()
    return config


FILTERS = {"user_id": "u1"}
FILTERS_AGENT = {"user_id": "u1", "agent_id": "a1"}
FILTERS_RUN = {"user_id": "u1", "agent_id": "a1", "run_id": "r1"}


@pytest.fixture
def mg():
    """Create a MemoryGraph with mocked external dependencies."""
    with patch("mem0.memory.graph_memory.Neo4jGraph") as MockNeo4j, \
         patch("mem0.memory.graph_memory.EmbedderFactory") as MockEmb, \
         patch("mem0.memory.graph_memory.LlmFactory") as MockLlm:

        mock_graph = MagicMock()
        MockNeo4j.return_value = mock_graph

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 1536
        MockEmb.create.return_value = mock_embedder

        mock_llm = MagicMock()
        MockLlm.create.return_value = mock_llm

        from mem0.memory.graph_memory import MemoryGraph
        graph = MemoryGraph(_make_config())
        graph.graph = mock_graph
        graph.llm = mock_llm
        graph.embedding_model = mock_embedder
        yield graph


# ---------------------------------------------------------------------------
# _build_scope_filter helper
# ---------------------------------------------------------------------------

class TestBuildScopeFilter:
    """Tests for the _build_scope_filter helper."""

    def test_user_id_only(self, mg):
        clause, params = mg._build_scope_filter("n", FILTERS)
        assert "n.user_id = $user_id" in clause
        assert params["user_id"] == "u1"
        assert "agent_id" not in clause
        assert "run_id" not in clause

    def test_with_agent_id(self, mg):
        clause, params = mg._build_scope_filter("n", FILTERS_AGENT)
        assert "n.agent_id = $agent_id" in clause
        assert params["agent_id"] == "a1"

    def test_with_run_id(self, mg):
        clause, params = mg._build_scope_filter("n", FILTERS_RUN)
        assert "n.run_id = $run_id" in clause
        assert params["run_id"] == "r1"

    def test_include_name(self, mg):
        clause, params = mg._build_scope_filter("n", FILTERS, include_name="alice")
        assert "n.name = $node_name" in clause
        assert params["node_name"] == "alice"


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

class TestGetNode:
    """Tests for get_node() method."""

    def test_returns_node_with_properties(self, mg):
        mg.graph.query.return_value = [{
            "node": MagicMock(),
            "props": {
                "name": "alice",
                "user_id": "u1",
                "entity_type": "person",
                "embedding": [0.1],
                "mentions": 3,
                "skill": "python",
                "location": "NYC",
            },
            "edge_count": 5,
        }]

        result = mg.get_node("alice", FILTERS)

        assert result is not None
        assert result["name"] == "alice"
        assert result["entity_type"] == "person"
        assert result["edge_count"] == 5
        assert result["is_anchor"] is False
        # System keys filtered
        assert "user_id" not in result["properties"]
        assert "embedding" not in result["properties"]
        assert "mentions" not in result["properties"]
        # Custom properties preserved
        assert result["properties"]["skill"] == "python"
        assert result["properties"]["location"] == "NYC"

    def test_returns_none_when_not_found(self, mg):
        mg.graph.query.return_value = []
        result = mg.get_node("nonexistent", FILTERS)
        assert result is None

    def test_anchor_node_flag(self, mg):
        mg.graph.query.return_value = [{
            "node": MagicMock(),
            "props": {"name": "me", "is_anchor": True, "entity_type": "self"},
            "edge_count": 10,
        }]

        result = mg.get_node("me", FILTERS)
        assert result["is_anchor"] is True
        assert result["entity_type"] == "self"

    def test_scoped_by_user_id(self, mg):
        mg.graph.query.return_value = []
        mg.get_node("alice", FILTERS)

        cypher = mg.graph.query.call_args[0][0]
        params = mg.graph.query.call_args[1]["params"]
        assert "user_id" in cypher
        assert params["user_id"] == "u1"
        assert params["node_name"] == "alice"

    def test_scoped_by_agent_id(self, mg):
        mg.graph.query.return_value = []
        mg.get_node("alice", FILTERS_AGENT)

        cypher = mg.graph.query.call_args[0][0]
        params = mg.graph.query.call_args[1]["params"]
        assert "agent_id" in cypher
        assert params["agent_id"] == "a1"


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    """Tests for get_neighbors() method."""

    def test_returns_neighbors(self, mg):
        mg.graph.query.return_value = [
            {
                "source": "alice",
                "relationship": "KNOWS",
                "destination": "bob",
                "edge_properties": {"since": "2024", "valid": True, "mentions": 1},
                "neighbor_properties": {"name": "bob", "user_id": "u1", "role": "engineer"},
            },
        ]

        result = mg.get_neighbors("alice", FILTERS)

        assert len(result) == 1
        assert result[0]["source"] == "alice"
        assert result[0]["relationship"] == "KNOWS"
        assert result[0]["destination"] == "bob"
        # System keys filtered
        assert "valid" not in result[0].get("edge_properties", {})
        assert "name" not in result[0].get("destination_properties", {})
        # Custom properties preserved
        assert result[0]["edge_properties"]["since"] == "2024"
        assert result[0]["destination_properties"]["role"] == "engineer"

    def test_outgoing_direction(self, mg):
        mg.graph.query.return_value = []
        mg.get_neighbors("alice", FILTERS, direction="outgoing")

        cypher = mg.graph.query.call_args[0][0]
        assert "]->(neighbor" in cypher

    def test_incoming_direction(self, mg):
        mg.graph.query.return_value = []
        mg.get_neighbors("alice", FILTERS, direction="incoming")

        cypher = mg.graph.query.call_args[0][0]
        assert "]-(neighbor" in cypher or "<-[r" in cypher

    def test_both_direction_default(self, mg):
        mg.graph.query.return_value = []
        mg.get_neighbors("alice", FILTERS)

        cypher = mg.graph.query.call_args[0][0]
        assert "startNode(r)" in cypher  # Direction-agnostic logic

    def test_relationship_type_filter(self, mg):
        mg.graph.query.return_value = []
        mg.get_neighbors("alice", FILTERS, relationship_type="WORKS_AT")

        cypher = mg.graph.query.call_args[0][0]
        assert ":WORKS_AT" in cypher

    def test_limit_param(self, mg):
        mg.graph.query.return_value = []
        mg.get_neighbors("alice", FILTERS, limit=10)

        params = mg.graph.query.call_args[1]["params"]
        assert params["limit"] == 10

    def test_empty_when_no_results(self, mg):
        mg.graph.query.return_value = []
        result = mg.get_neighbors("alice", FILTERS)
        assert result == []


# ---------------------------------------------------------------------------
# walk
# ---------------------------------------------------------------------------

class TestWalk:
    """Tests for walk() method."""

    def test_returns_walked_edges(self, mg):
        mg.graph.query.return_value = [
            {
                "source": "alice",
                "relationship": "KNOWS",
                "destination": "bob",
                "edge_properties": {"since": "2024", "valid": True},
                "destination_properties": {"name": "bob", "user_id": "u1"},
                "depth": 1,
            },
            {
                "source": "bob",
                "relationship": "WORKS_AT",
                "destination": "acme",
                "edge_properties": {"role": "engineer", "valid": True},
                "destination_properties": {"name": "acme", "user_id": "u1", "industry": "tech"},
                "depth": 2,
            },
        ]

        result = mg.walk("alice", FILTERS, depth=2)

        assert len(result) == 2
        assert result[0]["source"] == "alice"
        assert result[0]["depth"] == 1
        assert result[1]["source"] == "bob"
        assert result[1]["depth"] == 2
        # System keys filtered
        assert "valid" not in result[0].get("edge_properties", {})
        # Custom properties preserved
        assert result[1].get("destination_properties", {}).get("industry") == "tech"

    def test_depth_clamped_min(self, mg):
        mg.graph.query.return_value = []
        mg.walk("alice", FILTERS, depth=0)
        cypher = mg.graph.query.call_args[0][0]
        assert "*1..1" in cypher

    def test_depth_clamped_max(self, mg):
        mg.graph.query.return_value = []
        mg.walk("alice", FILTERS, depth=100)
        cypher = mg.graph.query.call_args[0][0]
        assert "*1..5" in cypher

    def test_depth_interpolated_not_parameterized(self, mg):
        """Depth must be interpolated (Neo4j doesn't support $depth in path bounds)."""
        mg.graph.query.return_value = []
        mg.walk("alice", FILTERS, depth=3)

        cypher = mg.graph.query.call_args[0][0]
        params = mg.graph.query.call_args[1]["params"]
        assert "*1..3" in cypher
        assert "depth" not in params  # NOT a Cypher parameter

    def test_relationship_type_filter(self, mg):
        mg.graph.query.return_value = []
        mg.walk("alice", FILTERS, relationship_type="KNOWS")
        cypher = mg.graph.query.call_args[0][0]
        assert ":KNOWS" in cypher

    def test_valid_filter_in_path(self, mg):
        """Soft-deleted edges should be excluded."""
        mg.graph.query.return_value = []
        mg.walk("alice", FILTERS)
        cypher = mg.graph.query.call_args[0][0]
        assert "rel.valid IS NULL OR rel.valid = true" in cypher

    def test_scoped_by_agent_id(self, mg):
        mg.graph.query.return_value = []
        mg.walk("alice", FILTERS_AGENT)

        params = mg.graph.query.call_args[1]["params"]
        assert params["agent_id"] == "a1"

    def test_limit_param(self, mg):
        mg.graph.query.return_value = []
        mg.walk("alice", FILTERS, limit=50)

        params = mg.graph.query.call_args[1]["params"]
        assert params["limit"] == 50


# ---------------------------------------------------------------------------
# find_path
# ---------------------------------------------------------------------------

class TestFindPath:
    """Tests for find_path() method."""

    def test_returns_path_hops(self, mg):
        # Mock Neo4j returning path relationships and nodes as dicts
        mg.graph.query.return_value = [{
            "rels": [
                {"since": "2024"},
                {"role": "CTO"},
            ],
            "path_nodes": [
                {"name": "alice"},
                {"name": "bob"},
                {"name": "acme"},
            ],
        }]

        result = mg.find_path("alice", "acme", FILTERS)

        assert result is not None
        assert len(result) == 2
        assert result[0]["source"] == "alice"
        assert result[0]["destination"] == "bob"
        assert result[1]["source"] == "bob"
        assert result[1]["destination"] == "acme"

    def test_returns_none_when_no_path(self, mg):
        mg.graph.query.return_value = []
        result = mg.find_path("alice", "unknown", FILTERS)
        assert result is None

    def test_uses_shortest_path(self, mg):
        mg.graph.query.return_value = []
        mg.find_path("alice", "bob", FILTERS)
        cypher = mg.graph.query.call_args[0][0]
        assert "shortestPath" in cypher

    def test_max_depth_interpolated(self, mg):
        mg.graph.query.return_value = []
        mg.find_path("alice", "bob", FILTERS, max_depth=3)

        cypher = mg.graph.query.call_args[0][0]
        assert "*1..3" in cypher

    def test_max_depth_clamped(self, mg):
        mg.graph.query.return_value = []
        mg.find_path("alice", "bob", FILTERS, max_depth=100)
        cypher = mg.graph.query.call_args[0][0]
        assert "*1..10" in cypher

    def test_valid_filter_in_path(self, mg):
        mg.graph.query.return_value = []
        mg.find_path("alice", "bob", FILTERS)
        cypher = mg.graph.query.call_args[0][0]
        assert "rel.valid IS NULL OR rel.valid = true" in cypher

    def test_both_nodes_scoped(self, mg):
        """Both from_node and to_node should be scoped by filters."""
        mg.graph.query.return_value = []
        mg.find_path("alice", "bob", FILTERS_AGENT)

        params = mg.graph.query.call_args[1]["params"]
        assert params["node_name"] == "alice"
        assert params["to_node_name"] == "bob"
        assert params["agent_id"] == "a1"


# ---------------------------------------------------------------------------
# get_edges
# ---------------------------------------------------------------------------

class TestGetEdges:
    """Tests for get_edges() method."""

    def test_returns_edges(self, mg):
        mg.graph.query.return_value = [
            {
                "source": "alice",
                "relationship": "KNOWS",
                "destination": "bob",
                "edge_properties": {"since": "2024", "valid": True, "mentions": 2},
                "valid": True,
            },
        ]

        result = mg.get_edges("alice", FILTERS)

        assert len(result) == 1
        assert result[0]["source"] == "alice"
        assert result[0]["relationship"] == "KNOWS"
        assert result[0]["destination"] == "bob"
        assert result[0]["valid"] is True
        # System keys filtered from edge_properties
        assert "valid" not in result[0].get("edge_properties", {})
        assert "mentions" not in result[0].get("edge_properties", {})
        assert result[0]["edge_properties"]["since"] == "2024"

    def test_outgoing_direction(self, mg):
        mg.graph.query.return_value = []
        mg.get_edges("alice", FILTERS, direction="outgoing")
        cypher = mg.graph.query.call_args[0][0]
        assert "]->(other" in cypher

    def test_incoming_direction(self, mg):
        mg.graph.query.return_value = []
        mg.get_edges("alice", FILTERS, direction="incoming")
        cypher = mg.graph.query.call_args[0][0]
        assert "<-[r" in cypher

    def test_relationship_type_filter(self, mg):
        mg.graph.query.return_value = []
        mg.get_edges("alice", FILTERS, relationship_type="KNOWS")
        cypher = mg.graph.query.call_args[0][0]
        assert ":KNOWS" in cypher

    def test_excludes_invalid_by_default(self, mg):
        mg.graph.query.return_value = []
        mg.get_edges("alice", FILTERS)
        cypher = mg.graph.query.call_args[0][0]
        assert "r.valid IS NULL OR r.valid = true" in cypher

    def test_include_invalid(self, mg):
        mg.graph.query.return_value = []
        mg.get_edges("alice", FILTERS, include_invalid=True)
        cypher = mg.graph.query.call_args[0][0]
        assert "r.valid IS NULL OR r.valid = true" not in cypher

    def test_valid_none_treated_as_true(self, mg):
        """Edges with valid=None (legacy) should be treated as valid."""
        mg.graph.query.return_value = [{
            "source": "alice",
            "relationship": "KNOWS",
            "destination": "bob",
            "edge_properties": {},
            "valid": None,
        }]

        result = mg.get_edges("alice", FILTERS)
        assert result[0]["valid"] is True

    def test_valid_false_treated_as_invalid(self, mg):
        """Edges explicitly marked invalid should report valid=False."""
        mg.graph.query.return_value = [{
            "source": "alice",
            "relationship": "KNOWS",
            "destination": "bob",
            "edge_properties": {},
            "valid": False,
        }]

        result = mg.get_edges("alice", FILTERS, include_invalid=True)
        assert result[0]["valid"] is False

    def test_limit_param(self, mg):
        mg.graph.query.return_value = []
        mg.get_edges("alice", FILTERS, limit=25)
        params = mg.graph.query.call_args[1]["params"]
        assert params["limit"] == 25


# ---------------------------------------------------------------------------
# _SYSTEM_RESERVED_KEYS filtering
# ---------------------------------------------------------------------------

class TestSystemKeyFiltering:
    """Verify that all Phase 3 methods properly filter system keys from properties."""

    def test_get_node_filters_system_keys(self, mg):
        from mem0.memory.graph_memory import _SYSTEM_RESERVED_KEYS

        all_system_props = {k: "val" for k in _SYSTEM_RESERVED_KEYS}
        all_system_props["custom_key"] = "custom_val"

        mg.graph.query.return_value = [{
            "node": MagicMock(),
            "props": all_system_props,
            "edge_count": 0,
        }]

        result = mg.get_node("test", FILTERS)
        for key in _SYSTEM_RESERVED_KEYS:
            assert key not in result["properties"]
        assert result["properties"]["custom_key"] == "custom_val"

    def test_get_neighbors_filters_system_keys(self, mg):
        from mem0.memory.graph_memory import _SYSTEM_RESERVED_KEYS

        mg.graph.query.return_value = [{
            "source": "a",
            "relationship": "R",
            "destination": "b",
            "edge_properties": {"valid": True, "custom_edge": "val"},
            "neighbor_properties": {"name": "b", "custom_dest": "val2"},
        }]

        result = mg.get_neighbors("a", FILTERS)
        edge_props = result[0].get("edge_properties", {})
        dest_props = result[0].get("destination_properties", {})

        assert "valid" not in edge_props
        assert edge_props.get("custom_edge") == "val"
        assert "name" not in dest_props
        assert dest_props.get("custom_dest") == "val2"

    def test_walk_filters_system_keys(self, mg):
        mg.graph.query.return_value = [{
            "source": "a",
            "relationship": "R",
            "destination": "b",
            "edge_properties": {"mentions": 3, "custom_edge": "val"},
            "destination_properties": {"embedding": [0.1], "custom_dest": "val2"},
            "depth": 1,
        }]

        result = mg.walk("a", FILTERS)
        edge_props = result[0].get("edge_properties", {})
        dest_props = result[0].get("destination_properties", {})

        assert "mentions" not in edge_props
        assert edge_props.get("custom_edge") == "val"
        assert "embedding" not in dest_props
        assert dest_props.get("custom_dest") == "val2"

    def test_get_edges_filters_system_keys(self, mg):
        mg.graph.query.return_value = [{
            "source": "a",
            "relationship": "R",
            "destination": "b",
            "edge_properties": {"created_at": "123", "valid": True, "custom": "val"},
            "valid": True,
        }]

        result = mg.get_edges("a", FILTERS)
        edge_props = result[0].get("edge_properties", {})

        assert "created_at" not in edge_props
        assert "valid" not in edge_props
        assert edge_props.get("custom") == "val"
