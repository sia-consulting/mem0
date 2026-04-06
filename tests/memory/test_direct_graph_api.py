"""Tests for Phase 4: Direct Node/Edge Creation API.

Tests for add_node(), add_edge(), update_node_properties(), and
update_edge_properties() methods on MemoryGraph.
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
# add_node — basic creation
# ---------------------------------------------------------------------------

class TestAddNode:
    """Tests for add_node()."""

    def test_add_node_with_source(self, mg):
        """add_node() creates a node connected to a specified source node."""
        # Mock get_node to confirm source exists
        mg.get_node = MagicMock(return_value={
            "name": "alice", "entity_type": "person", "is_anchor": False,
            "properties": {}, "edge_count": 2,
        })

        mg.graph.query.return_value = [{
            "name": "python",
            "labels": ["__Entity__"],
            "connected_from": "alice",
            "via_relationship": "KNOWS",
        }]

        result = mg.add_node(
            "Python", FILTERS,
            entity_type="skill",
            properties={"level": "expert"},
            source_node="Alice",
            relationship="KNOWS",
        )

        assert result["name"] == "python"
        assert result["entity_type"] == "skill"
        assert result["connected_from"] == "alice"
        assert result["via_relationship"] == "KNOWS"
        assert result["properties"] == {"level": "expert"}

        # The Cypher should contain MERGE and relationship
        cypher = mg.graph.query.call_args_list[-1][0][0]
        assert "MERGE" in cypher
        assert ":knows" in cypher

    def test_add_node_defaults_to_me_anchor(self, mg):
        """add_node() without source_node connects to the Me anchor."""
        mg.graph.query.return_value = [{
            "name": "python",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        result = mg.add_node("Python", FILTERS, entity_type="skill")

        # _ensure_me_node should be called
        calls = mg.graph.query.call_args_list
        # First call is _ensure_me_node, second is the actual add_node
        me_cypher = calls[-2][0][0]
        assert "is_anchor" in me_cypher

        # The add Cypher should use KNOWS_ABOUT
        add_cypher = calls[-1][0][0]
        assert "KNOWS_ABOUT" in add_cypher

    def test_add_node_lowercases_name(self, mg):
        """Node names are lowercased and spaces replaced with underscores."""
        mg.graph.query.return_value = [{
            "name": "new_york",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("New York", FILTERS)

        params = mg.graph.query.call_args_list[-1][1].get("params") or mg.graph.query.call_args_list[-1][0][1]
        assert params["new_node_name"] == "new_york"

    def test_add_node_with_properties(self, mg):
        """Properties are sanitized and passed to Cypher."""
        mg.graph.query.return_value = [{
            "name": "company_x",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        result = mg.add_node(
            "Company X", FILTERS,
            properties={"founded": "2020", "industry": "tech"},
        )

        params = mg.graph.query.call_args_list[-1][1].get("params") or mg.graph.query.call_args_list[-1][0][1]
        assert params["node_props"]["founded"] == "2020"
        assert params["node_props"]["industry"] == "tech"

    def test_add_node_creates_embedding(self, mg):
        """add_node() embeds the node name."""
        mg.graph.query.return_value = [{
            "name": "test_node",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("test_node", FILTERS)

        mg.embedding_model.embed.assert_called()

    def test_add_node_with_agent_id(self, mg):
        """add_node() includes agent_id in scope filters."""
        mg.graph.query.return_value = [{
            "name": "item",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("item", FILTERS_AGENT)

        params = mg.graph.query.call_args_list[-1][1].get("params") or mg.graph.query.call_args_list[-1][0][1]
        assert params["agent_id"] == "a1"

    def test_add_node_with_run_id(self, mg):
        """add_node() includes run_id in scope filters."""
        mg.graph.query.return_value = [{
            "name": "item",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("item", FILTERS_RUN)

        params = mg.graph.query.call_args_list[-1][1].get("params") or mg.graph.query.call_args_list[-1][0][1]
        assert params["agent_id"] == "a1"
        assert params["run_id"] == "r1"


# ---------------------------------------------------------------------------
# add_node — validation errors
# ---------------------------------------------------------------------------

class TestAddNodeValidation:
    """Tests for add_node() input validation."""

    def test_empty_name_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty"):
            mg.add_node("", FILTERS)

    def test_none_name_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty"):
            mg.add_node(None, FILTERS)

    def test_source_without_relationship_raises(self, mg):
        with pytest.raises(ValueError, match="relationship is required"):
            mg.add_node("test", FILTERS, source_node="alice")

    def test_nonexistent_source_raises(self, mg):
        """add_node() with a source that doesn't exist raises ValueError."""
        mg.get_node = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="does not exist"):
            mg.add_node("test", FILTERS, source_node="nonexistent",
                        relationship="KNOWS")


# ---------------------------------------------------------------------------
# add_edge — basic creation
# ---------------------------------------------------------------------------

class TestAddEdge:
    """Tests for add_edge()."""

    def test_add_edge_between_existing_nodes(self, mg):
        """add_edge() creates an edge between two existing nodes."""
        mg.get_node = MagicMock(side_effect=[
            {"name": "alice", "entity_type": "person", "is_anchor": False, "properties": {}, "edge_count": 1},
            {"name": "bob", "entity_type": "person", "is_anchor": False, "properties": {}, "edge_count": 1},
        ])

        mg.graph.query.return_value = [{
            "source": "alice",
            "relationship": "WORKS_WITH",
            "destination": "bob",
        }]

        result = mg.add_edge("Alice", "Bob", "WORKS_WITH", FILTERS)

        assert result["source"] == "alice"
        assert result["relationship"] == "WORKS_WITH"
        assert result["destination"] == "bob"

        cypher = mg.graph.query.call_args[0][0]
        assert "MERGE" in cypher
        assert ":works_with" in cypher

    def test_add_edge_with_properties(self, mg):
        """add_edge() passes sanitized edge properties."""
        mg.get_node = MagicMock(side_effect=[
            {"name": "alice", "entity_type": "person", "is_anchor": False, "properties": {}, "edge_count": 1},
            {"name": "company", "entity_type": "org", "is_anchor": False, "properties": {}, "edge_count": 1},
        ])

        mg.graph.query.return_value = [{
            "source": "alice",
            "relationship": "WORKS_AT",
            "destination": "company",
        }]

        result = mg.add_edge("alice", "company", "WORKS_AT", FILTERS,
                             properties={"since": "2020", "role": "engineer"})

        params = mg.graph.query.call_args[1].get("params") or mg.graph.query.call_args[0][1]
        assert params["edge_props"]["since"] == "2020"
        assert params["edge_props"]["role"] == "engineer"
        assert "edge_properties" in result

    def test_add_edge_lowercases_names(self, mg):
        """Node names in add_edge() are lowercased."""
        mg.get_node = MagicMock(side_effect=[
            {"name": "alice", "entity_type": "person", "is_anchor": False, "properties": {}, "edge_count": 1},
            {"name": "bob", "entity_type": "person", "is_anchor": False, "properties": {}, "edge_count": 1},
        ])

        mg.graph.query.return_value = [{
            "source": "alice",
            "relationship": "KNOWS",
            "destination": "bob",
        }]

        mg.add_edge("Alice", "Bob", "KNOWS", FILTERS)

        # get_node should have been called with lowercased names
        mg.get_node.assert_any_call("alice", FILTERS)
        mg.get_node.assert_any_call("bob", FILTERS)

    def test_add_edge_with_agent_id(self, mg):
        """add_edge() includes agent_id in scope."""
        mg.get_node = MagicMock(side_effect=[
            {"name": "a", "entity_type": None, "is_anchor": False, "properties": {}, "edge_count": 0},
            {"name": "b", "entity_type": None, "is_anchor": False, "properties": {}, "edge_count": 0},
        ])

        mg.graph.query.return_value = [{"source": "a", "relationship": "R", "destination": "b"}]

        mg.add_edge("a", "b", "R", FILTERS_AGENT)

        params = mg.graph.query.call_args[1].get("params") or mg.graph.query.call_args[0][1]
        assert params["agent_id"] == "a1"


# ---------------------------------------------------------------------------
# add_edge — validation errors
# ---------------------------------------------------------------------------

class TestAddEdgeValidation:
    """Tests for add_edge() input validation."""

    def test_empty_source_raises(self, mg):
        with pytest.raises(ValueError, match="source"):
            mg.add_edge("", "bob", "KNOWS", FILTERS)

    def test_empty_destination_raises(self, mg):
        with pytest.raises(ValueError, match="destination"):
            mg.add_edge("alice", "", "KNOWS", FILTERS)

    def test_empty_relationship_raises(self, mg):
        with pytest.raises(ValueError, match="relationship"):
            mg.add_edge("alice", "bob", "", FILTERS)

    def test_source_not_found_raises(self, mg):
        """add_edge() raises ValueError if source node doesn't exist."""
        mg.get_node = MagicMock(side_effect=[None])

        with pytest.raises(ValueError, match="does not exist"):
            mg.add_edge("nonexistent", "bob", "KNOWS", FILTERS)

    def test_destination_not_found_raises(self, mg):
        """add_edge() raises ValueError if destination node doesn't exist."""
        mg.get_node = MagicMock(side_effect=[
            {"name": "alice", "entity_type": "person", "is_anchor": False, "properties": {}, "edge_count": 1},
            None,
        ])

        with pytest.raises(ValueError, match="does not exist"):
            mg.add_edge("alice", "nonexistent", "KNOWS", FILTERS)


# ---------------------------------------------------------------------------
# update_node_properties
# ---------------------------------------------------------------------------

class TestUpdateNodeProperties:
    """Tests for update_node_properties()."""

    def test_update_merges_properties(self, mg):
        """update_node_properties() merges new properties with existing."""
        mg.graph.query.return_value = [{
            "name": "alice",
            "props": {
                "name": "alice", "user_id": "u1", "embedding": [0.1],
                "age": "30", "city": "new_york",
            },
        }]

        result = mg.update_node_properties("Alice", FILTERS, {"age": "31", "city": "SF"})

        assert result is not None
        assert result["name"] == "alice"
        # System keys should be filtered out
        assert "user_id" not in result["properties"]
        assert "embedding" not in result["properties"]
        # User properties should be present
        assert result["properties"]["age"] == "30"  # from the returned props
        assert result["properties"]["city"] == "new_york"

        # Verify Cypher uses SET n += $update_props
        cypher = mg.graph.query.call_args[0][0]
        assert "+= $update_props" in cypher

    def test_update_returns_none_if_not_found(self, mg):
        """update_node_properties() returns None if the node doesn't exist."""
        mg.graph.query.return_value = []

        result = mg.update_node_properties("nonexistent", FILTERS, {"key": "val"})
        assert result is None

    def test_update_filters_reserved_keys(self, mg):
        """Reserved keys in properties are filtered out during sanitization."""
        mg.graph.query.return_value = [{"name": "alice", "props": {"name": "alice", "color": "blue"}}]

        mg.update_node_properties("alice", FILTERS, {"name": "override", "color": "red"})

        params = mg.graph.query.call_args[1].get("params") or mg.graph.query.call_args[0][1]
        # "name" should be filtered out by _sanitize_properties
        assert "name" not in params["update_props"]
        assert params["update_props"]["color"] == "red"

    def test_update_lowercases_node_name(self, mg):
        """Node name is lowercased and spaces replaced."""
        mg.graph.query.return_value = [{"name": "new_york", "props": {"name": "new_york"}}]

        mg.update_node_properties("New York", FILTERS, {"population": "8M"})

        params = mg.graph.query.call_args[1].get("params") or mg.graph.query.call_args[0][1]
        assert params["node_name"] == "new_york"


# ---------------------------------------------------------------------------
# update_node_properties — validation errors
# ---------------------------------------------------------------------------

class TestUpdateNodeValidation:
    """Tests for update_node_properties() input validation."""

    def test_empty_properties_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty dict"):
            mg.update_node_properties("alice", FILTERS, {})

    def test_none_properties_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty dict"):
            mg.update_node_properties("alice", FILTERS, None)

    def test_empty_node_name_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty string"):
            mg.update_node_properties("", FILTERS, {"key": "val"})

    def test_only_reserved_keys_raises(self, mg):
        """Properties containing only reserved keys raises ValueError."""
        with pytest.raises(ValueError, match="No valid properties"):
            mg.update_node_properties("alice", FILTERS, {"name": "x", "user_id": "u1"})


# ---------------------------------------------------------------------------
# update_edge_properties
# ---------------------------------------------------------------------------

class TestUpdateEdgeProperties:
    """Tests for update_edge_properties()."""

    def test_update_edge_merges_properties(self, mg):
        """update_edge_properties() merges new properties with existing."""
        mg.graph.query.return_value = [{
            "source": "alice",
            "relationship": "WORKS_AT",
            "destination": "acme",
            "edge_properties": {
                "since": "2020", "role": "engineer",
                "valid": True, "created_at": 123,
            },
        }]

        result = mg.update_edge_properties(
            "Alice", "Acme", "WORKS_AT", FILTERS,
            {"role": "senior_engineer"},
        )

        assert result is not None
        assert result["source"] == "alice"
        assert result["relationship"] == "WORKS_AT"
        assert result["destination"] == "acme"
        # System keys filtered from edge_properties
        assert "valid" not in result["edge_properties"]
        assert "created_at" not in result["edge_properties"]
        # User properties preserved
        assert result["edge_properties"]["since"] == "2020"
        assert result["edge_properties"]["role"] == "engineer"

    def test_update_edge_returns_none_if_not_found(self, mg):
        """update_edge_properties() returns None if the edge doesn't exist."""
        mg.graph.query.return_value = []

        result = mg.update_edge_properties(
            "alice", "nonexistent", "KNOWS", FILTERS,
            {"key": "val"},
        )
        assert result is None

    def test_update_edge_sanitizes_relationship(self, mg):
        """Relationship type is sanitized for Cypher."""
        mg.graph.query.return_value = [{
            "source": "a", "relationship": "WORKS_AT", "destination": "b",
            "edge_properties": {},
        }]

        mg.update_edge_properties("a", "b", "works-at", FILTERS, {"key": "val"})

        cypher = mg.graph.query.call_args[0][0]
        assert ":works_at" in cypher  # dash replaced with underscore

    def test_update_edge_only_valid_edges(self, mg):
        """update_edge_properties() only updates valid (non-soft-deleted) edges."""
        mg.graph.query.return_value = []

        mg.update_edge_properties("a", "b", "R", FILTERS, {"key": "val"})

        cypher = mg.graph.query.call_args[0][0]
        assert "r.valid IS NULL OR r.valid = true" in cypher


# ---------------------------------------------------------------------------
# update_edge_properties — validation errors
# ---------------------------------------------------------------------------

class TestUpdateEdgeValidation:
    """Tests for update_edge_properties() input validation."""

    def test_empty_properties_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty dict"):
            mg.update_edge_properties("a", "b", "R", FILTERS, {})

    def test_none_properties_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty dict"):
            mg.update_edge_properties("a", "b", "R", FILTERS, None)

    def test_empty_source_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty"):
            mg.update_edge_properties("", "b", "R", FILTERS, {"k": "v"})

    def test_empty_destination_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty"):
            mg.update_edge_properties("a", "", "R", FILTERS, {"k": "v"})

    def test_empty_relationship_raises(self, mg):
        with pytest.raises(ValueError, match="non-empty"):
            mg.update_edge_properties("a", "b", "", FILTERS, {"k": "v"})


# ---------------------------------------------------------------------------
# Properties sanitization integration
# ---------------------------------------------------------------------------

class TestPropertySanitization:
    """Verify _sanitize_properties behavior used across Phase 4."""

    def test_system_keys_excluded_from_node_props(self, mg):
        """Reserved keys are filtered from add_node properties."""
        mg.graph.query.return_value = [{
            "name": "test",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("test", FILTERS, properties={
            "embedding": "should_be_filtered",
            "user_id": "u1",
            "custom_key": "allowed",
        })

        params = mg.graph.query.call_args_list[-1][1].get("params") or mg.graph.query.call_args_list[-1][0][1]
        assert "embedding" not in params["node_props"]
        assert "user_id" not in params["node_props"]
        assert params["node_props"]["custom_key"] == "allowed"

    def test_properties_values_coerced_to_strings(self, mg):
        """Property values are coerced to strings for Neo4j compatibility."""
        mg.graph.query.return_value = [{
            "name": "test",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("test", FILTERS, properties={
            "count": 42,
            "active": True,
        })

        params = mg.graph.query.call_args_list[-1][1].get("params") or mg.graph.query.call_args_list[-1][0][1]
        assert params["node_props"]["count"] == "42"
        assert params["node_props"]["active"] == "True"

    def test_invalid_identifier_keys_filtered(self, mg):
        """Keys that aren't valid Python identifiers are filtered out."""
        mg.graph.query.return_value = [{
            "name": "test",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("test", FILTERS, properties={
            "valid_key": "yes",
            "123invalid": "filtered",
            "": "filtered",
        })

        params = mg.graph.query.call_args_list[-1][1].get("params") or mg.graph.query.call_args_list[-1][0][1]
        assert "valid_key" in params["node_props"]
        assert "123invalid" not in params["node_props"]


# ---------------------------------------------------------------------------
# Integration: nodes created via direct API are visible to search/walk
# ---------------------------------------------------------------------------

class TestDirectAPIDiscoverability:
    """Verify direct-API nodes use the same scope and labels as LLM-extracted ones."""

    def test_add_node_uses_scope_filters(self, mg):
        """Nodes created via add_node() carry the same scope props as LLM nodes."""
        mg.graph.query.return_value = [{
            "name": "test_node",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("test_node", FILTERS_RUN)

        params = mg.graph.query.call_args_list[-1][1].get("params") or mg.graph.query.call_args_list[-1][0][1]
        assert params["user_id"] == "u1"
        assert params["agent_id"] == "a1"
        assert params["run_id"] == "r1"

    def test_add_node_uses_base_label(self, mg):
        """When base_label is configured, add_node() uses the :`__Entity__` label."""
        mg.graph.query.return_value = [{
            "name": "item",
            "labels": ["__Entity__"],
            "connected_from": "me",
            "via_relationship": "KNOWS_ABOUT",
        }]

        mg.add_node("item", FILTERS)

        cypher = mg.graph.query.call_args_list[-1][0][0]
        assert "__Entity__" in cypher

    def test_add_edge_creates_valid_relationship(self, mg):
        """Edges created via add_edge() are marked as valid."""
        mg.get_node = MagicMock(side_effect=[
            {"name": "a", "entity_type": None, "is_anchor": False, "properties": {}, "edge_count": 0},
            {"name": "b", "entity_type": None, "is_anchor": False, "properties": {}, "edge_count": 0},
        ])

        mg.graph.query.return_value = [{"source": "a", "relationship": "R", "destination": "b"}]

        mg.add_edge("a", "b", "R", FILTERS)

        cypher = mg.graph.query.call_args[0][0]
        assert "r.valid = true" in cypher
