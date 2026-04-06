"""Tests for Phase 1: Rich Properties on Nodes and Edges.

Tests that entity extraction, relation extraction, storage, search, and
get_all correctly handle properties on nodes and edges.
"""

import json
from unittest.mock import Mock, MagicMock, patch

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
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(base_label=True, threshold=0.7, custom_prompt=None):
    """Build a minimal MemoryConfig-like object for MemoryGraph."""
    config = MagicMock()
    config.graph_store.config.url = "bolt://localhost:7687"
    config.graph_store.config.username = "neo4j"
    config.graph_store.config.password = "password"
    config.graph_store.config.database = "neo4j"
    config.graph_store.config.base_label = base_label
    config.graph_store.threshold = threshold
    config.graph_store.custom_prompt = custom_prompt
    config.graph_store.llm = None
    config.llm.provider = "openai"
    config.embedder.provider = "openai"
    config.embedder.config = {}
    config.vector_store.config = MagicMock()
    return config


@pytest.fixture
def memory_graph():
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
        mg = MemoryGraph(_make_config())
        mg.graph = mock_graph
        mg.llm = mock_llm
        mg.embedding_model = mock_embedder
        yield mg


# ---------------------------------------------------------------------------
# Helper parsers
# ---------------------------------------------------------------------------

class TestParseEntityProperties:
    """Test _parse_entity_properties helper."""

    def test_dict_properties(self):
        from mem0.memory.graph_memory import _parse_entity_properties
        item = {"entity": "alice", "entity_type": "person", "properties": {"email": "a@b.com", "role": "manager"}}
        assert _parse_entity_properties(item) == {"email": "a@b.com", "role": "manager"}

    def test_properties_json_string(self):
        from mem0.memory.graph_memory import _parse_entity_properties
        item = {"entity": "alice", "entity_type": "person", "properties_json": '{"email": "a@b.com"}'}
        assert _parse_entity_properties(item) == {"email": "a@b.com"}

    def test_empty_properties_json(self):
        from mem0.memory.graph_memory import _parse_entity_properties
        item = {"entity": "alice", "entity_type": "person", "properties_json": ""}
        assert _parse_entity_properties(item) == {}

    def test_invalid_json(self):
        from mem0.memory.graph_memory import _parse_entity_properties
        item = {"entity": "alice", "entity_type": "person", "properties_json": "not json"}
        assert _parse_entity_properties(item) == {}

    def test_no_properties(self):
        from mem0.memory.graph_memory import _parse_entity_properties
        item = {"entity": "alice", "entity_type": "person"}
        assert _parse_entity_properties(item) == {}

    def test_values_coerced_to_string(self):
        from mem0.memory.graph_memory import _parse_entity_properties
        item = {"entity": "x", "entity_type": "t", "properties": {"count": 42, "active": True}}
        result = _parse_entity_properties(item)
        assert result == {"count": "42", "active": "True"}


class TestParseRelationProperties:
    """Test _parse_relation_properties helper."""

    def test_dict_properties(self):
        from mem0.memory.graph_memory import _parse_relation_properties
        item = {"source": "a", "relationship": "r", "destination": "b", "properties": {"since": "2024"}}
        assert _parse_relation_properties(item) == {"since": "2024"}

    def test_properties_json_string(self):
        from mem0.memory.graph_memory import _parse_relation_properties
        item = {"source": "a", "relationship": "r", "destination": "b", "properties_json": '{"method": "email"}'}
        assert _parse_relation_properties(item) == {"method": "email"}


class TestSanitizeProperties:
    """Test _sanitize_properties helper."""

    def test_filters_reserved_keys(self):
        from mem0.memory.graph_memory import _sanitize_properties
        props = {"name": "should_be_filtered", "email": "a@b.com", "user_id": "x"}
        result = _sanitize_properties(props)
        assert "name" not in result
        assert "user_id" not in result
        assert result == {"email": "a@b.com"}

    def test_normalizes_key_names(self):
        from mem0.memory.graph_memory import _sanitize_properties
        props = {"First Name": "Alice", "last-name": "Smith"}
        result = _sanitize_properties(props)
        assert result == {"first_name": "Alice", "last_name": "Smith"}

    def test_empty_input(self):
        from mem0.memory.graph_memory import _sanitize_properties
        assert _sanitize_properties({}) == {}
        assert _sanitize_properties(None) == {}

    def test_none_values_become_empty_string(self):
        from mem0.memory.graph_memory import _sanitize_properties
        result = _sanitize_properties({"title": None})
        assert result == {"title": ""}


# ---------------------------------------------------------------------------
# Entity extraction with properties
# ---------------------------------------------------------------------------

class TestRetrieveNodesWithProperties:
    """Test that _retrieve_nodes_from_data returns properties."""

    def test_entities_with_properties(self, memory_graph):
        memory_graph.llm.generate_response.return_value = {
            "tool_calls": [{
                "name": "extract_entities",
                "arguments": {
                    "entities": [
                        {"entity": "Sia Ghassemi", "entity_type": "person", "properties": {"email": "sia@example.com"}},
                        {"entity": "Project Alpha", "entity_type": "project"},
                    ]
                }
            }]
        }
        result = memory_graph._retrieve_nodes_from_data("test", {"user_id": "u1"})
        assert "sia_ghassemi" in result
        assert result["sia_ghassemi"]["type"] == "person"
        assert result["sia_ghassemi"]["properties"] == {"email": "sia@example.com"}
        assert "project_alpha" in result
        assert result["project_alpha"]["properties"] == {}

    def test_entities_with_struct_properties_json(self, memory_graph):
        """Struct tool returns properties_json as a JSON string."""
        memory_graph.llm.generate_response.return_value = {
            "tool_calls": [{
                "name": "extract_entities",
                "arguments": {
                    "entities": [
                        {"entity": "Alice", "entity_type": "person", "properties_json": '{"role": "manager"}'},
                    ]
                }
            }]
        }
        result = memory_graph._retrieve_nodes_from_data("test", {"user_id": "u1"})
        assert result["alice"]["properties"] == {"role": "manager"}

    def test_entities_without_properties_backward_compat(self, memory_graph):
        """Legacy LLM responses without properties still work."""
        memory_graph.llm.generate_response.return_value = {
            "tool_calls": [{
                "name": "extract_entities",
                "arguments": {
                    "entities": [
                        {"entity": "Bob", "entity_type": "person"},
                    ]
                }
            }]
        }
        result = memory_graph._retrieve_nodes_from_data("test", {"user_id": "u1"})
        assert result["bob"]["type"] == "person"
        assert result["bob"]["properties"] == {}


# ---------------------------------------------------------------------------
# Relation extraction with properties
# ---------------------------------------------------------------------------

class TestEstablishRelationsWithProperties:
    """Test that _establish_nodes_relations_from_data parses edge properties."""

    def test_relations_with_properties(self, memory_graph):
        memory_graph.llm.generate_response.return_value = {
            "tool_calls": [{
                "name": "establish_relationships",
                "arguments": {
                    "entities": [
                        {
                            "source": "Alice",
                            "relationship": "works at",
                            "destination": "Acme Corp",
                            "properties": {"since": "2020", "role": "engineer"},
                        }
                    ]
                }
            }]
        }
        entity_type_map = {"alice": {"type": "person", "properties": {}}, "acme_corp": {"type": "company", "properties": {}}}
        result = memory_graph._establish_nodes_relations_from_data("test", {"user_id": "u1"}, entity_type_map)
        assert len(result) == 1
        assert result[0]["edge_properties"] == {"since": "2020", "role": "engineer"}
        # Raw properties/properties_json should be cleaned up
        assert "properties" not in result[0]
        assert "properties_json" not in result[0]

    def test_relations_with_properties_json(self, memory_graph):
        """Struct tool returns properties_json."""
        memory_graph.llm.generate_response.return_value = {
            "tool_calls": [{
                "name": "establish_relations",
                "arguments": {
                    "entities": [
                        {
                            "source": "Alice",
                            "relationship": "knows",
                            "destination": "Bob",
                            "properties_json": '{"via": "conference"}',
                        }
                    ]
                }
            }]
        }
        entity_type_map = {"alice": {"type": "person", "properties": {}}, "bob": {"type": "person", "properties": {}}}
        result = memory_graph._establish_nodes_relations_from_data("test", {"user_id": "u1"}, entity_type_map)
        assert result[0]["edge_properties"] == {"via": "conference"}

    def test_relations_without_properties(self, memory_graph):
        """Backward compat: no properties field at all."""
        memory_graph.llm.generate_response.return_value = {
            "tool_calls": [{
                "name": "establish_relationships",
                "arguments": {
                    "entities": [
                        {"source": "Alice", "relationship": "knows", "destination": "Bob"}
                    ]
                }
            }]
        }
        entity_type_map = {"alice": {"type": "person", "properties": {}}, "bob": {"type": "person", "properties": {}}}
        result = memory_graph._establish_nodes_relations_from_data("test", {"user_id": "u1"}, entity_type_map)
        assert result[0]["edge_properties"] == {}


# ---------------------------------------------------------------------------
# _add_entities stores properties
# ---------------------------------------------------------------------------

class TestAddEntitiesStoresProperties:
    """Test that _add_entities passes node/edge properties to Cypher."""

    def test_new_nodes_get_properties(self, memory_graph):
        """When neither source nor destination exist, properties are passed to Cypher."""
        memory_graph._search_source_node = MagicMock(return_value=[])
        memory_graph._search_destination_node = MagicMock(return_value=[])
        memory_graph.graph.query = MagicMock(return_value=[{"source": "alice", "relationship": "KNOWS", "target": "bob"}])

        to_be_added = [{
            "source": "alice",
            "relationship": "KNOWS",
            "destination": "bob",
            "edge_properties": {"via": "conference"},
        }]
        entity_type_map = {
            "alice": {"type": "person", "properties": {"email": "alice@example.com"}},
            "bob": {"type": "person", "properties": {}},
        }

        memory_graph._add_entities(to_be_added, {"user_id": "u1"}, entity_type_map)

        # Verify Cypher was called
        assert memory_graph.graph.query.call_count == 1
        call_args = memory_graph.graph.query.call_args
        cypher = call_args[0][0]
        params = call_args[1]["params"]

        # Node properties should be passed
        assert params["source_node_props"] == {"email": "alice@example.com"}
        assert params["dest_node_props"] == {}
        assert params["edge_props"] == {"via": "conference"}

        # Cypher should include SET ... += $source_node_props
        assert "source += $source_node_props" in cypher
        assert "destination += $dest_node_props" in cypher
        assert "r += $edge_props" in cypher

    def test_existing_nodes_get_properties(self, memory_graph):
        """When both source and destination already exist, properties are still applied."""
        memory_graph._search_source_node = MagicMock(
            return_value=[{"elementId(source_candidate)": "4:abc:0"}]
        )
        memory_graph._search_destination_node = MagicMock(
            return_value=[{"elementId(destination_candidate)": "4:def:1"}]
        )
        memory_graph.graph.query = MagicMock(return_value=[{"source": "alice", "relationship": "KNOWS", "target": "bob"}])

        to_be_added = [{
            "source": "alice",
            "relationship": "KNOWS",
            "destination": "bob",
            "edge_properties": {"capacity": "lead"},
        }]
        entity_type_map = {
            "alice": {"type": "person", "properties": {"title": "Dr."}},
            "bob": {"type": "person", "properties": {}},
        }

        memory_graph._add_entities(to_be_added, {"user_id": "u1"}, entity_type_map)

        call_args = memory_graph.graph.query.call_args
        params = call_args[1]["params"]
        assert params["source_node_props"] == {"title": "Dr."}
        assert params["edge_props"] == {"capacity": "lead"}

    def test_no_properties_backward_compat(self, memory_graph):
        """Legacy entity_type_map with plain string values still works."""
        memory_graph._search_source_node = MagicMock(return_value=[])
        memory_graph._search_destination_node = MagicMock(return_value=[])
        memory_graph.graph.query = MagicMock(return_value=[{"source": "a", "relationship": "R", "target": "b"}])

        to_be_added = [{"source": "a", "relationship": "R", "destination": "b"}]
        # Legacy format: entity_type_map values are plain strings
        entity_type_map = {"a": "person", "b": "thing"}

        memory_graph._add_entities(to_be_added, {"user_id": "u1"}, entity_type_map)

        call_args = memory_graph.graph.query.call_args
        params = call_args[1]["params"]
        assert params["source_node_props"] == {}
        assert params["dest_node_props"] == {}
        assert params["edge_props"] == {}


# ---------------------------------------------------------------------------
# Search returns properties
# ---------------------------------------------------------------------------

class TestSearchReturnsProperties:
    """Test that search() returns properties in results."""

    def test_search_includes_properties(self, memory_graph):
        """search() should include filtered properties in results."""
        # Mock entity extraction
        memory_graph._retrieve_nodes_from_data = MagicMock(
            return_value={"alice": {"type": "person", "properties": {}}}
        )
        # Mock graph search returning properties
        memory_graph._search_graph_db = MagicMock(return_value=[
            {
                "source": "alice",
                "relationship": "KNOWS",
                "destination": "bob",
                "source_properties": {"name": "alice", "email": "a@b.com", "user_id": "u1", "embedding": [0.1]},
                "edge_properties": {"since": "2024", "valid": True, "created_at": 123},
                "destination_properties": {"name": "bob", "role": "dev"},
            }
        ])

        # Mock BM25Okapi to return the items as-is
        mock_bm25_cls = MagicMock()
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_top_n.return_value = [["alice", "KNOWS", "bob"]]
        mock_bm25_cls.return_value = mock_bm25_instance

        with patch("mem0.memory.graph_memory.BM25Okapi", mock_bm25_cls):
            result = memory_graph.search("alice", {"user_id": "u1"})

        assert len(result) == 1
        entry = result[0]
        assert entry["source"] == "alice"
        assert entry["source_properties"] == {"email": "a@b.com"}  # system keys filtered
        assert entry["edge_properties"] == {"since": "2024"}  # valid, created_at filtered
        assert entry["destination_properties"] == {"role": "dev"}  # name filtered

    def test_search_without_properties(self, memory_graph):
        """search() still works when graph returns no properties."""
        memory_graph._retrieve_nodes_from_data = MagicMock(
            return_value={"alice": {"type": "person", "properties": {}}}
        )
        memory_graph._search_graph_db = MagicMock(return_value=[
            {"source": "alice", "relationship": "KNOWS", "destination": "bob"}
        ])

        mock_bm25_cls = MagicMock()
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_top_n.return_value = [["alice", "KNOWS", "bob"]]
        mock_bm25_cls.return_value = mock_bm25_instance

        with patch("mem0.memory.graph_memory.BM25Okapi", mock_bm25_cls):
            result = memory_graph.search("alice", {"user_id": "u1"})

        assert len(result) == 1
        assert "source_properties" not in result[0]
        assert "edge_properties" not in result[0]


# ---------------------------------------------------------------------------
# get_all returns properties
# ---------------------------------------------------------------------------

class TestGetAllReturnsProperties:
    """Test that get_all() returns properties in results."""

    def test_get_all_includes_properties(self, memory_graph):
        """get_all() should include filtered properties."""
        memory_graph.graph.query = MagicMock(return_value=[
            {
                "source": "alice",
                "relationship": "KNOWS",
                "target": "bob",
                "source_properties": {"name": "alice", "custom_field": "val1", "embedding": [0.1]},
                "edge_properties": {"since": "2024", "mentions": 3},
                "target_properties": {"name": "bob"},
            }
        ])

        result = memory_graph.get_all({"user_id": "u1"})
        assert len(result) == 1
        entry = result[0]
        assert entry["source_properties"] == {"custom_field": "val1"}
        assert entry["edge_properties"] == {"since": "2024"}
        assert "target_properties" not in entry  # name is a system key, nothing left

    def test_get_all_without_properties(self, memory_graph):
        """get_all() backward compat with no properties columns."""
        memory_graph.graph.query = MagicMock(return_value=[
            {"source": "alice", "relationship": "KNOWS", "target": "bob"}
        ])

        result = memory_graph.get_all({"user_id": "u1"})
        assert len(result) == 1
        assert "source_properties" not in result[0]


# ---------------------------------------------------------------------------
# remove_spaces_from_entities preserves edge_properties
# ---------------------------------------------------------------------------

class TestRemoveSpacesPreservesProperties:
    """Test that remove_spaces_from_entities preserves edge_properties."""

    def test_preserves_edge_properties(self):
        from mem0.memory.utils import remove_spaces_from_entities
        entities = [{
            "source": "Alice",
            "relationship": "works at",
            "destination": "Acme Corp",
            "edge_properties": {"since": "2024"},
        }]
        result = remove_spaces_from_entities(entities)
        assert len(result) == 1
        assert result[0]["edge_properties"] == {"since": "2024"}
        assert result[0]["source"] == "alice"
        assert result[0]["destination"] == "acme_corp"
