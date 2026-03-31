from unittest.mock import MagicMock, Mock, patch

# langchain_memgraph and rank_bm25 are optional deps — mock them so tests run without install
_memgraph_mock = Mock()
patch.dict("sys.modules", {
    "langchain_memgraph": _memgraph_mock,
    "langchain_memgraph.graphs": _memgraph_mock,
    "langchain_memgraph.graphs.memgraph": _memgraph_mock,
    "rank_bm25": Mock(),
}).start()

from mem0.memory.memgraph_memory import MemoryGraph as MemgraphMemoryGraph  # noqa: E402

MemoryGraph = MemgraphMemoryGraph


def _make_instance():
    with patch.object(MemoryGraph, "__init__", return_value=None):
        instance = MemoryGraph.__new__(MemoryGraph)
        instance.llm_provider = "openai"
        instance.llm = MagicMock()
        instance.embedding_model = MagicMock()
        instance.config = MagicMock()
        instance.config.graph_store.custom_prompt = None
        return instance


class TestRetrieveNodesFromData:
    """Tests for _retrieve_nodes_from_data in MemoryGraph."""

    def test_normal_entities_extracted(self):
        instance = _make_instance()
        instance.llm.generate_response.return_value = {
            "tool_calls": [{"name": "extract_entities", "arguments": {"entities": [
                {"entity": "Alice", "entity_type": "person"},
                {"entity": "hiking", "entity_type": "activity"},
            ]}}]
        }
        result = instance._retrieve_nodes_from_data("Alice loves hiking", {"user_id": "u1"})
        assert result == {"alice": "person", "hiking": "activity"}

    def test_malformed_entity_missing_entity_type_is_skipped(self):
        """LLM returns entity dict without entity_type — should skip it, keep valid ones.
        Reproduces the exact data from issue #4055."""
        instance = _make_instance()
        instance.llm.generate_response.return_value = {
            "tool_calls": [{"name": "extract_entities", "arguments": {"entities": [
                {"entity": "matrix multiplication", "entity_type": "task"},
                {"entity": "task"},
                {"entity": "ReLU", "entity_type": "task"},
            ]}}]
        }
        result = instance._retrieve_nodes_from_data("some text", {"user_id": "u1"})
        assert "matrix_multiplication" in result
        assert "relu" in result
        assert "task" not in result

    def test_missing_entities_key_returns_empty(self):
        """LLM returns extract_entities tool call without 'entities' key — should not crash.
        Reproduces the exact scenario from issue #4238."""
        instance = _make_instance()
        instance.llm.generate_response.return_value = {
            "tool_calls": [{"name": "extract_entities", "arguments": {"text": "Hello."}}]
        }
        result = instance._retrieve_nodes_from_data("Hello.", {"user_id": "u1"})
        assert result == {}

    def test_none_tool_calls_returns_empty(self):
        instance = _make_instance()
        instance.llm.generate_response.return_value = {"tool_calls": None}
        result = instance._retrieve_nodes_from_data("hello world", {"user_id": "u1"})
        assert result == {}


class TestEstablishNodesRelationsFromData:
    """Tests for _establish_nodes_relations_from_data in MemoryGraph."""

    def test_none_response_does_not_crash(self):
        """openai_structured returns None when no relations found — must not crash.
        Exact crash from issue #4055: TypeError: 'NoneType' object is not subscriptable."""
        instance = _make_instance()
        instance.llm.generate_response.return_value = None
        result = instance._establish_nodes_relations_from_data(
            "Hello world", {"user_id": "u1"}, {}
        )
        assert result == []

    def test_empty_tool_calls_returns_empty(self):
        instance = _make_instance()
        instance.llm.generate_response.return_value = {"tool_calls": []}
        result = instance._establish_nodes_relations_from_data(
            "Hello world", {"user_id": "u1"}, {}
        )
        assert result == []

    def test_valid_entities_returned(self):
        instance = _make_instance()
        instance.llm.generate_response.return_value = {
            "tool_calls": [{"name": "add_entities", "arguments": {"entities": [
                {"source": "alice", "relationship": "loves", "destination": "hiking"}
            ]}}]
        }
        result = instance._establish_nodes_relations_from_data(
            "Alice loves hiking", {"user_id": "u1"}, {"alice": "person"}
        )
        assert len(result) == 1
        assert result[0]["source"] == "alice"


class TestAddEntitiesCypherEscaping:
    """Tests that _add_entities wraps entity types in backticks so special
    characters (e.g. '/') don't break the Cypher parser."""

    def _run_add_entities(self, entity_type_map, source_search=None, dest_search=None):
        """Helper: call _add_entities with mocked graph and return the Cypher string."""
        instance = _make_instance()
        instance.threshold = 0.9
        instance.embedding_model.embed.return_value = [0.1, 0.2]
        instance.graph = MagicMock()
        instance._search_source_node = MagicMock(return_value=source_search)
        instance._search_destination_node = MagicMock(return_value=dest_search)

        to_be_added = [{"source": "acme", "relationship": "IS_A", "destination": "corp"}]
        instance._add_entities(to_be_added, {"user_id": "u1"}, entity_type_map)

        cypher = instance.graph.query.call_args[0][0]
        return cypher

    def test_slash_in_entity_type_both_new(self):
        """Entity type 'organization/team' must be backtick-escaped in Cypher labels."""
        cypher = self._run_add_entities(
            {"acme": "organization/team", "corp": "organization/parent"},
        )
        assert ":`organization/team`" in cypher
        assert ":`organization/parent`" in cypher

    def test_slash_in_entity_type_source_exists(self):
        """When source exists but destination is new, destination type is backtick-escaped."""
        cypher = self._run_add_entities(
            {"acme": "person", "corp": "org/unit"},
            source_search=[{"id(source_candidate)": 42}],
            dest_search=None,
        )
        assert ":`org/unit`" in cypher

    def test_slash_in_entity_type_dest_exists(self):
        """When destination exists but source is new, source type is backtick-escaped."""
        cypher = self._run_add_entities(
            {"acme": "org/unit", "corp": "company"},
            source_search=None,
            dest_search=[{"id(destination_candidate)": 99}],
        )
        assert ":`org/unit`" in cypher

    def test_normal_entity_type_still_backtick_escaped(self):
        """Even plain entity types get backticks (harmless, consistent)."""
        cypher = self._run_add_entities({"acme": "person", "corp": "company"})
        assert ":`person`" in cypher
        assert ":`company`" in cypher
