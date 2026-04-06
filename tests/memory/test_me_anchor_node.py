"""Tests for Phase 2: "Me" Anchor Node + Per-Agent Graphs.

Tests that the anchor "Me" node is created lazily on first add(), orphan nodes
are connected to it, and get_me_node() returns the correct structure.
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

def _make_config(base_label=True, threshold=0.7, custom_prompt=None,
                 anchor_node_name="me"):
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
    config.graph_store.anchor_node_name = anchor_node_name
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


@pytest.fixture
def memory_graph_custom_anchor():
    """MemoryGraph with a custom anchor node name."""
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
        mg = MemoryGraph(_make_config(anchor_node_name="agent_brain"))
        mg.graph = mock_graph
        mg.llm = mock_llm
        mg.embedding_model = mock_embedder
        yield mg


# ---------------------------------------------------------------------------
# _ensure_me_node
# ---------------------------------------------------------------------------

class TestEnsureMeNode:
    """Tests for the _ensure_me_node() method."""

    def test_first_add_creates_me_node(self, memory_graph):
        """Calling _ensure_me_node issues a MERGE Cypher for the anchor node."""
        memory_graph.graph.query.reset_mock()  # Clear __init__ index calls
        memory_graph._ensure_me_node({"user_id": "u1"})

        memory_graph.graph.query.assert_called_once()
        cypher = memory_graph.graph.query.call_args[0][0]
        params = memory_graph.graph.query.call_args[1]["params"]

        assert "MERGE" in cypher
        assert "is_anchor" in cypher
        assert params["anchor_name"] == "me"
        assert params["user_id"] == "u1"

    def test_subsequent_calls_use_merge_not_create(self, memory_graph):
        """MERGE ensures idempotency — no duplicates on repeated calls."""
        memory_graph.graph.query.reset_mock()  # Clear __init__ index calls
        memory_graph._ensure_me_node({"user_id": "u1"})
        memory_graph._ensure_me_node({"user_id": "u1"})

        # Both calls should use MERGE (idempotent)
        assert memory_graph.graph.query.call_count == 2
        for c in memory_graph.graph.query.call_args_list:
            assert "MERGE" in c[0][0]
            assert "CREATE" not in c[0][0].split("ON CREATE")[0]

    def test_me_node_scoped_by_user_id(self, memory_graph):
        """user_id is included in the MERGE pattern."""
        memory_graph._ensure_me_node({"user_id": "user_alice"})
        params = memory_graph.graph.query.call_args[1]["params"]
        assert params["user_id"] == "user_alice"

    def test_me_node_scoped_by_agent_id(self, memory_graph):
        """agent_id is included in the MERGE pattern when present."""
        memory_graph._ensure_me_node({"user_id": "u1", "agent_id": "agent_007"})
        cypher = memory_graph.graph.query.call_args[0][0]
        params = memory_graph.graph.query.call_args[1]["params"]
        assert "agent_id" in cypher
        assert params["agent_id"] == "agent_007"

    def test_me_node_scoped_by_run_id(self, memory_graph):
        """run_id is included in the MERGE pattern when present."""
        memory_graph._ensure_me_node({"user_id": "u1", "run_id": "run_42"})
        params = memory_graph.graph.query.call_args[1]["params"]
        assert params["run_id"] == "run_42"

    def test_anchor_node_gets_embedding(self, memory_graph):
        """The anchor node should get an embedding set."""
        memory_graph._ensure_me_node({"user_id": "u1"})
        cypher = memory_graph.graph.query.call_args[0][0]
        params = memory_graph.graph.query.call_args[1]["params"]
        assert "setNodeVectorProperty" in cypher
        assert "anchor_embedding" in params


class TestEnsureMeNodeCustomName:
    """Tests for configurable anchor node names."""

    def test_custom_anchor_name(self, memory_graph_custom_anchor):
        """Custom anchor_node_name flows into _ensure_me_node."""
        memory_graph_custom_anchor._ensure_me_node({"user_id": "u1"})
        params = memory_graph_custom_anchor.graph.query.call_args[1]["params"]
        assert params["anchor_name"] == "agent_brain"


# ---------------------------------------------------------------------------
# add() integration
# ---------------------------------------------------------------------------

class TestAddCallsEnsureMeNode:
    """Test that add() creates the anchor node before processing entities."""

    def test_add_creates_me_node_before_entities(self, memory_graph):
        """add() should call _ensure_me_node as the first graph operation."""
        # Mock out internal methods so add() doesn't fail
        memory_graph._ensure_me_node = MagicMock()
        memory_graph._retrieve_nodes_from_data = MagicMock(return_value={})
        memory_graph._establish_nodes_relations_from_data = MagicMock(return_value=[])
        memory_graph._search_graph_db = MagicMock(return_value=[])
        memory_graph._get_delete_entities_from_search_output = MagicMock(return_value=[])
        memory_graph._delete_entities = MagicMock(return_value=[])
        memory_graph._add_entities = MagicMock(return_value=[])
        memory_graph._connect_orphans_to_me = MagicMock()

        memory_graph.add("test data", {"user_id": "u1"})

        memory_graph._ensure_me_node.assert_called_once_with({"user_id": "u1"})

    def test_add_connects_orphans_after_entities(self, memory_graph):
        """add() should call _connect_orphans_to_me after _add_entities."""
        to_be_added = [{"source": "alice", "destination": "pizza", "relationship": "LIKES"}]

        memory_graph._ensure_me_node = MagicMock()
        memory_graph._retrieve_nodes_from_data = MagicMock(return_value={"alice": {"type": "person", "properties": {}}})
        memory_graph._establish_nodes_relations_from_data = MagicMock(return_value=to_be_added)
        memory_graph._search_graph_db = MagicMock(return_value=[])
        memory_graph._get_delete_entities_from_search_output = MagicMock(return_value=[])
        memory_graph._delete_entities = MagicMock(return_value=[])
        memory_graph._add_entities = MagicMock(return_value=[])
        memory_graph._connect_orphans_to_me = MagicMock()

        memory_graph.add("Alice likes pizza", {"user_id": "u1"})

        memory_graph._connect_orphans_to_me.assert_called_once_with(to_be_added, {"user_id": "u1"})


# ---------------------------------------------------------------------------
# _connect_orphans_to_me
# ---------------------------------------------------------------------------

class TestConnectOrphansToMe:
    """Tests for orphan-node → Me connection logic."""

    def test_orphan_nodes_get_connected_to_me(self, memory_graph):
        """Nodes with no other connections should be linked to anchor."""
        to_be_added = [
            {"source": "alice", "destination": "pizza", "relationship": "LIKES"},
        ]

        memory_graph.graph.query.reset_mock()  # Clear __init__ index calls
        # Simulate Neo4j returning one orphan
        memory_graph.graph.query.return_value = [{"orphan_name": "pizza"}]

        memory_graph._connect_orphans_to_me(to_be_added, {"user_id": "u1"})

        memory_graph.graph.query.assert_called_once()
        cypher = memory_graph.graph.query.call_args[0][0]
        params = memory_graph.graph.query.call_args[1]["params"]

        assert "KNOWS_ABOUT" in cypher
        assert "MERGE" in cypher
        assert params["anchor_name"] == "me"
        assert set(params["node_names"]) == {"alice", "pizza"}

    def test_anchor_node_not_in_orphan_candidates(self, memory_graph):
        """The 'me' anchor node itself should never be in the orphan candidate list."""
        to_be_added = [
            {"source": "me", "destination": "python", "relationship": "KNOWS"},
        ]
        memory_graph.graph.query.reset_mock()  # Clear __init__ index calls
        memory_graph.graph.query.return_value = []

        memory_graph._connect_orphans_to_me(to_be_added, {"user_id": "u1"})

        params = memory_graph.graph.query.call_args[1]["params"]
        # "me" should be excluded from node_names
        assert "me" not in params["node_names"]
        assert "python" in params["node_names"]

    def test_empty_to_be_added_skips_orphan_check(self, memory_graph):
        """No orphan check if nothing was added."""
        memory_graph.graph.query.reset_mock()  # Clear __init__ index calls
        memory_graph._connect_orphans_to_me([], {"user_id": "u1"})
        memory_graph.graph.query.assert_not_called()

    def test_orphan_connection_with_agent_id(self, memory_graph):
        """agent_id filter is passed into the orphan-detection Cypher."""
        to_be_added = [
            {"source": "bob", "destination": "tennis", "relationship": "PLAYS"},
        ]
        memory_graph.graph.query.return_value = []

        memory_graph._connect_orphans_to_me(
            to_be_added, {"user_id": "u1", "agent_id": "a1"}
        )

        cypher = memory_graph.graph.query.call_args[0][0]
        params = memory_graph.graph.query.call_args[1]["params"]
        assert "agent_id" in cypher
        assert params["agent_id"] == "a1"


# ---------------------------------------------------------------------------
# get_me_node
# ---------------------------------------------------------------------------

class TestGetMeNode:
    """Tests for get_me_node() retrieval."""

    def test_get_me_node_returns_correct_structure(self, memory_graph):
        """get_me_node returns dict with 'me' and 'connections' keys."""
        memory_graph.graph.query.return_value = [
            {
                "me_node": {"name": "me", "entity_type": "self", "is_anchor": True, "created": 1234},
                "connection": {
                    "source": "me",
                    "relationship": "KNOWS_ABOUT",
                    "destination": "python",
                    "edge_properties": {"created_at": "2024", "mentions": 3},
                    "destination_properties": {"name": "python", "user_id": "u1", "skill_level": "expert"},
                },
            },
        ]

        result = memory_graph.get_me_node({"user_id": "u1"})

        assert result is not None
        assert result["me"]["name"] == "me"
        assert result["me"]["is_anchor"] is True
        assert len(result["connections"]) == 1
        conn = result["connections"][0]
        assert conn["source"] == "me"
        assert conn["relationship"] == "KNOWS_ABOUT"
        assert conn["destination"] == "python"
        # System keys should be filtered from properties
        assert "name" not in conn.get("destination_properties", {})
        assert "user_id" not in conn.get("destination_properties", {})
        # Custom properties should pass through
        assert conn["destination_properties"]["skill_level"] == "expert"

    def test_get_me_node_returns_none_when_not_found(self, memory_graph):
        """Returns None if no Me node exists for the scope."""
        memory_graph.graph.query.return_value = []

        result = memory_graph.get_me_node({"user_id": "u1"})
        assert result is None

    def test_get_me_node_with_no_connections(self, memory_graph):
        """Me node exists but has no connections → empty connections list."""
        memory_graph.graph.query.return_value = [
            {
                "me_node": {"name": "me", "entity_type": "self", "is_anchor": True, "created": 1234},
                "connection": None,
            },
        ]

        result = memory_graph.get_me_node({"user_id": "u1"})
        assert result is not None
        assert result["connections"] == []

    def test_get_me_node_filters_system_keys(self, memory_graph):
        """System reserved keys are filtered from edge and destination properties."""
        from mem0.memory.graph_memory import _SYSTEM_RESERVED_KEYS

        memory_graph.graph.query.return_value = [
            {
                "me_node": {"name": "me", "entity_type": "self", "is_anchor": True, "created": 1234},
                "connection": {
                    "source": "me",
                    "relationship": "KNOWS_ABOUT",
                    "destination": "item",
                    "edge_properties": {"valid": True, "mentions": 2, "custom_key": "val"},
                    "destination_properties": {"name": "item", "embedding": [0.1], "color": "blue"},
                },
            },
        ]

        result = memory_graph.get_me_node({"user_id": "u1"})
        conn = result["connections"][0]

        # System keys filtered
        edge_props = conn.get("edge_properties", {})
        assert "valid" not in edge_props
        assert "mentions" not in edge_props
        assert edge_props.get("custom_key") == "val"

        dest_props = conn.get("destination_properties", {})
        assert "name" not in dest_props
        assert "embedding" not in dest_props
        assert dest_props.get("color") == "blue"

    def test_get_me_node_passes_depth_param(self, memory_graph):
        """depth parameter is forwarded to the Cypher query."""
        memory_graph.graph.query.return_value = []

        memory_graph.get_me_node({"user_id": "u1"}, depth=3)

        params = memory_graph.graph.query.call_args[1]["params"]
        assert params["depth"] == 3

    def test_get_me_node_scoped_by_agent_id(self, memory_graph):
        """agent_id is included in the query when present."""
        memory_graph.graph.query.return_value = []

        memory_graph.get_me_node({"user_id": "u1", "agent_id": "a1"})

        cypher = memory_graph.graph.query.call_args[0][0]
        params = memory_graph.graph.query.call_args[1]["params"]
        assert "agent_id" in cypher
        assert params["agent_id"] == "a1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestAnchorNodeConfig:
    """Test that anchor_node_name config is respected."""

    def test_default_anchor_name(self, memory_graph):
        """Default anchor_node_name is 'me'."""
        assert memory_graph.anchor_node_name == "me"

    def test_custom_anchor_name_from_config(self, memory_graph_custom_anchor):
        """Custom anchor_node_name flows from config."""
        assert memory_graph_custom_anchor.anchor_node_name == "agent_brain"

    def test_custom_anchor_used_in_ensure_me_node(self, memory_graph_custom_anchor):
        """_ensure_me_node uses the custom anchor name."""
        memory_graph_custom_anchor._ensure_me_node({"user_id": "u1"})
        params = memory_graph_custom_anchor.graph.query.call_args[1]["params"]
        assert params["anchor_name"] == "agent_brain"

    def test_custom_anchor_used_in_get_me_node(self, memory_graph_custom_anchor):
        """get_me_node uses the custom anchor name."""
        memory_graph_custom_anchor.graph.query.return_value = []
        memory_graph_custom_anchor.get_me_node({"user_id": "u1"})
        params = memory_graph_custom_anchor.graph.query.call_args[1]["params"]
        assert params["anchor_name"] == "agent_brain"


# ---------------------------------------------------------------------------
# _SYSTEM_RESERVED_KEYS includes Phase 2 keys
# ---------------------------------------------------------------------------

class TestSystemReservedKeysPhase2:
    """Verify is_anchor and entity_type are in _SYSTEM_RESERVED_KEYS."""

    def test_is_anchor_reserved(self):
        from mem0.memory.graph_memory import _SYSTEM_RESERVED_KEYS
        assert "is_anchor" in _SYSTEM_RESERVED_KEYS

    def test_entity_type_reserved(self):
        from mem0.memory.graph_memory import _SYSTEM_RESERVED_KEYS
        assert "entity_type" in _SYSTEM_RESERVED_KEYS


# ---------------------------------------------------------------------------
# Prompt update
# ---------------------------------------------------------------------------

class TestPromptIncludesAnchorGuidance:
    """Verify EXTRACT_RELATIONS_PROMPT mentions the anchor node concept."""

    def test_prompt_mentions_anchor(self):
        from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT
        assert "anchor" in EXTRACT_RELATIONS_PROMPT.lower()

    def test_prompt_mentions_me_node(self):
        from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT
        # The prompt should explain that USER_ID is the "Me" anchor
        assert "Me" in EXTRACT_RELATIONS_PROMPT
