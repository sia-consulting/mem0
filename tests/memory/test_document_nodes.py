"""Tests for Phase 5: Document Nodes with Vector Store Bridge.

Tests for chunk_text(), add_document(), and load_document() methods.
"""

import hashlib
import uuid
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
# chunk_text() tests
# ---------------------------------------------------------------------------

from mem0.memory.utils import chunk_text


class TestChunkText:
    """Tests for the chunk_text() utility."""

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_none_string(self):
        assert chunk_text(None) == []

    def test_short_text(self):
        """Text shorter than chunk_size returns a single chunk."""
        result = chunk_text("Hello world", chunk_size=100, overlap=20)
        assert result == ["Hello world"]

    def test_basic_chunking(self):
        """Long text is split into multiple chunks."""
        text = "word " * 300  # 1500 chars
        result = chunk_text(text, chunk_size=500, overlap=100)
        assert len(result) > 1
        # All chunks non-empty
        assert all(c for c in result)

    def test_overlap(self):
        """Chunks overlap — the end of one chunk and start of the next share content."""
        text = " ".join(f"word{i}" for i in range(200))
        result = chunk_text(text, chunk_size=200, overlap=50)
        assert len(result) > 2
        # Check that chunks do overlap (last part of one should be in start of next)
        for i in range(len(result) - 1):
            # Some suffix of chunk[i] should appear as prefix of chunk[i+1]
            tail = result[i][-40:]
            assert any(
                tail[j:] in result[i + 1][:80]
                for j in range(len(tail))
                if len(tail[j:]) > 5
            ), "Expected overlap between consecutive chunks"

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            chunk_text("hello", chunk_size=100, overlap=100)

    def test_exact_boundary(self):
        """Text whose length equals chunk_size produces one chunk."""
        text = "x" * 100
        result = chunk_text(text, chunk_size=100, overlap=10)
        assert len(result) == 1
        assert result[0] == text

    def test_whitespace_break(self):
        """Chunks should prefer breaking at whitespace."""
        text = "alpha " * 200  # 1200 chars
        result = chunk_text(text, chunk_size=500, overlap=50)
        # Every chunk (except possibly last) should end without a partial word
        for c in result:
            # No word should be cut
            assert not c.endswith("alph")


# ---------------------------------------------------------------------------
# MemoryGraph helpers
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
        mock_llm.generate_response.return_value = "A concise summary."
        MockLlm.create.return_value = mock_llm

        from mem0.memory.graph_memory import MemoryGraph
        graph = MemoryGraph(_make_config())
        graph.graph = mock_graph
        graph.llm = mock_llm
        graph.embedding_model = mock_embedder
        yield graph


@pytest.fixture
def mock_vs():
    """A mock vector store."""
    vs = MagicMock()
    vs.insert = MagicMock()
    vs.search = MagicMock(return_value=[])
    vs.list = MagicMock(return_value=[])
    return vs


@pytest.fixture
def mock_emb():
    """A mock embedding model."""
    emb = MagicMock()
    emb.embed.return_value = [0.2] * 1536
    return emb


# ---------------------------------------------------------------------------
# add_document() tests
# ---------------------------------------------------------------------------

class TestAddDocument:
    """Tests for MemoryGraph.add_document()."""

    def test_basic_add(self, mg, mock_vs, mock_emb):
        """add_document creates chunks, stores them, and creates a graph node."""
        # The graph mock needs to return results for add_node's MERGE/MATCH queries.
        # add_node will: 1) _ensure_me_node query, 2) the main MERGE query
        mg.graph.query.return_value = [
            {"name": "test_doc", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        content = "Hello world. " * 100  # ~1300 chars
        result = mg.add_document(
            content, FILTERS,
            title="Test Doc",
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        # Should have created graph node
        assert result["name"] == "test_doc"
        assert result["vector_tag"].startswith("doc:")
        assert result["chunk_count"] > 0

        # Should have inserted chunks into vector store
        assert mock_vs.insert.call_count == result["chunk_count"]

        # Each insert should have correct metadata
        first_call_payload = mock_vs.insert.call_args_list[0]
        payload = first_call_payload[1]["payloads"][0] if "payloads" in first_call_payload[1] else first_call_payload[0][2][0]
        assert payload["document_tag"] == result["vector_tag"]
        assert payload["is_document_chunk"] is True
        assert payload["user_id"] == "u1"

    def test_title_used_as_slug(self, mg, mock_vs, mock_emb):
        """Title is used to derive the document tag."""
        mg.graph.query.return_value = [
            {"name": "my_report", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        result = mg.add_document(
            "Some long content here." * 50,
            FILTERS,
            title="My Report",
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        assert result["vector_tag"] == "doc:my_report"

    def test_no_title_uses_hash(self, mg, mock_vs, mock_emb):
        """Without title, a hash-based slug is used."""
        mg.graph.query.return_value = [
            {"name": "dummy", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        content = "Some content for testing hashing."
        result = mg.add_document(
            content, FILTERS,
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        assert result["vector_tag"].startswith("doc:")
        # Should contain a hex hash
        tag_suffix = result["vector_tag"][4:]
        assert len(tag_suffix) == 12  # md5 hex truncated

    def test_source_node_and_relationship(self, mg, mock_vs, mock_emb):
        """add_document can connect to a specific source node."""
        # Mock get_node for source check in add_node
        orig_get_node = mg.get_node
        mg.get_node = MagicMock(return_value={
            "name": "project_alpha", "entity_type": "project",
            "is_anchor": False, "properties": {}, "edge_count": 2,
        })
        mg.graph.query.return_value = [
            {"name": "design_spec", "labels": ["__Entity__"],
             "connected_from": "project_alpha", "via_relationship": "HAS_DOCUMENT"}
        ]

        result = mg.add_document(
            "Document content " * 100,
            FILTERS,
            title="Design Spec",
            source_node="Project Alpha",
            relationship="HAS_DOCUMENT",
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        assert result.get("connected_from") == "project_alpha"
        assert result.get("via_relationship") == "HAS_DOCUMENT"

    def test_source_url_in_properties(self, mg, mock_vs, mock_emb):
        """source_url is stored as a node property."""
        mg.graph.query.return_value = [
            {"name": "report", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        result = mg.add_document(
            "Report content " * 100,
            FILTERS,
            title="Report",
            source_url="https://example.com/report.pdf",
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        assert result["properties"]["source_url"] == "https://example.com/report.pdf"

    def test_custom_properties(self, mg, mock_vs, mock_emb):
        """Extra properties are merged into the document node."""
        mg.graph.query.return_value = [
            {"name": "analysis", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        result = mg.add_document(
            "Analysis text " * 100,
            FILTERS,
            title="Analysis",
            properties={"department": "engineering", "priority": "high"},
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        assert result["properties"]["department"] == "engineering"
        assert result["properties"]["priority"] == "high"

    def test_agent_and_run_in_chunk_metadata(self, mg, mock_vs, mock_emb):
        """agent_id and run_id are included in chunk metadata."""
        mg.graph.query.return_value = [
            {"name": "doc", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        mg.add_document(
            "Chunk content " * 100,
            FILTERS_RUN,
            title="Doc",
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        payload = mock_vs.insert.call_args_list[0][1]["payloads"][0] if "payloads" in mock_vs.insert.call_args_list[0][1] else mock_vs.insert.call_args_list[0][0][2][0]
        assert payload["agent_id"] == "a1"
        assert payload["run_id"] == "r1"

    def test_llm_summary_used(self, mg, mock_vs, mock_emb):
        """The LLM is called to produce a summary stored as content_summary."""
        mg.llm.generate_response.return_value = "LLM summary of the doc."
        mg.graph.query.return_value = [
            {"name": "doc", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        result = mg.add_document(
            "Content " * 100,
            FILTERS,
            title="Doc",
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        assert result["properties"]["content_summary"] == "LLM summary of the doc."
        mg.llm.generate_response.assert_called_once()

    def test_llm_failure_fallback(self, mg, mock_vs, mock_emb):
        """If LLM fails, summary falls back to truncated content."""
        mg.llm.generate_response.side_effect = RuntimeError("LLM down")
        mg.graph.query.return_value = [
            {"name": "doc", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        result = mg.add_document(
            "Fallback " * 50,
            FILTERS,
            title="Doc",
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        # Should still succeed with a truncated summary
        assert "content_summary" in result["properties"]
        assert "Fallback" in result["properties"]["content_summary"]

    def test_custom_chunk_size(self, mg, mock_vs, mock_emb):
        """Custom chunk_size and overlap are respected."""
        mg.graph.query.return_value = [
            {"name": "doc", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        content = "x " * 2000  # 4000 chars
        result = mg.add_document(
            content, FILTERS,
            title="Doc",
            vector_store=mock_vs,
            embedding_model=mock_emb,
            chunk_size=500,
            overlap=50,
        )

        # With 4000 chars / 500 chunk_size, we expect many chunks
        assert result["chunk_count"] >= 4

    def test_entity_type_is_document(self, mg, mock_vs, mock_emb):
        """The created graph node should have entity_type='document'."""
        mg.graph.query.return_value = [
            {"name": "doc", "labels": ["__Entity__"],
             "connected_from": "me", "via_relationship": "KNOWS_ABOUT"}
        ]

        result = mg.add_document(
            "Content " * 100,
            FILTERS,
            title="Doc",
            vector_store=mock_vs,
            embedding_model=mock_emb,
        )

        assert result["entity_type"] == "document"


# ---------------------------------------------------------------------------
# add_document() — validation
# ---------------------------------------------------------------------------

class TestAddDocumentValidation:
    """Validation tests for add_document."""

    def test_empty_content(self, mg, mock_vs, mock_emb):
        with pytest.raises(ValueError, match="non-empty string"):
            mg.add_document("", FILTERS, vector_store=mock_vs, embedding_model=mock_emb)

    def test_none_content(self, mg, mock_vs, mock_emb):
        with pytest.raises(ValueError, match="non-empty string"):
            mg.add_document(None, FILTERS, vector_store=mock_vs, embedding_model=mock_emb)

    def test_no_vector_store(self, mg, mock_emb):
        with pytest.raises(ValueError, match="vector_store"):
            mg.add_document("content", FILTERS, embedding_model=mock_emb)

    def test_no_embedding_model(self, mg, mock_vs):
        with pytest.raises(ValueError, match="embedding_model"):
            mg.add_document("content", FILTERS, vector_store=mock_vs)


# ---------------------------------------------------------------------------
# load_document() tests
# ---------------------------------------------------------------------------

class TestLoadDocument:
    """Tests for MemoryGraph.load_document()."""

    def test_load_without_query(self, mg, mock_vs, mock_emb):
        """load_document without query returns chunks via list()."""
        # Mock get_node to return a document node
        mg.get_node = MagicMock(return_value={
            "name": "my_doc",
            "entity_type": "document",
            "is_anchor": False,
            "properties": {
                "vector_tag": "doc:my_doc",
                "chunk_count": "3",
                "title": "My Doc",
            },
            "edge_count": 1,
        })

        mock_vs.list.return_value = [
            ("id1", {"data": "chunk 0 text", "chunk_index": 0}),
            ("id2", {"data": "chunk 1 text", "chunk_index": 1}),
            ("id3", {"data": "chunk 2 text", "chunk_index": 2}),
        ]

        result = mg.load_document(
            "my_doc", FILTERS,
            vector_store=mock_vs, embedding_model=mock_emb,
        )

        assert len(result) == 3
        assert result[0]["chunk_index"] == 0
        assert result[0]["text"] == "chunk 0 text"
        assert result[2]["chunk_index"] == 2

    def test_load_with_query(self, mg, mock_vs, mock_emb):
        """load_document with query performs semantic search."""
        mg.get_node = MagicMock(return_value={
            "name": "report",
            "entity_type": "document",
            "is_anchor": False,
            "properties": {"vector_tag": "doc:report"},
            "edge_count": 1,
        })

        mock_vs.search.return_value = [
            {"payload": {"data": "relevant chunk", "chunk_index": 5}, "score": 0.95},
            {"payload": {"data": "less relevant", "chunk_index": 2}, "score": 0.80},
        ]

        result = mg.load_document(
            "report", FILTERS,
            query="quarterly revenue",
            vector_store=mock_vs, embedding_model=mock_emb,
        )

        assert len(result) == 2
        assert result[0]["text"] == "relevant chunk"
        assert result[0]["score"] == 0.95
        assert result[1]["chunk_index"] == 2

        # Verify search was called with correct filters
        mock_vs.search.assert_called_once()
        call_kwargs = mock_vs.search.call_args
        filters_arg = call_kwargs[1].get("filters") or call_kwargs[0][3]
        assert filters_arg["document_tag"] == "doc:report"
        assert filters_arg["user_id"] == "u1"

    def test_load_with_limit(self, mg, mock_vs, mock_emb):
        """load_document respects the limit parameter."""
        mg.get_node = MagicMock(return_value={
            "name": "doc",
            "entity_type": "document",
            "is_anchor": False,
            "properties": {"vector_tag": "doc:doc"},
            "edge_count": 1,
        })

        mock_vs.list.return_value = [
            ("id1", {"data": "chunk 0", "chunk_index": 0}),
            ("id2", {"data": "chunk 1", "chunk_index": 1}),
        ]

        result = mg.load_document(
            "doc", FILTERS, limit=2,
            vector_store=mock_vs, embedding_model=mock_emb,
        )

        assert len(result) <= 2

    def test_load_sorted_by_chunk_index(self, mg, mock_vs, mock_emb):
        """Chunks returned without query are sorted by chunk_index."""
        mg.get_node = MagicMock(return_value={
            "name": "doc",
            "entity_type": "document",
            "is_anchor": False,
            "properties": {"vector_tag": "doc:doc"},
            "edge_count": 1,
        })

        mock_vs.list.return_value = [
            ("id3", {"data": "chunk 2", "chunk_index": 2}),
            ("id1", {"data": "chunk 0", "chunk_index": 0}),
            ("id2", {"data": "chunk 1", "chunk_index": 1}),
        ]

        result = mg.load_document(
            "doc", FILTERS,
            vector_store=mock_vs, embedding_model=mock_emb,
            limit=10,
        )

        indices = [r["chunk_index"] for r in result]
        assert indices == sorted(indices)

    def test_load_with_agent_scope(self, mg, mock_vs, mock_emb):
        """agent_id is included in vector store filters."""
        mg.get_node = MagicMock(return_value={
            "name": "doc",
            "entity_type": "document",
            "is_anchor": False,
            "properties": {"vector_tag": "doc:doc"},
            "edge_count": 1,
        })

        mock_vs.list.return_value = []

        mg.load_document(
            "doc", FILTERS_AGENT,
            vector_store=mock_vs, embedding_model=mock_emb,
        )

        call_kwargs = mock_vs.list.call_args
        filters_arg = call_kwargs[1].get("filters") or call_kwargs[0][0]
        assert filters_arg["agent_id"] == "a1"


# ---------------------------------------------------------------------------
# load_document() — validation
# ---------------------------------------------------------------------------

class TestLoadDocumentValidation:
    """Validation tests for load_document."""

    def test_empty_node_name(self, mg, mock_vs, mock_emb):
        with pytest.raises(ValueError, match="non-empty string"):
            mg.load_document("", FILTERS, vector_store=mock_vs, embedding_model=mock_emb)

    def test_no_vector_store(self, mg, mock_emb):
        with pytest.raises(ValueError, match="vector_store"):
            mg.load_document("doc", FILTERS, embedding_model=mock_emb)

    def test_no_embedding_model(self, mg, mock_vs):
        with pytest.raises(ValueError, match="embedding_model"):
            mg.load_document("doc", FILTERS, vector_store=mock_vs)

    def test_node_not_found(self, mg, mock_vs, mock_emb):
        mg.get_node = MagicMock(return_value=None)
        with pytest.raises(ValueError, match="not found"):
            mg.load_document("missing", FILTERS, vector_store=mock_vs, embedding_model=mock_emb)

    def test_node_not_document(self, mg, mock_vs, mock_emb):
        """A node without vector_tag is not a document node."""
        mg.get_node = MagicMock(return_value={
            "name": "alice",
            "entity_type": "person",
            "is_anchor": False,
            "properties": {"age": "30"},
            "edge_count": 3,
        })
        with pytest.raises(ValueError, match="not a document node"):
            mg.load_document("alice", FILTERS, vector_store=mock_vs, embedding_model=mock_emb)
