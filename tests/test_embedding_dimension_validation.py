import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

from mem0.configs.base import MemoryConfig


class TestEmbeddingDimensionValidation:
    """Tests for the embedding dimension validation at Memory init."""

    @patch("mem0.memory.main.capture_event")
    @patch("mem0.memory.main.SQLiteManager")
    @patch("mem0.memory.main.VectorStoreFactory")
    @patch("mem0.memory.main.EmbedderFactory")
    @patch("mem0.memory.main.LlmFactory")
    def test_dimension_mismatch_logs_warning(
        self, mock_llm_factory, mock_embedder_factory, mock_vs_factory,
        mock_db, mock_telemetry, caplog
    ):
        """When embedder dims != vector store dims, a warning is logged."""
        from mem0.memory.main import Memory

        # Build a config with mismatched dimensions
        config = MemoryConfig(
            embedder={"provider": "openai", "config": {"embedding_dims": 768}},
            vector_store={"provider": "qdrant", "config": {
                "embedding_model_dims": 1536,
                "path": "/tmp/test_qdrant",
            }},
            llm={"provider": "openai", "config": {}},
        )

        # Mock the factories
        mock_embedding_model = MagicMock()
        mock_embedding_model.config = MagicMock(model="text-embedding-3-small")
        mock_embedder_factory.create.return_value = mock_embedding_model

        mock_vs = MagicMock()
        mock_vs_factory.create.return_value = mock_vs

        mock_llm = MagicMock()
        mock_llm.config = MagicMock(model="gpt-4o")
        mock_llm_factory.create.return_value = mock_llm

        with caplog.at_level(logging.WARNING, logger="mem0.memory.main"):
            Memory(config)

        assert any("Embedding dimension mismatch" in msg for msg in caplog.messages), (
            f"Expected dimension mismatch warning, got: {caplog.messages}"
        )

    @patch("mem0.memory.main.capture_event")
    @patch("mem0.memory.main.SQLiteManager")
    @patch("mem0.memory.main.VectorStoreFactory")
    @patch("mem0.memory.main.EmbedderFactory")
    @patch("mem0.memory.main.LlmFactory")
    def test_matching_dimensions_no_warning(
        self, mock_llm_factory, mock_embedder_factory, mock_vs_factory,
        mock_db, mock_telemetry, caplog
    ):
        """When embedder dims == vector store dims, no warning is logged."""
        from mem0.memory.main import Memory

        config = MemoryConfig(
            embedder={"provider": "openai", "config": {"embedding_dims": 1536}},
            vector_store={"provider": "qdrant", "config": {
                "embedding_model_dims": 1536,
                "path": "/tmp/test_qdrant",
            }},
            llm={"provider": "openai", "config": {}},
        )

        mock_embedding_model = MagicMock()
        mock_embedding_model.config = MagicMock(model="text-embedding-3-small")
        mock_embedder_factory.create.return_value = mock_embedding_model

        mock_vs_factory.create.return_value = MagicMock()

        mock_llm = MagicMock()
        mock_llm.config = MagicMock(model="gpt-4o")
        mock_llm_factory.create.return_value = mock_llm

        with caplog.at_level(logging.WARNING, logger="mem0.memory.main"):
            Memory(config)

        assert not any("Embedding dimension mismatch" in msg for msg in caplog.messages), (
            f"Unexpected dimension mismatch warning: {caplog.messages}"
        )

    @patch("mem0.memory.main.capture_event")
    @patch("mem0.memory.main.SQLiteManager")
    @patch("mem0.memory.main.VectorStoreFactory")
    @patch("mem0.memory.main.EmbedderFactory")
    @patch("mem0.memory.main.LlmFactory")
    def test_no_dims_configured_no_warning(
        self, mock_llm_factory, mock_embedder_factory, mock_vs_factory,
        mock_db, mock_telemetry, caplog
    ):
        """When neither embedder nor vector store dims are explicitly set, no warning."""
        from mem0.memory.main import Memory

        config = MemoryConfig(
            embedder={"provider": "openai", "config": {}},
            vector_store={"provider": "qdrant", "config": {
                "path": "/tmp/test_qdrant",
            }},
            llm={"provider": "openai", "config": {}},
        )

        mock_embedding_model = MagicMock()
        mock_embedding_model.config = MagicMock(model="text-embedding-3-small")
        mock_embedder_factory.create.return_value = mock_embedding_model

        mock_vs_factory.create.return_value = MagicMock()

        mock_llm = MagicMock()
        mock_llm.config = MagicMock(model="gpt-4o")
        mock_llm_factory.create.return_value = mock_llm

        with caplog.at_level(logging.WARNING, logger="mem0.memory.main"):
            Memory(config)

        assert not any("Embedding dimension mismatch" in msg for msg in caplog.messages), (
            f"Unexpected dimension mismatch warning: {caplog.messages}"
        )

    @patch("mem0.memory.main.capture_event")
    @patch("mem0.memory.main.SQLiteManager")
    @patch("mem0.memory.main.VectorStoreFactory")
    @patch("mem0.memory.main.EmbedderFactory")
    @patch("mem0.memory.main.LlmFactory")
    def test_provider_config_is_logged(
        self, mock_llm_factory, mock_embedder_factory, mock_vs_factory,
        mock_db, mock_telemetry, caplog
    ):
        """Memory init logs the configured providers."""
        from mem0.memory.main import Memory

        config = MemoryConfig(
            embedder={"provider": "openai", "config": {}},
            vector_store={"provider": "qdrant", "config": {
                "path": "/tmp/test_qdrant",
            }},
            llm={"provider": "openai", "config": {}},
        )

        mock_embedding_model = MagicMock()
        mock_embedding_model.config = MagicMock(model="text-embedding-3-small")
        mock_embedder_factory.create.return_value = mock_embedding_model

        mock_vs_factory.create.return_value = MagicMock()

        mock_llm = MagicMock()
        mock_llm.config = MagicMock(model="gpt-4o")
        mock_llm_factory.create.return_value = mock_llm

        with caplog.at_level(logging.INFO, logger="mem0.memory.main"):
            Memory(config)

        assert any("Memory initialized" in msg for msg in caplog.messages), (
            f"Expected provider config log, got: {caplog.messages}"
        )
        assert any("openai" in msg for msg in caplog.messages)
