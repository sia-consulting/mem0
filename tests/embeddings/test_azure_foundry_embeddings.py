from unittest.mock import Mock, patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.azure_foundry import AzureFoundryEmbedding

ENDPOINT = "https://test-resource.services.ai.azure.com/models"
API_KEY = "test-api-key"
MODEL = "text-embedding-3-small"


@pytest.fixture
def mock_embeddings_client():
    with patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls:
        mock_client = Mock()
        mock_client_cls.return_value = mock_client
        yield mock_client


def test_embed(mock_embeddings_client):
    config = BaseEmbedderConfig(model=MODEL, api_key=API_KEY, openai_base_url=ENDPOINT)

    mock_embedding = Mock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_response = Mock()
    mock_response.data = [mock_embedding]
    mock_embeddings_client.embed.return_value = mock_response

    embedder = AzureFoundryEmbedding(config)
    result = embedder.embed("Hello, world!")

    mock_embeddings_client.embed.assert_called_once_with(input=["Hello, world!"], model=MODEL)
    assert result == [0.1, 0.2, 0.3]


def test_init_with_config(monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_ENDPOINT", raising=False)

    config = BaseEmbedderConfig(model=MODEL, api_key=API_KEY, openai_base_url=ENDPOINT)

    with patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls:
        with patch("mem0.embeddings.azure_foundry.AzureKeyCredential") as mock_cred:
            mock_cred.return_value = "mock-credential"
            embedder = AzureFoundryEmbedding(config)
            mock_cred.assert_called_once_with(API_KEY)
            mock_client_cls.assert_called_once_with(
                endpoint=ENDPOINT,
                credential="mock-credential",
            )


def test_init_with_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_FOUNDRY_EMBEDDING_ENDPOINT", "https://env-endpoint.services.ai.azure.com/models")

    config = BaseEmbedderConfig(model=MODEL)

    with patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls:
        with patch("mem0.embeddings.azure_foundry.AzureKeyCredential") as mock_cred:
            mock_cred.return_value = "mock-credential"
            embedder = AzureFoundryEmbedding(config)
            mock_cred.assert_called_once_with("env-key")
            mock_client_cls.assert_called_once_with(
                endpoint="https://env-endpoint.services.ai.azure.com/models",
                credential="mock-credential",
            )


def test_init_missing_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_ENDPOINT", raising=False)
    config = BaseEmbedderConfig(model=MODEL, api_key=API_KEY)

    with patch("mem0.embeddings.azure_foundry.EmbeddingsClient"):
        with pytest.raises(ValueError, match="endpoint is required"):
            AzureFoundryEmbedding(config)


def test_init_missing_api_key(monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=ENDPOINT)

    with patch("mem0.embeddings.azure_foundry.EmbeddingsClient"):
        with pytest.raises(ValueError, match="API key is required"):
            AzureFoundryEmbedding(config)
