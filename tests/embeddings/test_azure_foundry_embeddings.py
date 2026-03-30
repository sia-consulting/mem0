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


def test_init_missing_api_key_uses_default_credential(monkeypatch):
    """When no API key is provided, DefaultAzureCredential is used for managed identity auth."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=ENDPOINT)

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls,
        patch("mem0.embeddings.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        embedder = AzureFoundryEmbedding(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=None)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )


def test_init_with_placeholder_api_key_uses_default_credential(monkeypatch):
    """Placeholder API key should trigger DefaultAzureCredential."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    config = BaseEmbedderConfig(model=MODEL, api_key="your-api-key", openai_base_url=ENDPOINT)

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls,
        patch("mem0.embeddings.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryEmbedding(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=None)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )


def test_init_with_managed_identity_client_id(monkeypatch):
    """User-assigned managed identity client ID should be passed to DefaultAzureCredential."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    client_id = "12345678-1234-1234-1234-123456789abc"
    config = BaseEmbedderConfig(
        model=MODEL, openai_base_url=ENDPOINT, managed_identity_client_id=client_id,
    )

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls,
        patch("mem0.embeddings.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryEmbedding(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=client_id)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )


def test_init_cognitive_services_endpoint_sets_credential_scopes(monkeypatch):
    """Cognitive Services endpoints should auto-detect the correct credential scope."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    cog_endpoint = "https://myresource.cognitiveservices.azure.com/models"
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=cog_endpoint)

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls,
        patch("mem0.embeddings.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryEmbedding(config)
        mock_client_cls.assert_called_once_with(
            endpoint=cog_endpoint,
            credential=mock_cred_instance,
            credential_scopes=["https://cognitiveservices.azure.com/.default"],
        )


def test_init_ai_foundry_endpoint_uses_sdk_default_scopes(monkeypatch):
    """AI Foundry endpoints (*.services.ai.azure.com) should use ai.azure.com scope."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=ENDPOINT)

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls,
        patch("mem0.embeddings.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryEmbedding(config)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )


def test_init_azure_client_id_env_var_fallback(monkeypatch):
    """AZURE_CLIENT_ID env var should be used when managed_identity_client_id is not in config."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id-1234")
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=ENDPOINT)

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient"),
        patch("mem0.embeddings.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        AzureFoundryEmbedding(config)
        mock_cred.assert_called_once_with(managed_identity_client_id="env-client-id-1234")


def test_init_config_client_id_takes_precedence_over_env_var(monkeypatch):
    """Config managed_identity_client_id should take precedence over AZURE_CLIENT_ID env var."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id")
    config_id = "config-client-id-5678"
    config = BaseEmbedderConfig(
        model=MODEL, openai_base_url=ENDPOINT, managed_identity_client_id=config_id,
    )

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient"),
        patch("mem0.embeddings.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        AzureFoundryEmbedding(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=config_id)


def test_init_api_key_auth_no_credential_scopes(monkeypatch):
    """API key auth should not pass credential_scopes."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    cog_endpoint = "https://myresource.cognitiveservices.azure.com/models"
    config = BaseEmbedderConfig(
        model=MODEL, api_key=API_KEY, openai_base_url=cog_endpoint,
    )

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls,
        patch("mem0.embeddings.azure_foundry.AzureKeyCredential") as mock_cred,
    ):
        mock_cred.return_value = "mock-credential"
        AzureFoundryEmbedding(config)
        # API key auth should NOT pass credential_scopes
        mock_client_cls.assert_called_once_with(
            endpoint=cog_endpoint,
            credential="mock-credential",
        )


def test_init_with_api_version_from_env_var(monkeypatch):
    """AZURE_FOUNDRY_API_VERSION env var should be passed to EmbeddingsClient."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_FOUNDRY_API_VERSION", "2025-03-01-preview")
    config = BaseEmbedderConfig(model=MODEL, api_key=API_KEY, openai_base_url=ENDPOINT)

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls,
        patch("mem0.embeddings.azure_foundry.AzureKeyCredential") as mock_cred,
    ):
        mock_cred.return_value = "mock-credential"
        AzureFoundryEmbedding(config)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential="mock-credential",
            api_version="2025-03-01-preview",
        )


def test_init_no_api_version_uses_sdk_default(monkeypatch):
    """When api_version is not set, it should not be passed to the client (SDK default used)."""
    monkeypatch.delenv("AZURE_FOUNDRY_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    config = BaseEmbedderConfig(model=MODEL, api_key=API_KEY, openai_base_url=ENDPOINT)

    with (
        patch("mem0.embeddings.azure_foundry.EmbeddingsClient") as mock_client_cls,
        patch("mem0.embeddings.azure_foundry.AzureKeyCredential") as mock_cred,
    ):
        mock_cred.return_value = "mock-credential"
        AzureFoundryEmbedding(config)
        # No api_version kwarg should be passed
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential="mock-credential",
        )
