from unittest.mock import Mock, patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.azure_foundry_projects import AzureFoundryProjectsEmbedding

ENDPOINT = "https://test-resource.services.ai.azure.com/api/projects/my-project"
MODEL = "text-embedding-3-small"


@pytest.fixture
def mock_project_client():
    """Mock AIProjectClient and the OpenAI client it returns."""
    with (
        patch("mem0.embeddings.azure_foundry_projects.AIProjectClient") as mock_proj_cls,
        patch("mem0.embeddings.azure_foundry_projects.DefaultAzureCredential") as mock_cred,
    ):
        mock_proj = Mock()
        mock_proj_cls.return_value = mock_proj
        mock_openai_client = Mock()
        mock_proj.get_openai_client.return_value = mock_openai_client
        mock_cred_instance = mock_cred.return_value
        yield {
            "project_cls": mock_proj_cls,
            "project_client": mock_proj,
            "openai_client": mock_openai_client,
            "credential_cls": mock_cred,
            "credential_instance": mock_cred_instance,
        }


def test_embed(mock_project_client):
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=ENDPOINT)

    mock_embedding = Mock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_response = Mock()
    mock_response.data = [mock_embedding]
    mock_project_client["openai_client"].embeddings.create.return_value = mock_response

    embedder = AzureFoundryProjectsEmbedding(config)
    result = embedder.embed("Hello, world!")

    mock_project_client["openai_client"].embeddings.create.assert_called_once_with(
        input=["Hello, world!"], model=MODEL
    )
    assert result == [0.1, 0.2, 0.3]


def test_init_with_config(mock_project_client):
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=ENDPOINT)

    embedder = AzureFoundryProjectsEmbedding(config)

    mock_project_client["credential_cls"].assert_called_once_with(
        managed_identity_client_id=None,
    )
    mock_project_client["project_cls"].assert_called_once_with(
        endpoint=ENDPOINT,
        credential=mock_project_client["credential_instance"],
    )
    # Verify base_url is set to /models path for Model Inference API (not /openai/v1)
    mock_project_client["project_client"].get_openai_client.assert_called_once_with(
        base_url=ENDPOINT + "/models",
    )


def test_init_with_env_vars(monkeypatch, mock_project_client):
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", ENDPOINT)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)

    config = BaseEmbedderConfig(model=MODEL)
    embedder = AzureFoundryProjectsEmbedding(config)

    mock_project_client["project_cls"].assert_called_once_with(
        endpoint=ENDPOINT,
        credential=mock_project_client["credential_instance"],
    )


def test_init_missing_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
    config = BaseEmbedderConfig(model=MODEL)

    with (
        patch("mem0.embeddings.azure_foundry_projects.AIProjectClient"),
        patch("mem0.embeddings.azure_foundry_projects.DefaultAzureCredential"),
    ):
        with pytest.raises(ValueError, match="project endpoint is required"):
            AzureFoundryProjectsEmbedding(config)


def test_init_with_managed_identity_client_id(monkeypatch, mock_project_client):
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    client_id = "12345678-1234-1234-1234-123456789abc"
    config = BaseEmbedderConfig(
        model=MODEL, openai_base_url=ENDPOINT, managed_identity_client_id=client_id,
    )

    AzureFoundryProjectsEmbedding(config)
    mock_project_client["credential_cls"].assert_called_once_with(
        managed_identity_client_id=client_id,
    )


def test_init_azure_client_id_env_var_fallback(monkeypatch, mock_project_client):
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id-1234")
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=ENDPOINT)

    AzureFoundryProjectsEmbedding(config)
    mock_project_client["credential_cls"].assert_called_once_with(
        managed_identity_client_id="env-client-id-1234",
    )


def test_init_config_client_id_takes_precedence_over_env_var(monkeypatch, mock_project_client):
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id")
    config_id = "config-client-id-5678"
    config = BaseEmbedderConfig(
        model=MODEL, openai_base_url=ENDPOINT, managed_identity_client_id=config_id,
    )

    AzureFoundryProjectsEmbedding(config)
    mock_project_client["credential_cls"].assert_called_once_with(
        managed_identity_client_id=config_id,
    )


def test_init_trailing_slash_stripped_from_base_url(mock_project_client):
    endpoint_with_slash = ENDPOINT + "/"
    config = BaseEmbedderConfig(model=MODEL, openai_base_url=endpoint_with_slash)

    AzureFoundryProjectsEmbedding(config)

    # Trailing slash should be stripped before appending /models
    mock_project_client["project_client"].get_openai_client.assert_called_once_with(
        base_url=ENDPOINT + "/models",
    )
