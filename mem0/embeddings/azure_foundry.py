import os
from typing import Literal, Optional

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase
from mem0.utils.azure_foundry import get_credential_scopes


class AzureFoundryEmbedding(EmbeddingBase):
    """
    Embedding provider for Azure AI Foundry using the azure-ai-inference SDK.

    Uses EmbeddingsClient with endpoint + credential authentication.
    Supports both API key and managed identity (passwordless) authentication.
    When no API key is provided, falls back to DefaultAzureCredential for
    passwordless auth via managed identities, Azure CLI, etc.
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        api_key = self.config.api_key or os.getenv("AZURE_FOUNDRY_EMBEDDING_API_KEY")
        endpoint = self.config.openai_base_url or os.getenv("AZURE_FOUNDRY_EMBEDDING_ENDPOINT")

        if not endpoint:
            raise ValueError(
                "Azure AI Foundry embedding endpoint is required. "
                "Set it via config.openai_base_url or the AZURE_FOUNDRY_EMBEDDING_ENDPOINT environment variable."
            )

        # If the API key is not provided or is a placeholder, use DefaultAzureCredential
        # for passwordless authentication via managed identities, Azure CLI, etc.
        # To specify a user-assigned managed identity, set managed_identity_client_id in
        # config or the AZURE_CLIENT_ID env var (used as fallback).
        client_kwargs = {}

        # Allow overriding the API version sent to the Azure endpoint.
        # The azure-ai-inference SDK defaults to "2024-05-01-preview" which may
        # not be supported by all Azure AI Foundry endpoint configurations.
        api_version = getattr(self.config, "api_version", None) or os.getenv("AZURE_FOUNDRY_API_VERSION")
        if api_version:
            client_kwargs["api_version"] = api_version

        if api_key is None or api_key == "" or api_key == "your-api-key":
            managed_identity_client_id = (
                self.config.managed_identity_client_id or os.getenv("AZURE_CLIENT_ID")
            )
            credential = DefaultAzureCredential(
                managed_identity_client_id=managed_identity_client_id,
            )
            # The azure-ai-inference SDK defaults credential_scopes to
            # ["https://ml.azure.com/.default"], which is incorrect for
            # Cognitive Services endpoints (*.cognitiveservices.azure.com).
            # Auto-detect the correct scope from the endpoint URL, or use an
            # explicit override from config.
            scopes = get_credential_scopes(
                endpoint, getattr(self.config, "credential_scopes", None)
            )
            if scopes:
                client_kwargs["credential_scopes"] = scopes
        else:
            credential = AzureKeyCredential(api_key)

        self.client = EmbeddingsClient(
            endpoint=endpoint,
            credential=credential,
            **client_kwargs,
        )

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Azure AI Foundry.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Defaults to None.

        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        return self.client.embed(input=[text], model=self.config.model).data[0].embedding
