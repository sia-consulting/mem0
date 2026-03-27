import os
from typing import Literal, Optional

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


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
        if api_key is None or api_key == "" or api_key == "your-api-key":
            credential = DefaultAzureCredential()
        else:
            credential = AzureKeyCredential(api_key)

        self.client = EmbeddingsClient(
            endpoint=endpoint,
            credential=credential,
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
