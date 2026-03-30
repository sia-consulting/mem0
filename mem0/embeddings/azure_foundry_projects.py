import os
from typing import Literal, Optional

try:
    from azure.ai.projects import AIProjectClient
except ImportError:
    raise ImportError(
        "The 'azure-ai-projects' library is required. "
        "Please install it using 'pip install azure-ai-projects>=2.0.0'."
    )

try:
    from azure.identity import DefaultAzureCredential
except ImportError:
    raise ImportError(
        "The 'azure-identity' library is required. "
        "Please install it using 'pip install azure-identity>=1.24.0'."
    )

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class AzureFoundryProjectsEmbedding(EmbeddingBase):
    """
    Embedding provider for Azure AI Foundry Projects using the azure-ai-projects SDK.

    Uses AIProjectClient to obtain a properly authenticated OpenAI client via
    get_openai_client(), then uses the standard OpenAI embeddings API.
    This is the recommended approach for Azure AI Foundry project endpoints
    (https://<resource>.services.ai.azure.com/api/projects/<project>).

    Authentication is handled via DefaultAzureCredential (Entra ID only).
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        endpoint = self.config.openai_base_url or os.getenv("AZURE_AI_PROJECT_ENDPOINT")

        if not endpoint:
            raise ValueError(
                "Azure AI Foundry project endpoint is required. "
                "Set it via config.openai_base_url or the AZURE_AI_PROJECT_ENDPOINT environment variable."
            )

        managed_identity_client_id = (
            self.config.managed_identity_client_id or os.getenv("AZURE_CLIENT_ID")
        )
        credential = DefaultAzureCredential(
            managed_identity_client_id=managed_identity_client_id,
        )

        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=credential,
        )

        # Let the SDK discover the project's connected Azure OpenAI resource.
        # Do NOT override base_url: AI Foundry project endpoints do not expose
        # an embeddings route at any path (/openai/v1/embeddings returns 404,
        # /models/embeddings returns 404, even the resource-level /models
        # path returns 404).  Without base_url, the SDK returns an AzureOpenAI
        # client that routes to the connected Cognitive Services deployment
        # (e.g. https://<resource>.cognitiveservices.azure.com/openai/
        # deployments/<model>/embeddings) which does work.
        # See: https://github.com/Azure/azure-sdk-for-python/issues/44532
        self.client = project_client.get_openai_client()

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Azure AI Foundry Projects.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Defaults to None.

        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.config.model).data[0].embedding
