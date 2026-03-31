import os
from typing import Literal, Optional

try:
    from azure.ai.inference import EmbeddingsClient
except ImportError:
    raise ImportError(
        "The 'azure-ai-inference' library is required. "
        "Please install it using 'pip install azure-ai-inference>=1.0.0b9'."
    )

try:
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    raise ImportError(
        "The 'azure-core' library is required. "
        "Please install it using 'pip install azure-core'."
    )

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    raise ImportError(
        "The 'azure-identity' library is required. "
        "Please install it using 'pip install azure-identity>=1.24.0'."
    )

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase
from mem0.utils.azure_foundry import get_credential_scopes, get_openai_base_url, is_openai_endpoint


class AzureFoundryEmbedding(EmbeddingBase):
    """
    Embedding provider for Azure AI Foundry.

    Automatically selects the underlying SDK based on the endpoint URL:

    * **AI Foundry endpoints** (``*.services.ai.azure.com``) – uses the
      ``azure-ai-inference`` SDK's ``EmbeddingsClient``.
    * **Azure OpenAI / Cognitive Services endpoints**
      (``*.cognitiveservices.azure.com``, ``*.openai.azure.com``) – uses the
      standard ``openai.OpenAI`` client with ``base_url`` set to
      ``{endpoint}/openai/v1/``.  This is the OpenAI-compatible path that
      Azure AI Foundry exposes and is the only path that reliably supports
      embeddings (the AI Foundry gateway's own paths return 404).

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

        managed_identity_client_id = (
            self.config.managed_identity_client_id or os.getenv("AZURE_CLIENT_ID")
        )

        # Resolve API version: config attribute → dedicated embedding env var.
        # Only used for the azure-ai-inference SDK path (AI Foundry endpoints).
        # The OpenAI SDK path uses /openai/v1/ and does not need an api-version.
        api_version = (
            getattr(self.config, "api_version", None)
            or os.getenv("AZURE_FOUNDRY_EMBEDDING_API_VERSION")
        )

        if is_openai_endpoint(endpoint):
            # ── Azure OpenAI / Cognitive Services endpoint ──────────────
            # Use the standard OpenAI client with the /openai/v1/ base path.
            # This is the OpenAI-compatible endpoint that Azure exposes and
            # works with the regular openai SDK (not AzureOpenAI).
            self._use_openai_sdk = True

            base_url = get_openai_base_url(endpoint)

            if api_key and api_key not in ("", "your-api-key"):
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )
            else:
                credential = DefaultAzureCredential(
                    managed_identity_client_id=managed_identity_client_id,
                )
                scopes = get_credential_scopes(endpoint) or [
                    "https://cognitiveservices.azure.com/.default"
                ]
                token_provider = get_bearer_token_provider(credential, *scopes)
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=token_provider(),
                )
        else:
            # ── AI Foundry endpoint ─────────────────────────────────────
            # Use the azure-ai-inference SDK's EmbeddingsClient.
            self._use_openai_sdk = False
            client_kwargs = {}

            if api_version:
                client_kwargs["api_version"] = api_version

            if api_key is None or api_key == "" or api_key == "your-api-key":
                credential = DefaultAzureCredential(
                    managed_identity_client_id=managed_identity_client_id,
                )
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
        if self._use_openai_sdk:
            return self.client.embeddings.create(input=[text], model=self.config.model).data[0].embedding
        else:
            return self.client.embed(input=[text], model=self.config.model).data[0].embedding
