from typing import List, Optional

from mem0.configs.llms.base import BaseLlmConfig


class AzureFoundryConfig(BaseLlmConfig):
    """
    Configuration class for Azure AI Foundry-specific parameters.
    Inherits from BaseLlmConfig and adds Azure AI Foundry-specific settings.

    Azure AI Foundry uses the azure-ai-inference SDK with a simplified
    endpoint + credential authentication model, unlike the legacy Azure OpenAI
    which requires azure_deployment, azure_endpoint, and api_version separately.
    """

    def __init__(
        self,
        # Base parameters
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        http_client_proxies: Optional[dict] = None,
        # Azure AI Foundry-specific parameters
        endpoint: Optional[str] = None,
        managed_identity_client_id: Optional[str] = None,
        credential_scopes: Optional[List[str]] = None,
        api_version: Optional[str] = None,
    ):
        """
        Initialize Azure AI Foundry configuration.

        Args:
            model: Model deployment name to use, defaults to None
            temperature: Controls randomness, defaults to 0.1
            api_key: Azure AI Foundry API key, defaults to None. When not provided,
                DefaultAzureCredential is used for passwordless auth.
            max_tokens: Maximum tokens to generate, defaults to 2000
            top_p: Nucleus sampling parameter, defaults to 0.1
            top_k: Top-k sampling parameter, defaults to 1
            enable_vision: Enable vision capabilities, defaults to False
            vision_details: Vision detail level, defaults to "auto"
            http_client_proxies: HTTP client proxy settings, defaults to None
            endpoint: Azure AI Foundry endpoint URL (e.g.,
                "https://<resource>.services.ai.azure.com/models"), defaults to None
            managed_identity_client_id: Client ID of a user-assigned managed identity
                to use with DefaultAzureCredential. Only used when api_key is not
                provided. Defaults to None (system-assigned identity).
            credential_scopes: OAuth token scopes for managed identity auth. When
                None, the scope is auto-detected from the endpoint URL. Explicitly
                set this to override auto-detection (e.g.,
                ["https://cognitiveservices.azure.com/.default"]).
            api_version: Azure API version to use for requests. When None, the
                azure-ai-inference SDK default is used. Override this if the
                endpoint returns "API version not supported" (e.g.,
                "2025-03-01-preview"). Can also be set via the
                AZURE_FOUNDRY_API_VERSION environment variable.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            http_client_proxies=http_client_proxies,
        )

        self.endpoint = endpoint
        self.managed_identity_client_id = managed_identity_client_id
        self.credential_scopes = credential_scopes
        self.api_version = api_version
