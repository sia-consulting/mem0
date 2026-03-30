from typing import Optional

from mem0.configs.llms.base import BaseLlmConfig


class AzureFoundryProjectsConfig(BaseLlmConfig):
    """
    Configuration class for Azure AI Foundry Projects-specific parameters.
    Inherits from BaseLlmConfig and adds Azure AI Foundry Projects-specific settings.

    Azure AI Foundry Projects uses the azure-ai-projects SDK with AIProjectClient
    which handles authentication, endpoint routing, and credential management
    automatically via project endpoints.
    """

    def __init__(
        self,
        # Base parameters
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        http_client_proxies: Optional[dict] = None,
        # Azure AI Foundry Projects-specific parameters
        endpoint: Optional[str] = None,
        managed_identity_client_id: Optional[str] = None,
    ):
        """
        Initialize Azure AI Foundry Projects configuration.

        Args:
            model: Model deployment name to use, defaults to None
            temperature: Controls randomness, defaults to 0.1
            max_tokens: Maximum tokens to generate, defaults to 2000
            top_p: Nucleus sampling parameter, defaults to 0.1
            top_k: Top-k sampling parameter, defaults to 1
            enable_vision: Enable vision capabilities, defaults to False
            vision_details: Vision detail level, defaults to "auto"
            http_client_proxies: HTTP client proxy settings, defaults to None
            endpoint: Azure AI Foundry project endpoint URL (e.g.,
                "https://<resource>.services.ai.azure.com/api/projects/<project>"),
                defaults to None
            managed_identity_client_id: Client ID of a user-assigned managed identity
                to use with DefaultAzureCredential. Only used when api_key is not
                provided. Defaults to None (system-assigned identity).
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            http_client_proxies=http_client_proxies,
        )

        self.endpoint = endpoint
        self.managed_identity_client_id = managed_identity_client_id
