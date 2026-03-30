from typing import List, Optional
from urllib.parse import urlparse


def get_credential_scopes(endpoint: str, config_scopes: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Determine the correct OAuth credential scopes for an Azure endpoint.

    The azure-ai-inference SDK defaults to ["https://ml.azure.com/.default"]
    which only works for AI Foundry (*.services.ai.azure.com) endpoints.
    Cognitive Services endpoints (*.cognitiveservices.azure.com) and Azure
    OpenAI endpoints (*.openai.azure.com) require a different scope.

    Args:
        endpoint: The Azure endpoint URL.
        config_scopes: Explicit scopes from user configuration. When provided,
            these take precedence over auto-detection.

    Returns:
        A list of scope strings, or None to use the SDK default.
    """
    if config_scopes:
        return config_scopes

    host = urlparse(endpoint).hostname or ""
    if host.endswith(".cognitiveservices.azure.com") or host.endswith(".openai.azure.com"):
        return ["https://cognitiveservices.azure.com/.default"]

    # AI Foundry endpoints (*.services.ai.azure.com) require the
    # "https://ai.azure.com/.default" scope for managed-identity /
    # DefaultAzureCredential auth.  The SDK default
    # ("https://ml.azure.com/.default") is only valid for user-delegated
    # flows and the Cognitive Services scope causes an "audience is
    # incorrect" 401 error from the AI Foundry gateway.
    if host.endswith(".services.ai.azure.com"):
        return ["https://ai.azure.com/.default"]

    return None
