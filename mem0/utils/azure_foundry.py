from typing import List, Optional
from urllib.parse import urlparse


def is_openai_endpoint(endpoint: str) -> bool:
    """Check if an endpoint is an Azure OpenAI / Cognitive Services endpoint.

    Returns True for ``*.cognitiveservices.azure.com`` and
    ``*.openai.azure.com`` endpoints.  Returns False for AI Foundry
    endpoints (``*.services.ai.azure.com``) and anything else.

    When True, the caller should use the ``openai.OpenAI`` client
    instead of the ``azure-ai-inference`` SDK.
    """
    host = (urlparse(endpoint).hostname or "").lower()
    return host.endswith(".cognitiveservices.azure.com") or host.endswith(".openai.azure.com")


def get_openai_base_url(endpoint: str) -> str:
    """Build the ``base_url`` for :class:`openai.OpenAI` from an Azure endpoint.

    Accepts both short and full forms::

        https://<resource>.openai.azure.com
        https://<resource>.openai.azure.com/
        https://<resource>.openai.azure.com/openai/v1/
        https://<resource>.cognitiveservices.azure.com/openai/v1

    If the path already contains ``/openai/v1``, the URL is normalised
    (trailing slash ensured) and returned as-is.  Otherwise
    ``/openai/v1/`` is appended to the origin.
    """
    parsed = urlparse(endpoint)
    path = parsed.path.rstrip("/")
    if path.endswith("/openai/v1"):
        return f"{parsed.scheme}://{parsed.netloc}/openai/v1/"
    return f"{parsed.scheme}://{parsed.netloc}/openai/v1/"


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
