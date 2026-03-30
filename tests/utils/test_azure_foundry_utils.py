from mem0.utils.azure_foundry import get_credential_scopes


def test_cognitive_services_endpoint():
    """Cognitive Services endpoint should return cognitiveservices scope."""
    scopes = get_credential_scopes("https://myresource.cognitiveservices.azure.com/models")
    assert scopes == ["https://cognitiveservices.azure.com/.default"]


def test_cognitive_services_endpoint_case_insensitive():
    """Endpoint matching should be case-insensitive."""
    scopes = get_credential_scopes("https://MyResource.CognitiveServices.Azure.Com/models")
    assert scopes == ["https://cognitiveservices.azure.com/.default"]


def test_openai_azure_endpoint():
    """Azure OpenAI endpoint should return cognitiveservices scope."""
    scopes = get_credential_scopes("https://myresource.openai.azure.com/")
    assert scopes == ["https://cognitiveservices.azure.com/.default"]


def test_ai_foundry_endpoint_returns_ai_scope():
    """AI Foundry endpoint should return ai.azure.com scope."""
    scopes = get_credential_scopes("https://myresource.services.ai.azure.com/models")
    assert scopes == ["https://ai.azure.com/.default"]


def test_explicit_scopes_override():
    """Explicit scopes should override auto-detection."""
    custom = ["https://custom.scope/.default"]
    # Even for a cognitive services endpoint, explicit scopes take precedence
    scopes = get_credential_scopes(
        "https://myresource.cognitiveservices.azure.com/models", custom
    )
    assert scopes == custom


def test_explicit_scopes_override_for_foundry_endpoint():
    """Explicit scopes should override even for AI Foundry endpoints."""
    custom = ["https://custom.scope/.default"]
    scopes = get_credential_scopes(
        "https://myresource.services.ai.azure.com/models", custom
    )
    assert scopes == custom


def test_none_config_scopes_triggers_auto_detection():
    """None config_scopes should fall through to auto-detection."""
    scopes = get_credential_scopes(
        "https://myresource.cognitiveservices.azure.com/models", None
    )
    assert scopes == ["https://cognitiveservices.azure.com/.default"]


def test_empty_list_scopes_treated_as_falsy():
    """Empty list config_scopes should fall through to auto-detection."""
    scopes = get_credential_scopes(
        "https://myresource.cognitiveservices.azure.com/models", []
    )
    assert scopes == ["https://cognitiveservices.azure.com/.default"]
