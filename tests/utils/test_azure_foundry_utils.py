from mem0.utils.azure_foundry import get_credential_scopes, get_openai_base_url, is_openai_endpoint


# ── is_openai_endpoint tests ─────────────────────────────────────────────────

def test_is_openai_endpoint_cognitive_services():
    assert is_openai_endpoint("https://myresource.cognitiveservices.azure.com") is True


def test_is_openai_endpoint_cognitive_services_with_path():
    assert is_openai_endpoint("https://myresource.cognitiveservices.azure.com/openai/deployments/text-embedding-3-small") is True


def test_is_openai_endpoint_openai_azure():
    assert is_openai_endpoint("https://myresource.openai.azure.com") is True


def test_is_openai_endpoint_ai_foundry():
    assert is_openai_endpoint("https://myresource.services.ai.azure.com/models") is False


def test_is_openai_endpoint_other():
    assert is_openai_endpoint("https://api.openai.com/v1") is False


def test_is_openai_endpoint_case_insensitive():
    assert is_openai_endpoint("https://MyResource.CognitiveServices.Azure.Com/models") is True


# ── get_openai_base_url tests ────────────────────────────────────────────────

def test_get_openai_base_url_bare_host():
    """Base host without path should get /openai/v1/ appended."""
    assert get_openai_base_url("https://res.openai.azure.com") == "https://res.openai.azure.com/openai/v1/"


def test_get_openai_base_url_bare_host_trailing_slash():
    """Base host with trailing slash should get /openai/v1/ appended."""
    assert get_openai_base_url("https://res.openai.azure.com/") == "https://res.openai.azure.com/openai/v1/"


def test_get_openai_base_url_already_has_openai_v1():
    """URL that already has /openai/v1/ should be returned as-is."""
    assert get_openai_base_url("https://res.openai.azure.com/openai/v1/") == "https://res.openai.azure.com/openai/v1/"


def test_get_openai_base_url_openai_v1_no_trailing_slash():
    """URL with /openai/v1 (no trailing slash) should get trailing slash added."""
    assert get_openai_base_url("https://res.openai.azure.com/openai/v1") == "https://res.openai.azure.com/openai/v1/"


def test_get_openai_base_url_cognitive_services():
    """Cognitive Services endpoint should get /openai/v1/ appended."""
    assert get_openai_base_url("https://res.cognitiveservices.azure.com") == "https://res.cognitiveservices.azure.com/openai/v1/"


def test_get_openai_base_url_strips_other_paths():
    """Non /openai/v1 paths should be replaced with /openai/v1/."""
    assert get_openai_base_url("https://res.cognitiveservices.azure.com/models") == "https://res.cognitiveservices.azure.com/openai/v1/"


# ── get_credential_scopes tests ──────────────────────────────────────────────


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
