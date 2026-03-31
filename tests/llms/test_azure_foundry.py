from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.azure_foundry import AzureFoundryConfig
from mem0.llms.azure_foundry import AzureFoundryLLM

MODEL = "gpt-4.1-nano-2025-04-14"
TEMPERATURE = 0.7
MAX_TOKENS = 100
TOP_P = 1.0
ENDPOINT = "https://test-resource.services.ai.azure.com/models"
API_KEY = "test-api-key"


@pytest.fixture
def mock_foundry_client():
    with patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls:
        mock_client = Mock()
        mock_client_cls.return_value = mock_client
        yield mock_client


def test_generate_response_without_tools(mock_foundry_client):
    config = AzureFoundryConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        api_key=API_KEY, endpoint=ENDPOINT,
    )
    llm = AzureFoundryLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I'm doing well, thank you for asking!"))]
    mock_foundry_client.complete.return_value = mock_response

    response = llm.generate_response(messages)

    mock_foundry_client.complete.assert_called_once_with(
        model=MODEL, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P
    )
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_dict_response_format(mock_foundry_client):
    """Dict response_format like {'type': 'json_object'} should be converted to string 'json_object'."""
    config = AzureFoundryConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        api_key=API_KEY, endpoint=ENDPOINT,
    )
    llm = AzureFoundryLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Return JSON."},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
    mock_foundry_client.complete.return_value = mock_response

    response = llm.generate_response(messages, response_format={"type": "json_object"})

    mock_foundry_client.complete.assert_called_once_with(
        model=MODEL, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        response_format="json_object",
    )
    assert response == '{"key": "value"}'


def test_generate_response_with_string_response_format(mock_foundry_client):
    """String response_format should be passed through unchanged."""
    config = AzureFoundryConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        api_key=API_KEY, endpoint=ENDPOINT,
    )
    llm = AzureFoundryLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Return JSON."},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
    mock_foundry_client.complete.return_value = mock_response

    response = llm.generate_response(messages, response_format="json_object")

    mock_foundry_client.complete.assert_called_once_with(
        model=MODEL, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        response_format="json_object",
    )
    assert response == '{"key": "value"}'


def test_generate_response_with_tools(mock_foundry_client):
    config = AzureFoundryConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        api_key=API_KEY, endpoint=ENDPOINT,
    )
    llm = AzureFoundryLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Add a new memory: Today is a sunny day."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add a memory",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string", "description": "Data to add to memory"}},
                    "required": ["data"],
                },
            },
        }
    ]

    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "I've added the memory for you."

    mock_tool_call = Mock()
    mock_tool_call.function.name = "add_memory"
    mock_tool_call.function.arguments = '{"data": "Today is a sunny day."}'

    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [Mock(message=mock_message)]
    mock_foundry_client.complete.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    mock_foundry_client.complete.assert_called_once_with(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        tools=tools,
        tool_choice="auto",
    )

    assert response["content"] == "I've added the memory for you."
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["name"] == "add_memory"
    assert response["tool_calls"][0]["arguments"] == {"data": "Today is a sunny day."}


def test_init_with_config(monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)

    config = AzureFoundryConfig(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        api_key=API_KEY,
        endpoint=ENDPOINT,
    )

    with patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls:
        with patch("mem0.llms.azure_foundry.AzureKeyCredential") as mock_cred:
            mock_cred.return_value = "mock-credential"
            llm = AzureFoundryLLM(config)
            mock_cred.assert_called_once_with(API_KEY)
            mock_client_cls.assert_called_once_with(
                endpoint=ENDPOINT,
                credential="mock-credential",
            )
            assert llm.config.model == MODEL


def test_init_with_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://env-endpoint.services.ai.azure.com/models")

    config = AzureFoundryConfig(model=None)

    with patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls:
        with patch("mem0.llms.azure_foundry.AzureKeyCredential") as mock_cred:
            mock_cred.return_value = "mock-credential"
            llm = AzureFoundryLLM(config)
            mock_cred.assert_called_once_with("env-key")
            mock_client_cls.assert_called_once_with(
                endpoint="https://env-endpoint.services.ai.azure.com/models",
                credential="mock-credential",
            )
            assert llm.config.model == "gpt-4.1-nano-2025-04-14"


def test_init_missing_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    config = AzureFoundryConfig(model=MODEL, api_key=API_KEY)

    with patch("mem0.llms.azure_foundry.ChatCompletionsClient"):
        with pytest.raises(ValueError, match="endpoint is required"):
            AzureFoundryLLM(config)


def test_init_missing_api_key_uses_default_credential(monkeypatch):
    """When no API key is provided, DefaultAzureCredential is used for managed identity auth."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    config = AzureFoundryConfig(model=MODEL, endpoint=ENDPOINT)

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        llm = AzureFoundryLLM(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=None)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )
        assert llm.config.model == MODEL


def test_init_with_placeholder_api_key_uses_default_credential(monkeypatch):
    """Placeholder API key should trigger DefaultAzureCredential."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    config = AzureFoundryConfig(model=MODEL, api_key="your-api-key", endpoint=ENDPOINT)

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryLLM(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=None)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )


def test_init_with_empty_api_key_uses_default_credential(monkeypatch):
    """Empty API key should trigger DefaultAzureCredential."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    config = AzureFoundryConfig(model=MODEL, api_key="", endpoint=ENDPOINT)

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryLLM(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=None)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )


def test_init_with_managed_identity_client_id(monkeypatch):
    """User-assigned managed identity client ID should be passed to DefaultAzureCredential."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    client_id = "12345678-1234-1234-1234-123456789abc"
    config = AzureFoundryConfig(
        model=MODEL, endpoint=ENDPOINT, managed_identity_client_id=client_id,
    )

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryLLM(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=client_id)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )


def test_init_azure_client_id_env_var_fallback(monkeypatch):
    """AZURE_CLIENT_ID env var should be used when managed_identity_client_id is not in config."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id-1234")
    config = AzureFoundryConfig(model=MODEL, endpoint=ENDPOINT)

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient"),
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        AzureFoundryLLM(config)
        mock_cred.assert_called_once_with(managed_identity_client_id="env-client-id-1234")


def test_init_config_client_id_takes_precedence_over_env_var(monkeypatch):
    """Config managed_identity_client_id should take precedence over AZURE_CLIENT_ID env var."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id")
    config_id = "config-client-id-5678"
    config = AzureFoundryConfig(
        model=MODEL, endpoint=ENDPOINT, managed_identity_client_id=config_id,
    )

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient"),
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        AzureFoundryLLM(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=config_id)


def test_init_cognitive_services_endpoint_uses_azure_openai(monkeypatch):
    """Cognitive Services endpoints should use OpenAI instead of ChatCompletionsClient."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    cog_endpoint = "https://myresource.cognitiveservices.azure.com/models"
    config = AzureFoundryConfig(model=MODEL, endpoint=cog_endpoint)

    with (
        patch("mem0.llms.azure_foundry.OpenAI") as mock_aoai,
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
        patch("mem0.llms.azure_foundry.get_bearer_token_provider") as mock_token,
    ):
        mock_token.return_value = Mock(return_value="mock-token")
        AzureFoundryLLM(config)
        mock_cred.assert_called_once_with(managed_identity_client_id=None)
        mock_token.assert_called_once_with(
            mock_cred.return_value,
            "https://cognitiveservices.azure.com/.default",
        )
        mock_aoai.assert_called_once_with(
            base_url="https://myresource.cognitiveservices.azure.com/openai/v1/",
            api_key="mock-token",
        )


def test_init_ai_foundry_endpoint_uses_sdk_default_scopes(monkeypatch):
    """AI Foundry endpoints (*.services.ai.azure.com) should use ai.azure.com scope."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    config = AzureFoundryConfig(model=MODEL, endpoint=ENDPOINT)

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryLLM(config)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=["https://ai.azure.com/.default"],
        )


def test_init_explicit_credential_scopes_override(monkeypatch):
    """Explicit credential_scopes in config should override auto-detection."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    custom_scopes = ["https://custom.scope/.default"]
    config = AzureFoundryConfig(
        model=MODEL, endpoint=ENDPOINT, credential_scopes=custom_scopes,
    )

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred_instance = mock_cred.return_value
        AzureFoundryLLM(config)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential=mock_cred_instance,
            credential_scopes=custom_scopes,
        )


def test_init_api_key_auth_cog_endpoint_uses_azure_openai(monkeypatch):
    """API key auth with Cognitive Services endpoint should use OpenAI."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    cog_endpoint = "https://myresource.cognitiveservices.azure.com/models"
    config = AzureFoundryConfig(
        model=MODEL, api_key=API_KEY, endpoint=cog_endpoint,
    )

    with patch("mem0.llms.azure_foundry.OpenAI") as mock_aoai:
        mock_aoai.return_value = Mock()
        AzureFoundryLLM(config)
        mock_aoai.assert_called_once_with(
            base_url="https://myresource.cognitiveservices.azure.com/openai/v1/",
            api_key=API_KEY,
        )


def test_init_with_api_version_from_config(monkeypatch):
    """api_version from config should be passed to ChatCompletionsClient."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    config = AzureFoundryConfig(
        model=MODEL, api_key=API_KEY, endpoint=ENDPOINT, api_version="2025-03-01-preview",
    )

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.AzureKeyCredential") as mock_cred,
    ):
        mock_cred.return_value = "mock-credential"
        AzureFoundryLLM(config)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential="mock-credential",
            api_version="2025-03-01-preview",
        )


def test_init_with_api_version_from_env_var(monkeypatch):
    """AZURE_FOUNDRY_API_VERSION env var should be passed to ChatCompletionsClient."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_FOUNDRY_API_VERSION", "2025-01-01-preview")
    config = AzureFoundryConfig(
        model=MODEL, api_key=API_KEY, endpoint=ENDPOINT,
    )

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.AzureKeyCredential") as mock_cred,
    ):
        mock_cred.return_value = "mock-credential"
        AzureFoundryLLM(config)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential="mock-credential",
            api_version="2025-01-01-preview",
        )


def test_init_api_version_config_takes_precedence_over_env_var(monkeypatch):
    """Config api_version should take precedence over AZURE_FOUNDRY_API_VERSION env var."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_FOUNDRY_API_VERSION", "env-version")
    config = AzureFoundryConfig(
        model=MODEL, api_key=API_KEY, endpoint=ENDPOINT, api_version="config-version",
    )

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.AzureKeyCredential") as mock_cred,
    ):
        mock_cred.return_value = "mock-credential"
        AzureFoundryLLM(config)
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential="mock-credential",
            api_version="config-version",
        )


def test_init_no_api_version_uses_sdk_default(monkeypatch):
    """When api_version is not set, it should not be passed to the client (SDK default used)."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    config = AzureFoundryConfig(
        model=MODEL, api_key=API_KEY, endpoint=ENDPOINT,
    )

    with (
        patch("mem0.llms.azure_foundry.ChatCompletionsClient") as mock_client_cls,
        patch("mem0.llms.azure_foundry.AzureKeyCredential") as mock_cred,
    ):
        mock_cred.return_value = "mock-credential"
        AzureFoundryLLM(config)
        # No api_version kwarg should be passed
        mock_client_cls.assert_called_once_with(
            endpoint=ENDPOINT,
            credential="mock-credential",
        )


# ── Azure OpenAI / Cognitive Services endpoint tests (OpenAI SDK) ────────────

COG_ENDPOINT = "https://myresource.cognitiveservices.azure.com"
OPENAI_AZURE_ENDPOINT = "https://myresource.openai.azure.com"


def test_cog_endpoint_generate_response_uses_openai_sdk(monkeypatch):
    """Cognitive Services endpoint should call client.chat.completions.create()."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    config = AzureFoundryConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        api_key=API_KEY, endpoint=COG_ENDPOINT,
    )

    with patch("mem0.llms.azure_foundry.OpenAI") as mock_aoai:
        mock_client = Mock()
        mock_aoai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="I'm doing well!"))]
        mock_client.chat.completions.create.return_value = mock_response

        llm = AzureFoundryLLM(config)
        messages = [
            {"role": "system", "content": "You are a helpful ai."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        response = llm.generate_response(messages)

        mock_client.chat.completions.create.assert_called_once()
        assert response == "I'm doing well!"
        assert llm._use_openai_sdk is True


def test_cog_endpoint_response_format_dict_passed_through(monkeypatch):
    """OpenAI SDK path should pass response_format dict as-is (no conversion)."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    config = AzureFoundryConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        api_key=API_KEY, endpoint=COG_ENDPOINT,
    )

    with patch("mem0.llms.azure_foundry.OpenAI") as mock_aoai:
        mock_client = Mock()
        mock_aoai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        llm = AzureFoundryLLM(config)
        messages = [{"role": "user", "content": "Return JSON."}]
        llm.generate_response(messages, response_format={"type": "json_object"})

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # Dict should be passed through unchanged (not converted to string)
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_openai_azure_endpoint_uses_azure_openai(monkeypatch):
    """*.openai.azure.com endpoints should also route through OpenAI."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    config = AzureFoundryConfig(
        model=MODEL, api_key=API_KEY, endpoint=OPENAI_AZURE_ENDPOINT,
    )

    with patch("mem0.llms.azure_foundry.OpenAI") as mock_aoai:
        mock_aoai.return_value = Mock()
        llm = AzureFoundryLLM(config)
        mock_aoai.assert_called_once_with(
            base_url="https://myresource.openai.azure.com/openai/v1/",
            api_key=API_KEY,
        )
        assert llm._use_openai_sdk is True


def test_cog_endpoint_api_version_from_env_var(monkeypatch):
    """AZURE_FOUNDRY_API_VERSION env var should NOT affect the OpenAI client."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_FOUNDRY_API_VERSION", "2023-05-15")
    config = AzureFoundryConfig(
        model=MODEL, api_key=API_KEY, endpoint=COG_ENDPOINT,
    )

    with patch("mem0.llms.azure_foundry.OpenAI") as mock_aoai:
        mock_aoai.return_value = Mock()
        AzureFoundryLLM(config)
        mock_aoai.assert_called_once_with(
            base_url="https://myresource.cognitiveservices.azure.com/openai/v1/",
            api_key=API_KEY,
        )


def test_cog_endpoint_strips_path_from_azure_endpoint(monkeypatch):
    """OpenAI base_url should only contain scheme + host + /openai/v1/."""
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_VERSION", raising=False)
    endpoint_with_path = "https://myresource.cognitiveservices.azure.com/openai/v1/models"
    config = AzureFoundryConfig(
        model=MODEL, api_key=API_KEY, endpoint=endpoint_with_path,
    )

    with patch("mem0.llms.azure_foundry.OpenAI") as mock_aoai:
        mock_aoai.return_value = Mock()
        AzureFoundryLLM(config)
        mock_aoai.assert_called_once_with(
            base_url="https://myresource.cognitiveservices.azure.com/openai/v1/",
            api_key=API_KEY,
        )
