from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.azure_foundry_projects import AzureFoundryProjectsConfig
from mem0.llms.azure_foundry_projects import AzureFoundryProjectsLLM

MODEL = "gpt-4.1-nano-2025-04-14"
TEMPERATURE = 0.7
MAX_TOKENS = 100
TOP_P = 1.0
ENDPOINT = "https://test-resource.services.ai.azure.com/api/projects/my-project"


@pytest.fixture
def mock_project_client():
    """Mock AIProjectClient and the OpenAI client it returns."""
    with (
        patch("mem0.llms.azure_foundry_projects.AIProjectClient") as mock_proj_cls,
        patch("mem0.llms.azure_foundry_projects.DefaultAzureCredential") as mock_cred,
    ):
        mock_proj = Mock()
        mock_proj_cls.return_value = mock_proj
        mock_openai_client = Mock()
        mock_proj.get_openai_client.return_value = mock_openai_client
        mock_cred_instance = mock_cred.return_value
        yield {
            "project_cls": mock_proj_cls,
            "project_client": mock_proj,
            "openai_client": mock_openai_client,
            "credential_cls": mock_cred,
            "credential_instance": mock_cred_instance,
        }


def test_generate_response_without_tools(mock_project_client):
    config = AzureFoundryProjectsConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        endpoint=ENDPOINT,
    )
    llm = AzureFoundryProjectsLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I'm doing well, thank you for asking!"))]
    mock_project_client["openai_client"].chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages)

    mock_project_client["openai_client"].chat.completions.create.assert_called_once_with(
        model=MODEL, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P
    )
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_tools(mock_project_client):
    config = AzureFoundryProjectsConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        endpoint=ENDPOINT,
    )
    llm = AzureFoundryProjectsLLM(config)
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
    mock_project_client["openai_client"].chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    mock_project_client["openai_client"].chat.completions.create.assert_called_once_with(
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


def test_generate_response_with_response_format(mock_project_client):
    config = AzureFoundryProjectsConfig(
        model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        endpoint=ENDPOINT,
    )
    llm = AzureFoundryProjectsLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Return JSON."},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
    mock_project_client["openai_client"].chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages, response_format={"type": "json_object"})

    mock_project_client["openai_client"].chat.completions.create.assert_called_once_with(
        model=MODEL, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P,
        response_format={"type": "json_object"},
    )
    assert response == '{"key": "value"}'


def test_init_with_config(mock_project_client):
    config = AzureFoundryProjectsConfig(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        endpoint=ENDPOINT,
    )

    llm = AzureFoundryProjectsLLM(config)

    mock_project_client["credential_cls"].assert_called_once_with(
        managed_identity_client_id=None,
    )
    mock_project_client["project_cls"].assert_called_once_with(
        endpoint=ENDPOINT,
        credential=mock_project_client["credential_instance"],
    )
    mock_project_client["project_client"].get_openai_client.assert_called_once()
    assert llm.config.model == MODEL


def test_init_with_env_vars(monkeypatch, mock_project_client):
    monkeypatch.setenv("AZURE_AI_PROJECT_ENDPOINT", ENDPOINT)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)

    config = AzureFoundryProjectsConfig(model=None)
    llm = AzureFoundryProjectsLLM(config)

    mock_project_client["project_cls"].assert_called_once_with(
        endpoint=ENDPOINT,
        credential=mock_project_client["credential_instance"],
    )
    assert llm.config.model == "gpt-4.1-nano-2025-04-14"


def test_init_missing_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
    config = AzureFoundryProjectsConfig(model=MODEL)

    with (
        patch("mem0.llms.azure_foundry_projects.AIProjectClient"),
        patch("mem0.llms.azure_foundry_projects.DefaultAzureCredential"),
    ):
        with pytest.raises(ValueError, match="project endpoint is required"):
            AzureFoundryProjectsLLM(config)


def test_init_with_managed_identity_client_id(monkeypatch, mock_project_client):
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    client_id = "12345678-1234-1234-1234-123456789abc"
    config = AzureFoundryProjectsConfig(
        model=MODEL, endpoint=ENDPOINT, managed_identity_client_id=client_id,
    )

    AzureFoundryProjectsLLM(config)
    mock_project_client["credential_cls"].assert_called_once_with(
        managed_identity_client_id=client_id,
    )


def test_init_azure_client_id_env_var_fallback(monkeypatch, mock_project_client):
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id-1234")
    config = AzureFoundryProjectsConfig(model=MODEL, endpoint=ENDPOINT)

    AzureFoundryProjectsLLM(config)
    mock_project_client["credential_cls"].assert_called_once_with(
        managed_identity_client_id="env-client-id-1234",
    )


def test_init_config_client_id_takes_precedence_over_env_var(monkeypatch, mock_project_client):
    monkeypatch.setenv("AZURE_CLIENT_ID", "env-client-id")
    config_id = "config-client-id-5678"
    config = AzureFoundryProjectsConfig(
        model=MODEL, endpoint=ENDPOINT, managed_identity_client_id=config_id,
    )

    AzureFoundryProjectsLLM(config)
    mock_project_client["credential_cls"].assert_called_once_with(
        managed_identity_client_id=config_id,
    )
