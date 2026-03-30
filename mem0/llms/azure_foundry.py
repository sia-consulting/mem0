import json
import os
from typing import Dict, List, Optional, Union

try:
    from azure.ai.inference import ChatCompletionsClient
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
    from azure.identity import DefaultAzureCredential
except ImportError:
    raise ImportError(
        "The 'azure-identity' library is required. "
        "Please install it using 'pip install azure-identity>=1.24.0'."
    )

from mem0.configs.llms.azure_foundry import AzureFoundryConfig
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json
from mem0.utils.azure_foundry import get_credential_scopes


class AzureFoundryLLM(LLMBase):
    """
    LLM provider for Azure AI Foundry using the azure-ai-inference SDK.

    Uses ChatCompletionsClient with endpoint + credential authentication.
    Supports both API key and managed identity (passwordless) authentication.
    When no API key is provided, falls back to DefaultAzureCredential for
    passwordless auth via managed identities, Azure CLI, etc.
    """

    def __init__(self, config: Optional[Union[BaseLlmConfig, AzureFoundryConfig, Dict]] = None):
        if config is None:
            config = AzureFoundryConfig()
        elif isinstance(config, dict):
            config = AzureFoundryConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, AzureFoundryConfig):
            config = AzureFoundryConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4.1-nano-2025-04-14"

        api_key = self.config.api_key or os.getenv("AZURE_FOUNDRY_API_KEY")
        endpoint = self.config.endpoint or os.getenv("AZURE_FOUNDRY_ENDPOINT")

        if not endpoint:
            raise ValueError(
                "Azure AI Foundry endpoint is required. "
                "Set it via config.endpoint or the AZURE_FOUNDRY_ENDPOINT environment variable."
            )

        # If the API key is not provided or is a placeholder, use DefaultAzureCredential
        # for passwordless authentication via managed identities, Azure CLI, etc.
        # To specify a user-assigned managed identity, set managed_identity_client_id in
        # config or the AZURE_CLIENT_ID env var (used as fallback).
        client_kwargs = {}

        # Allow overriding the API version sent to the Azure endpoint.
        # The azure-ai-inference SDK defaults to "2024-05-01-preview" which may
        # not be supported by all Azure AI Foundry endpoint configurations.
        api_version = getattr(self.config, "api_version", None) or os.getenv("AZURE_FOUNDRY_API_VERSION")
        if api_version:
            client_kwargs["api_version"] = api_version

        if api_key is None or api_key == "" or api_key == "your-api-key":
            managed_identity_client_id = (
                self.config.managed_identity_client_id or os.getenv("AZURE_CLIENT_ID")
            )
            credential = DefaultAzureCredential(
                managed_identity_client_id=managed_identity_client_id,
            )
            # The azure-ai-inference SDK defaults credential_scopes to
            # ["https://ml.azure.com/.default"], which is incorrect for
            # Cognitive Services endpoints (*.cognitiveservices.azure.com).
            # Auto-detect the correct scope from the endpoint URL, or use an
            # explicit override from config.
            scopes = get_credential_scopes(
                endpoint, getattr(self.config, "credential_scopes", None)
            )
            if scopes:
                client_kwargs["credential_scopes"] = scopes
        else:
            credential = AzureKeyCredential(api_key)

        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
            **client_kwargs,
        )

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using Azure AI Foundry.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to None.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional parameters.

        Returns:
            str or dict: The generated response.
        """
        user_prompt = messages[-1]["content"]
        user_prompt = user_prompt.replace("assistant", "ai")
        messages[-1]["content"] = user_prompt

        params = self._get_supported_params(messages=messages, **kwargs)

        params.update({
            "model": self.config.model,
            "messages": messages,
        })

        if response_format:
            # The azure-ai-inference SDK expects response_format as a string
            # literal ("text" or "json_object") or a JsonSchemaFormat object,
            # not a dict like {"type": "json_object"}.
            if isinstance(response_format, dict) and "type" in response_format:
                params["response_format"] = response_format["type"]
            else:
                params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.complete(**params)
        return self._parse_response(response, tools)
