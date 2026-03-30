import json
import os
from typing import Dict, List, Optional, Union

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from mem0.configs.llms.azure_foundry_projects import AzureFoundryProjectsConfig
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


class AzureFoundryProjectsLLM(LLMBase):
    """
    LLM provider for Azure AI Foundry Projects using the azure-ai-projects SDK.

    Uses AIProjectClient to obtain a properly authenticated OpenAI client via
    get_openai_client(), then uses the standard OpenAI chat.completions API.
    This is the recommended approach for Azure AI Foundry project endpoints
    (https://<resource>.services.ai.azure.com/api/projects/<project>).

    Authentication is handled via DefaultAzureCredential (Entra ID only).
    """

    def __init__(self, config: Optional[Union[BaseLlmConfig, AzureFoundryProjectsConfig, Dict]] = None):
        if config is None:
            config = AzureFoundryProjectsConfig()
        elif isinstance(config, dict):
            config = AzureFoundryProjectsConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, AzureFoundryProjectsConfig):
            config = AzureFoundryProjectsConfig(
                model=config.model,
                temperature=config.temperature,
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

        endpoint = self.config.endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT")

        if not endpoint:
            raise ValueError(
                "Azure AI Foundry project endpoint is required. "
                "Set it via config.endpoint or the AZURE_AI_PROJECT_ENDPOINT environment variable."
            )

        managed_identity_client_id = (
            self.config.managed_identity_client_id or os.getenv("AZURE_CLIENT_ID")
        )
        credential = DefaultAzureCredential(
            managed_identity_client_id=managed_identity_client_id,
        )

        project_client = AIProjectClient(
            endpoint=endpoint,
            credential=credential,
        )

        self.client = project_client.get_openai_client()

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
        Generate a response based on the given messages using Azure AI Foundry Projects.

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
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)
