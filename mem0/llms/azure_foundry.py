import json
import os
from typing import Dict, List, Optional, Union

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from mem0.configs.llms.azure_foundry import AzureFoundryConfig
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


class AzureFoundryLLM(LLMBase):
    """
    LLM provider for Azure AI Foundry using the azure-ai-inference SDK.

    Uses ChatCompletionsClient with endpoint + API key authentication,
    which is the recommended approach for Azure AI Foundry deployments.
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

        if not api_key:
            raise ValueError(
                "Azure AI Foundry API key is required. "
                "Set it via config.api_key or the AZURE_FOUNDRY_API_KEY environment variable."
            )

        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
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
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.complete(**params)
        return self._parse_response(response, tools)
