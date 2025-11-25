from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import openai
from openai import AzureOpenAI

from nte_aisdk import errors, types
from nte_aisdk.text_embedding_model import TextEmbeddingModel

if TYPE_CHECKING:
    from nte_aisdk.providers.azure import AzureProvider

logger = logging.getLogger(__name__)

class AzureEmbeddingModel(TextEmbeddingModel):
    provider: AzureProvider
    models: list[AzureOpenAI]

    def __init__(
        self,
        azure_deployment: str,
        model_name: str,
        provider: AzureProvider,
    ):
        """Initializes the Azure embedding model.

        Args:
            azure_deployment (str): The deployment name of the embedding model on Azure.
            model_name (str): The name of the model, e.g. "text-embedding-3-small". You can find the model name of your deployment in the Azure AI Foundry.
            provider (AzureProvider): The Azure provider that manages the connection to Azure AI services.
        """
        super().__init__(model_name, provider)
        self.azure_deployment = azure_deployment

        # Check conflict if user provides both provider configuration and instances.
        if ((provider.azure_endpoint or provider.api_key or provider.api_version) and
                provider.instances):
            msg = "You cannot provide both provider configuration and instances. Please provide either one."
            raise errors.InvalidArgumentError(msg)

        instance_configs: list[types.AzureInstanceConfig] = []

        # Create a single instance config, with the provider configuration provided.
        if provider.azure_endpoint and provider.api_key and provider.api_version:
            instance_configs = [
                types.AzureInstanceConfig(
                    azure_endpoint=provider.azure_endpoint,
                    api_key=provider.api_key,
                    api_version=provider.api_version
                )
            ]

        # If the provider has multiple instances, we will append the models for each instance.
        if provider.instances:
            instance_configs = [
                types.AzureInstanceConfig.model_validate(instance_config)
                if isinstance(instance_config, dict) else instance_config
                for instance_config in provider.instances
            ]

        # Create a list of AzureChatOpenAI models for each instance configuration, no matter it's single or multiple instances.
        self.models = [
            self._create_model_instance(instance_config)
            for instance_config in instance_configs
        ]

    @property
    def model(self) -> AzureOpenAI:
        return super().model

    def do_embed(
        self,
        text: str
    ) -> types.EmbedResponse:
        """Create embedding using the Azure embedding model."""
        if not text:
            msg = "Text is empty."
            raise errors.InvalidArgumentError(msg)

        if not self.model:
            msg = "Model is not initialized. Please check the Azure provider configuration."
            raise errors.SDKError(msg)

        response = handle_error(
            self.model.embeddings.create,
            input=text,
            model=self.azure_deployment
        )
        return types.EmbedResponse(
            embedding=response.data[0].embedding,
            metadata=types.EmbedResponseMetadata(
                model_id=response.model,
                usage=types.EmbedResponseUsage(
                    tokens=response.usage.total_tokens,
                )
            )
        )

    def _create_model_instance(self, instance_config: types.AzureInstanceConfig) -> AzureOpenAI:
        return AzureOpenAI(
            azure_endpoint=instance_config.azure_endpoint,
            api_key=instance_config.api_key,
            api_version=instance_config.api_version,
        )

def handle_error(operation, *args, **kwargs):
    """Handles errors raised during Azure embedding model operations with official OpenAI python SDK."""
    try:
        return operation(*args, **kwargs)
    except openai.APIStatusError as e:
        msg = f"API error with status code when calling Azure OpenAI API. {e.message}"
        logger.exception(msg)
        raise errors.APIError(msg, e.status_code) from e
    except openai.APIError as e:
        msg = f"API error when calling Azure OpenAI API. {e.message}"
        logger.exception(msg)
        raise errors.APIError(msg) from e
    except openai.OpenAIError as e:
        msg = f"Error from OpenAI SDK when calling Azure OpenAI API. {e!s}"
        logger.exception(msg)
        raise errors.APIError(msg) from e
    except Exception as e:
        msg = "Unexpected error when calling Azure OpenAI API."
        logger.exception(msg)
        raise errors.APIError(msg) from e
