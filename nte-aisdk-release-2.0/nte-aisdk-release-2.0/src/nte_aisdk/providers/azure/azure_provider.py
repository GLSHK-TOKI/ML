from nte_aisdk import types
from nte_aisdk.exception import ensure_arguments
from nte_aisdk.providers.azure import AzureEmbeddingModel, AzureLanguageModel


class AzureProvider:
    """AzureProvider offers methods for creating models to interact with Azure AI and OpenAI services."""
    azure_endpoint: str | None
    api_key: str | None
    api_version: str | None
    instances: list[types.AzureInstanceConfigOrDict] | None

    def __init__(
        self,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        instances: list[types.AzureInstanceConfigOrDict] | None = None,
    ):
        """Initializes the Azure provider

        Args:
            azure_endpoint (str | None): The endpoint for the Azure AI service, e.g., 'https://<your-resource-name>.openai.azure.com/, https://<your-resource-name>.services.ai.azure.com'.
            api_key (str | None): The API key for the Azure AI service.
            api_version (str | None): The API version for the Azure AI service. You can find the latest version in https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs.
            instances (list[types.AzureInstanceConfig] | None): A list of AzureInstanceConfig objects to configure multiple Azure AI instances.
                For backward compatibility, user can define multiple instances for load balancing. If instances is not provided, the provider will use the single instance configuration.
        """
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.instances = instances

    @ensure_arguments
    def create_language_model(
        self,
        azure_deployment: str,
        model_name: str,
    ):
        """Creates an Azure language model

        Args:
            azure_deployment (str): The deployment name of the Azure AI model.
            model_name (str): The name of the model, e.g. "gpt-4o", "DeepSeek-R1". You can find the model name of your deployment in the Azure AI Foundry.
        """
        return AzureLanguageModel(azure_deployment, model_name, self)

    @ensure_arguments
    def create_embedding_model(
        self,
        azure_deployment: str,
        model_name: str,
    ):
        """Creates an Azure embedding model

        Args:
            azure_deployment (str): The deployment name of the Azure AI model.
            model_name (str): The name of the model, e.g. "text-embedding-3-small". You can find the model name of your deployment in the Azure AI Foundry.
        """
        return AzureEmbeddingModel(azure_deployment, model_name, self)
