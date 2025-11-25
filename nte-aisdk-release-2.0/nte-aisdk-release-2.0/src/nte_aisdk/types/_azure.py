from pydantic import Field
from typing_extensions import TypedDict

from ._common import BaseModel


class AzureInstanceConfig(BaseModel):
    """Configuration for an Azure AI instance. When user need to use multiple Azure AI instances."""
    azure_endpoint: str = Field(
        description="The endpoint for the Azure AI service, e.g., 'https://<your-resource-name>.openai.azure.com/, https://<your-resource-name>.services.ai.azure.com'"
    )
    api_key: str = Field(
        description="The API key for the Azure AI service."
    )
    api_version: str = Field(
        description="The API version for the Azure AI service. You can find the latest version in https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs."
    )

class AzureInstanceConfigDict(TypedDict):
    """Configuration for an Azure AI instance. When user need to use multiple Azure AI instances."""
    azure_endpoint: str
    """The endpoint for the Azure AI service, e.g., 'https://<your-resource-name>.openai.azure.com/, https://<your-resource-name>.services.ai.azure.com'"""

    api_key: str
    """The API key for the Azure AI service."""

    api_version: str
    """The API version for the Azure AI service. You can find the latest version in https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs."""

AzureInstanceConfigOrDict = AzureInstanceConfig | AzureInstanceConfigDict

class AzureInstanceConfigWithoutAPIVersion(BaseModel):
    """Legacy style configuration of AI SDK for an Azure AI instance without api_version."""
    azure_endpoint: str = Field(
        description="The endpoint for the Azure AI service, e.g., 'https://<your-resource-name>.openai.azure.com/, https://<your-resource-name>.services.ai.azure.com'"
    )
    api_key: str = Field(
        description="The API key for the Azure AI service."
    )

class AzureInstanceConfigWithoutAPIVersionDict(TypedDict):
    """Legacy style configuration of AI SDK for an Azure AI instance without api_version."""
    azure_endpoint: str
    """The endpoint for the Azure AI service, e.g., 'https://<your-resource-name>.openai.azure.com/, https://<your-resource-name>.services.ai.azure.com'"""

    api_key: str
    """The API key for the Azure AI service."""

AzureInstanceConfigWithoutAPIVersionOrDict = AzureInstanceConfigWithoutAPIVersion | AzureInstanceConfigWithoutAPIVersionDict

class AzureModelConfigWithAPIVersion(BaseModel):
    """Legacy style configuration for an Azure AI model with api_version."""
    azure_deployment: str = Field(
        description="The deployment name of the Azure AI model."
    )
    api_version: str = Field(
        description="The API version for the Azure AI service. You can find the latest version in https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#api-specs."
    )
