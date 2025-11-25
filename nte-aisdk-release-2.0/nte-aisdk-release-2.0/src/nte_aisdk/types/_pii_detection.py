from pydantic import Field
from typing_extensions import TypedDict

from ._common import BaseModel


class PIICategory(BaseModel):
    """Represents a PII category with its name and definition.

    Args:
        name (str): The name of the PII category (e.g., "Email Address")
        definition (str | None): The definition or pattern description for detecting this PII type
    """

    name: str = Field(
        description="The name of the PII category"
    )
    definition: str | None = Field(
        default=None,
        description="The definition or pattern description for detecting this PII type"
    )

class PIICategoryDict(TypedDict):
    """Represents a PII category with its name and definition.

    Args:
        name (str): The name of the PII category (e.g., "Email Address")
        definition (str | None): The definition or pattern description for detecting this PII type
    """

    name: str
    """The name of the PII category"""

    definition: str | None
    """The definition or pattern description for detecting this PII type"""

class PIIDetectionConfig(BaseModel):
    """Configuration for PII detection.

    Args:
        categories (list[PIICategory]): List of PII categories to detect
    """

    categories: list[PIICategory] = Field(
        description="List of PII categories to detect"
    )

class PIIDetectionConfigDict(TypedDict):
    """Configuration for PII detection.

    Args:
        categories (list[PIICategory]): List of PII categories to detect
    """

    categories: list[PIICategoryDict]
    """List of PII categories to detect"""

class PIIItem(BaseModel):
    """A detected PII item with its category, text, and optional location."""

    entity: str = Field(
        description="The PII category/entity type"
    )
    text: str = Field(
        description="The detected PII text"
    )
    location: int | None = Field(
        default=None,
        description="The character position where the PII was found in the original text"
    )

PIIDetectionConfigOrDict = PIIDetectionConfig | PIIDetectionConfigDict
PIICategoryOrDict = PIICategory | PIICategoryDict