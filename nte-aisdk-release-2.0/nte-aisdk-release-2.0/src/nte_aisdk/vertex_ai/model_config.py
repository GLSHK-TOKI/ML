from dataclasses import dataclass

from ..model_config import BaseModelConfig


@dataclass
class VertexAIModelConfig(BaseModelConfig):
    model_name: str

@dataclass
class VertexAIReasoningModelConfig(VertexAIModelConfig):
    thinking_budget: int = 0