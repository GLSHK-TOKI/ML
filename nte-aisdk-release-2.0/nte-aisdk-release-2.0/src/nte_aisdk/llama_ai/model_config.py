from dataclasses import dataclass

from ..model_config import BaseModelConfig


@dataclass
class LlamaModelConfig(BaseModelConfig):
    model_name: str