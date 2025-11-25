from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseModelConfig(ABC):
    @abstractmethod
    def __init__(self):
        pass