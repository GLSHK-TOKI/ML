from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseInstanceConfig(ABC):
    @abstractmethod
    def __init__(self):
        pass