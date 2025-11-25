from abc import ABC, abstractmethod


class BaseExampleStore(ABC):
    """Base class for all example stores."""

    @property
    @abstractmethod
    def uses_environments(self) -> bool:
        """Whether this example store uses environments."""
        pass