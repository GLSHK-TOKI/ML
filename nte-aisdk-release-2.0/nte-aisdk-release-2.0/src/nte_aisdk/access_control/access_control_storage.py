from abc import ABC, abstractmethod


class AccessControlStorage(ABC):
    @abstractmethod
    def list_collections(self) -> list:
        """Retrieve a list collection from the access control storage."""