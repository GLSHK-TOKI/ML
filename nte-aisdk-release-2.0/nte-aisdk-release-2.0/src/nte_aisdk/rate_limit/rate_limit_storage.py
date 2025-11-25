from abc import ABC, abstractmethod


class RateLimitStorage(ABC):
    @abstractmethod
    def get(self, key: str, user_id: str):
        """Retrieve a record from the rate limit storage."""

    @abstractmethod
    def put(
        self,
        key: str,
        user_id: str,
        tokens: int,
        timestamp: float,
    ):
        """Create a record in the rate limit storage."""

    @abstractmethod
    def update(
        self,
        doc_id: str,
        tokens: int,
        timestamp: float | None = None
    ):
        """Update the last reset timestamp or tokens for the rate limit record."""