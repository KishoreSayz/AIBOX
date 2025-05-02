"""Base provider interface for LLMKit."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from llmkit.models import (
    ChatCompletionRequest,
    Message,
    ModelInfo,
    ProviderInfo,
)


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Get provider information."""
        pass

    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List all available models for this provider."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Message:
        """Generate a chat completion."""
        pass

    @abstractmethod
    async def embeddings(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings for the input text."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources used by the provider."""
        pass

    def __enter__(self) -> "BaseProvider":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        self.close() 