"""Main client class for LLMKit."""

import importlib
import inspect
import logging
from typing import Dict, List, Optional, Type, Union

from llmkit.exceptions import InvalidRequestError, ProviderError
from llmkit.models import (
    ChatCompletionRequest,
    Message,
    ModelInfo,
    ProviderInfo,
)
from llmkit.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class Client:
    """Main client class for LLMKit."""

    def __init__(self):
        """Initialize the client."""
        self._providers: Dict[str, BaseProvider] = {}
        self._models: Dict[str, ModelInfo] = {}
        self._provider_classes: Dict[str, Type[BaseProvider]] = {}

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a provider instance."""
        provider_info = provider.info
        self._providers[provider_info.id] = provider
        for model in provider_info.models:
            self._models[model.id] = model

    def register_provider_class(
        self,
        provider_id: str,
        provider_class: Type[BaseProvider],
    ) -> None:
        """Register a provider class for lazy loading."""
        self._provider_classes[provider_id] = provider_class

    def load_provider(self, provider_id: str, **kwargs) -> BaseProvider:
        """Load a provider by ID."""
        if provider_id in self._providers:
            return self._providers[provider_id]

        if provider_id not in self._provider_classes:
            raise InvalidRequestError(
                f"Provider '{provider_id}' not found. "
                "Make sure the provider package is installed and registered."
            )

        provider_class = self._provider_classes[provider_id]
        provider = provider_class(**kwargs)
        self.register_provider(provider)
        return provider

    def get_provider(self, provider_id: str) -> BaseProvider:
        """Get a provider by ID."""
        if provider_id not in self._providers:
            raise InvalidRequestError(f"Provider '{provider_id}' not found")
        return self._providers[provider_id]

    def list_providers(self) -> List[ProviderInfo]:
        """List all registered providers."""
        return [provider.info for provider in self._providers.values()]

    def list_models(
        self,
        provider_id: Optional[str] = None,
    ) -> List[ModelInfo]:
        """List all available models."""
        if provider_id:
            if provider_id not in self._providers:
                raise InvalidRequestError(f"Provider '{provider_id}' not found")
            return self._providers[provider_id].info.models
        return list(self._models.values())

    def get_model(self, model_id: str) -> ModelInfo:
        """Get model information by ID."""
        if model_id not in self._models:
            raise InvalidRequestError(f"Model '{model_id}' not found")
        return self._models[model_id]

    async def chat_completion(
        self,
        messages: List[Message],
        model: str,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Message:
        """Generate a chat completion."""
        model_info = self.get_model(model)
        provider_id = provider or model_info.provider

        try:
            provider_instance = self.get_provider(provider_id)
            request = ChatCompletionRequest(
                messages=messages,
                model=model,
                **kwargs,
            )
            return await provider_instance.chat_completion(request)
        except Exception as e:
            raise ProviderError(
                f"Error generating chat completion: {str(e)}",
                provider=provider_id,
            ) from e

    async def embeddings(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings for the input text."""
        if not model and not provider:
            raise InvalidRequestError(
                "Either model or provider must be specified"
            )

        if model:
            model_info = self.get_model(model)
            provider_id = provider or model_info.provider
        else:
            provider_id = provider

        try:
            provider_instance = self.get_provider(provider_id)
            return await provider_instance.embeddings(
                text=text,
                model=model,
                **kwargs,
            )
        except Exception as e:
            raise ProviderError(
                f"Error generating embeddings: {str(e)}",
                provider=provider_id,
            ) from e

    def close(self) -> None:
        """Close all providers."""
        for provider in self._providers.values():
            provider.close()
        self._providers.clear()
        self._models.clear()

    def __enter__(self) -> "Client":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        self.close() 