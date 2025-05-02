"""A modern, production-ready Python library for LLM interactions."""

from typing import Dict, Type

from llmkit.client import Client
from llmkit.exceptions import (
    LLMError,
    RateLimitError,
    InvalidRequestError,
    ProviderError,
)
from llmkit.models import Message, MessageRole, ModelInfo, ProviderInfo
from llmkit.providers.base import BaseProvider

__version__ = "0.1.0"

__all__ = [
    "Client",
    "BaseProvider",
    "Message",
    "MessageRole",
    "ModelInfo",
    "ProviderInfo",
    "LLMError",
    "RateLimitError",
    "InvalidRequestError",
    "ProviderError",
] 