"""Custom exceptions for LLMKit."""

from typing import Optional


class LLMError(Exception):
    """Base exception for all LLMKit errors."""

    def __init__(self, message: str, provider: Optional[str] = None):
        """Initialize the exception."""
        self.message = message
        self.provider = provider
        super().__init__(f"{f'[{provider}] ' if provider else ''}{message}")


class RateLimitError(LLMError):
    """Raised when a rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        """Initialize the exception."""
        super().__init__(message, provider)
        self.retry_after = retry_after


class InvalidRequestError(LLMError):
    """Raised when an invalid request is made."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        code: Optional[str] = None,
    ):
        """Initialize the exception."""
        super().__init__(message, provider)
        self.code = code


class ProviderError(LLMError):
    """Raised when a provider-specific error occurs."""

    def __init__(
        self,
        message: str,
        provider: str,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        """Initialize the exception."""
        super().__init__(message, provider)
        self.code = code
        self.status_code = status_code 