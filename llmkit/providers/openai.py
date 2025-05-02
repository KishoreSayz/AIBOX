"""OpenAI provider implementation."""

import os
from typing import List, Optional, Union

import openai
from openai import AsyncOpenAI

from llmkit.exceptions import (
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from llmkit.models import (
    ChatCompletionRequest,
    Message,
    MessageRole,
    ModelInfo,
    ProviderInfo,
)
from llmkit.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        """Initialize the OpenAI provider."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise InvalidRequestError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key."
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            id="openai",
            name="OpenAI",
            description="OpenAI's API for accessing GPT models",
            models=[
                ModelInfo(
                    id="gpt-4",
                    name="GPT-4",
                    provider="openai",
                    capabilities=["chat", "tools"],
                    max_tokens=8192,
                    context_window=8192,
                ),
                ModelInfo(
                    id="gpt-3.5-turbo",
                    name="GPT-3.5 Turbo",
                    provider="openai",
                    capabilities=["chat", "tools"],
                    max_tokens=4096,
                    context_window=4096,
                ),
                ModelInfo(
                    id="text-embedding-ada-002",
                    name="Ada Embeddings",
                    provider="openai",
                    capabilities=["embeddings"],
                ),
            ],
            capabilities=["chat", "embeddings", "tools"],
        )

    async def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        return self.info.models

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Message:
        """Generate a chat completion."""
        try:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=[
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        **({"name": msg.name} if msg.name else {}),
                        **(
                            {"tool_calls": [tc.dict() for tc in msg.tool_calls]}
                            if msg.tool_calls
                            else {}
                        ),
                        **(
                            {"tool_call_id": msg.tool_call_id}
                            if msg.tool_call_id
                            else {}
                        ),
                    }
                    for msg in request.messages
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                **kwargs,
            )

            choice = response.choices[0]
            message = choice.message

            return Message(
                role=MessageRole(message.role),
                content=message.content or "",
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls or []
                ],
            )

        except openai.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider="openai",
                retry_after=int(e.response.headers.get("retry-after", 0)),
            ) from e
        except openai.APIError as e:
            raise ProviderError(
                str(e),
                provider="openai",
                status_code=e.status_code,
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Error generating chat completion: {str(e)}",
                provider="openai",
            ) from e

    async def embeddings(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings for the input text."""
        try:
            response = await self._client.embeddings.create(
                model=model or "text-embedding-ada-002",
                input=text,
                **kwargs,
            )
            return [embedding.embedding for embedding in response.data]

        except openai.RateLimitError as e:
            raise RateLimitError(
                str(e),
                provider="openai",
                retry_after=int(e.response.headers.get("retry-after", 0)),
            ) from e
        except openai.APIError as e:
            raise ProviderError(
                str(e),
                provider="openai",
                status_code=e.status_code,
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Error generating embeddings: {str(e)}",
                provider="openai",
            ) from e

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self._client.close() 