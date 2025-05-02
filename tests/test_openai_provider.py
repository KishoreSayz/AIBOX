"""Tests for the OpenAI provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llmkit.models import Message, MessageRole
from llmkit.providers.openai import OpenAIProvider


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("llmkit.providers.openai.AsyncOpenAI") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_chat_completion(mock_openai):
    """Test chat completion."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                role="assistant",
                content="Hello!",
                tool_calls=[],
            )
        )
    ]
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    # Initialize provider
    provider = OpenAIProvider(api_key="test-key")

    # Test chat completion
    response = await provider.chat_completion(
        messages=[
            Message(
                role=MessageRole.USER,
                content="Hello",
            )
        ],
        model="gpt-3.5-turbo",
    )

    # Verify response
    assert response.role == MessageRole.ASSISTANT
    assert response.content == "Hello!"
    assert response.tool_calls == []

    # Verify API call
    mock_openai.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_embeddings(mock_openai):
    """Test embeddings."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_openai.embeddings.create = AsyncMock(return_value=mock_response)

    # Initialize provider
    provider = OpenAIProvider(api_key="test-key")

    # Test embeddings
    embeddings = await provider.embeddings(
        text=["Hello", "World"],
        model="text-embedding-ada-002",
    )

    # Verify response
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]

    # Verify API call
    mock_openai.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_close(mock_openai):
    """Test close method."""
    # Initialize provider
    provider = OpenAIProvider(api_key="test-key")

    # Test close
    await provider.close()

    # Verify close was called
    mock_openai.close.assert_called_once() 