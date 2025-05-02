# LLMKit

A modern, production-ready Python library for LLM interactions.

## Features

- ✅ Unified interface to multiple LLM providers (OpenAI, Anthropic, Google, HuggingFace, etc.)
- ✅ Modular architecture for chat, embeddings, vision, audio, fine-tuning
- ✅ Tool calling abstraction (manual + automatic) that works seamlessly across providers
- ✅ Dynamic model discovery: ability to list available models per provider
- ✅ Built-in retry logic, rate-limit handling, and exponential backoff
- ✅ Unified, typed exceptions (e.g., RateLimitError, InvalidRequestError)
- ✅ Strong type hints and PEP-8 compliance
- ✅ Plugin system for easily adding new providers via structured templates
- ✅ Lazy-loading provider SDKs to avoid bloated dependencies
- ✅ Optional telemetry, tracing, and metrics hooks (OpenTelemetry + Prometheus)
- ✅ CLI + FastAPI layer to expose library as an LLM API
- ✅ Prebuilt support for OpenAI-style system/user/assistant messages

## Installation

```bash
pip install llmkit
```

For specific providers:

```bash
pip install "llmkit[openai]"  # OpenAI support
pip install "llmkit[anthropic]"  # Anthropic support
pip install "llmkit[google]"  # Google support
pip install "llmkit[huggingface]"  # HuggingFace support
pip install "llmkit[all]"  # All providers
```

## Quick Start

```python
import asyncio
from llmkit import Client, Message, MessageRole
from llmkit.providers.openai import OpenAIProvider

async def main():
    # Initialize the client
    client = Client()

    # Register the OpenAI provider
    openai_provider = OpenAIProvider()
    client.register_provider(openai_provider)

    # Create a chat conversation
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant.",
        ),
        Message(
            role=MessageRole.USER,
            content="What's the weather like today?",
        ),
    ]

    # Generate a completion
    response = await client.chat_completion(
        messages=messages,
        model="gpt-3.5-turbo",
    )

    print(f"Assistant: {response.content}")

    # Close the client
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Provider Support

### OpenAI

```python
from llmkit.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-api-key",  # Optional, defaults to OPENAI_API_KEY env var
    organization="your-org",  # Optional
)
```

### Anthropic (Coming Soon)

```python
from llmkit.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    api_key="your-api-key",  # Optional, defaults to ANTHROPIC_API_KEY env var
)
```

### Google (Coming Soon)

```python
from llmkit.providers.google import GoogleProvider

provider = GoogleProvider(
    api_key="your-api-key",  # Optional, defaults to GOOGLE_API_KEY env var
)
```

### HuggingFace (Coming Soon)

```python
from llmkit.providers.huggingface import HuggingFaceProvider

provider = HuggingFaceProvider(
    api_key="your-api-key",  # Optional, defaults to HUGGINGFACE_API_KEY env var
)
```

## Error Handling

The library provides unified error handling across all providers:

```python
from llmkit.exceptions import RateLimitError, ProviderError

try:
    response = await client.chat_completion(...)
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after} seconds")
except ProviderError as e:
    print(f"Provider error: {e.message} (Status: {e.status_code})")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 