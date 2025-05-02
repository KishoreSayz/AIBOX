"""Example showing how to use LLMKit for chat completions."""

import asyncio
from typing import List

from llmkit import Client, Message, MessageRole
from llmkit.providers.openai import OpenAIProvider


async def main():
    """Run the example."""
    # Initialize the client
    client = Client()

    # Register the OpenAI provider
    openai_provider = OpenAIProvider()
    client.register_provider(openai_provider)

    # Create a chat conversation
    messages: List[Message] = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that speaks like a pirate.",
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

    # Print the response
    print(f"Assistant: {response.content}")

    # Close the client
    client.close()


if __name__ == "__main__":
    asyncio.run(main()) 