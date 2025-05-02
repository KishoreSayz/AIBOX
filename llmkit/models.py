"""Core data models for the LLMKit library."""

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Enum for message roles in chat conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A message in a chat conversation."""

    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List["ToolCall"]] = None
    tool_call_id: Optional[str] = None

    class Config:
        """Pydantic config."""

        use_enum_values = True


class ToolCall(BaseModel):
    """A tool call made by the model."""

    id: str
    type: str = "function"
    function: "FunctionCall"


class FunctionCall(BaseModel):
    """A function call made by the model."""

    name: str
    arguments: str


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    name: str
    provider: str
    capabilities: List[str] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    pricing: Optional[Dict[str, float]] = None


class ProviderInfo(BaseModel):
    """Information about a provider."""

    id: str
    name: str
    description: str
    models: List[ModelInfo] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    tools: Optional[List["Tool"]] = None
    tool_choice: Optional[Union[str, Dict[str, str]]] = None


class Tool(BaseModel):
    """A tool that can be used by the model."""

    type: str = "function"
    function: "Function"


class Function(BaseModel):
    """A function that can be called by the model."""

    name: str
    description: str
    parameters: Dict[str, object]


# Update forward references
Message.model_rebuild()
ToolCall.model_rebuild()
FunctionCall.model_rebuild()
ChatCompletionRequest.model_rebuild()
Tool.model_rebuild()
Function.model_rebuild() 