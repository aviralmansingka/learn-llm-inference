import time
import uuid

from typing import Optional
from pydantic import BaseModel, Field

from scratch_llm.runtime.models.engine import EngineRequest, EngineResponse, Message

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class ChatCompletionChoice(BaseModel):
    """Represents a single choice in a chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageStats(BaseModel):
    """Represents token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request.

    This schema captures the response in a structured format
    and is optimized for FastAPI serialization.
    """
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageStats
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    created: int = Field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion"

def convert_to_engine(openai_request) -> EngineRequest:
    """Convert OpenAI ChatCompletionRequest to internal EngineRequest.

    Args:
        openai_request: The OpenAI-formatted request from the API

    Returns:
        Internal EngineRequest object
    """
    internal_messages = [
        Message(role=msg.role, content=msg.content)
        for msg in openai_request.messages
    ]

    return EngineRequest(
        model=openai_request.model,
        messages=internal_messages,
        temperature=openai_request.temperature,
        max_tokens=openai_request.max_tokens,
    )

def convert_from_engine(engine_response: EngineResponse) -> ChatCompletionResponse:
    """Convert OpenAI ChatCompletionRequest to internal EngineRequest.

    Args:
        openai_request: The OpenAI-formatted request from the API

    Returns:
        Internal EngineRequest object
    """
    response = ChatCompletionResponse(
        model=engine_response.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=engine_response.prompt_response)
            )
        ],
        usage=UsageStats(
            prompt_tokens=engine_response.usage_stats.prompt_tokens,
            completion_tokens=engine_response.usage_stats.completion_tokens,
            total_tokens=engine_response.usage_stats.total_tokens,
        )
    )
    return response
