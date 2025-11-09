from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str
    content: str


@dataclass
class EngineRequest:
    """Internal representation of a chat completion request.

    This is independent of the OpenAI schema and can be used
    throughout the application for internal processing.
    """
    model: str
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None

@dataclass
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class EngineResponse:
    """Internal representation of a chat completion response

    This is independent of the OpenAI schema and can be used
    throughout the application for internal processing.
    """
    model: str
    prompt_response: str
    usage_stats: UsageStats
