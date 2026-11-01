"""Model clients.

The agent depends on one method, `chat(messages, tools)`, so swapping models is a config change. `Ollama`
talks to a local Ollama server over its HTTP API. `Scripted` replays canned responses for tests.
"""

import time
from dataclasses import dataclass, field

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class Ollama:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", timeout: float = 300):
        self.name = model
        self.client = httpx.Client(base_url=host, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        started = time.perf_counter()
        response = self.client.post(
            "/api/chat",
            json={
                "model": self.name,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 16384},
            },
        )
        response.raise_for_status()
        body = response.json()
        message = body["message"]
        return Reply(
            content=message.get("content", ""),
            tool_calls=[
                ToolCall(c["function"]["name"], c["function"].get("arguments") or {})
                for c in message.get("tool_calls", [])
            ],
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            seconds=time.perf_counter() - started,
        )


class Scripted:
    """Replays a fixed list of replies. Each item may also be a function of the messages so far."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools) -> Reply:
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply
