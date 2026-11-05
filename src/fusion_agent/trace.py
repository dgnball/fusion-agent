"""Structured traces of agent runs, one JSON object per line.

A trace records every model call and tool call with timings, token counts, arguments and outcomes. That's
enough to replay why the agent answered the way it did, which matters more than the answer itself when
something goes wrong.
"""

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path


class Trace:
    def __init__(self, directory: Path | None, listener: Callable[[dict], None] | None = None):
        self.listener = listener
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.events: list[dict] = []
        self.path = directory / f"{self.run_id}.jsonl" if directory else None
        if self.path:
            directory.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields) -> None:
        record = {
            "run_id": self.run_id,
            "at": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        self.events.append(record)
        if self.listener:
            self.listener(record)
        if self.path:
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
