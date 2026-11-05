"""Harbour Monitor, a small local web UI for the agent.

It serves one page that tells the story of the scenario, lets you ask the agent questions and watch each step
as it happens, and shows saved evaluation results. It uses only the standard library and binds to 127.0.0.1,
so the network guard stays on and nothing leaves the machine.

The browser talks to three endpoints:

- `GET /api/scenario` returns every record the agent can search, plus what the SQL analytics detect
- `POST /api/ask` runs the agent and streams its trace events as newline-delimited JSON
- `GET /img/<id>.png` returns a camera image
"""

import json
import queue
import re
import threading
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import yaml

from .agent import Agent
from .guardrails import scan_for_injection
from .store import Store

PAGE = Path(__file__).with_name("monitor.html")
IMAGE_ID = re.compile(r"I-\d{3}")


def jsonable(value):
    if isinstance(value, datetime | date):
        return value.isoformat(timespec="minutes") if isinstance(value, datetime) else value.isoformat()
    return str(value)


class Monitor:
    def __init__(self, store: Store, data_dir: Path, runs_dir: Path, ollama_url: str, model: str):
        self.store = store
        self.data_dir = data_dir
        self.runs_dir = runs_dir
        self.ollama_url = ollama_url
        self.model = model
        # DuckDB connections aren't safe to share across threads, so one request uses the store at a time.
        self.lock = threading.Lock()
        with self.lock:
            self.scenario = json.dumps(self.build_scenario(), default=jsonable)

    def build_scenario(self) -> dict:
        s = self.store
        reports = s.rows("SELECT id, ts, location, author, title, body FROM reports ORDER BY ts")
        for r in reports:
            r["injection"] = scan_for_injection(r["body"])
        return {
            "coverage": s.coverage(),
            "vessels": s.vessels(),
            "reports": reports,
            "images": s.rows("SELECT id, ts, camera, lat, lon FROM images ORDER BY ts"),
            "tracks": s.rows("SELECT vessel_id, ts, lat, lon, speed_kn FROM tracks ORDER BY vessel_id, ts"),
            "detected": {"zone_entries": s.zone_entries(), "ais_gaps": s.ais_gaps()},
            "questions": self.questions(),
            "evals": self.evals(),
        }

    def questions(self) -> list[dict]:
        path = Path("evals/questions.yaml")
        if not path.exists():
            return []
        return [{"id": q["id"], "question": q["question"]} for q in yaml.safe_load(path.read_text())]

    def evals(self) -> list[dict]:
        reports = []
        for path in sorted((self.runs_dir / "evals").glob("*.json")):
            body = json.loads(path.read_text())
            reports.append({"file": path.name, **body})
        return reports

    def models(self) -> list[str]:
        try:
            tags = httpx.get(f"{self.ollama_url}/api/tags", timeout=3).json()
            return sorted(m["name"] for m in tags.get("models", []))
        except httpx.HTTPError:
            return []

    def ask(self, question: str, model_name: str):
        """Run the agent in a worker thread and yield its trace events as they happen."""
        from .llm import Ollama

        events: queue.Queue = queue.Queue()
        done = object()

        def work():
            try:
                with self.lock:
                    agent = Agent(Ollama(model_name, self.ollama_url), self.store, self.runs_dir / "traces")
                    result = agent.run(question, listener=events.put)
                events.put({"event": "result", **result.__dict__})
            except Exception as error:  # surface failures in the UI rather than a dropped connection
                events.put({"event": "error", "message": f"{type(error).__name__}: {error}"})
            finally:
                events.put(done)

        threading.Thread(target=work, daemon=True).start()
        while (item := events.get()) is not done:
            yield item


def handler(monitor: Monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/scenario":
                self.send(200, monitor.scenario.encode(), "application/json")
            elif path == "/api/models":
                body = {"models": monitor.models(), "default": monitor.model}
                self.send(200, json.dumps(body).encode(), "application/json")
            elif (m := re.fullmatch(r"/img/(I-\d{3})\.png", path)) and IMAGE_ID.fullmatch(m.group(1)):
                image = monitor.data_dir / "images" / f"{m.group(1)}.png"
                if image.exists():
                    self.send(200, image.read_bytes(), "image/png")
                else:
                    self.send(404, b"not found", "text/plain")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/ask":
                self.send(404, b"not found", "text/plain")
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            question = str(body.get("question", "")).strip()
            if not question:
                self.send(400, b"question is required", "text/plain")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for event in monitor.ask(question, str(body.get("model") or monitor.model)):
                try:
                    self.wfile.write((json.dumps(event, default=jsonable) + "\n").encode())
                    self.wfile.flush()
                except BrokenPipeError:
                    break

    return Handler


def serve(monitor: Monitor, port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(monitor))
    print(f"Harbour Monitor running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
