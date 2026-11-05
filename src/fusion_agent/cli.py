"""Command line entry point.

Everything except `fetch-models` runs with the network guard on, so nothing can leave the machine.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from . import netguard

DATA = Path(os.environ.get("FUSION_DATA", "data"))
DB = DATA / "fusion.duckdb"
RUNS = Path(os.environ.get("FUSION_RUNS", "runs"))
ASSETS = Path(os.environ.get("FUSION_ASSETS", "assets/images"))
MODEL = os.environ.get("FUSION_MODEL", "qwen2.5:7b")
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")


def default_embedder() -> str:
    if "FUSION_EMBEDDER" in os.environ:
        return os.environ["FUSION_EMBEDDER"]
    return "local" if importlib.util.find_spec("sentence_transformers") else "hash"


def open_store(embedder_name: str):
    from .embed import get_embedder
    from .store import Store

    if not DB.exists():
        sys.exit(f"No database at {DB}. Run `fusion-agent generate` and `fusion-agent ingest` first.")
    return Store(DB, get_embedder(embedder_name))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fusion-agent", description=__doc__)
    parser.add_argument("--embedder", default=default_embedder(), choices=["local", "hash"])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate", help="Write the synthetic scenario to the data folder")
    commands.add_parser("fetch-models", help="Download embedding models (the only command that uses the network)")
    commands.add_parser("ingest", help="Embed and load the data into DuckDB")
    commands.add_parser("check-offline", help="Show that remote connections are refused")
    ask = commands.add_parser("ask", help="Ask the agent a question")
    ask.add_argument("question")
    ask.add_argument("--model", default=MODEL)
    ask.add_argument("--verbose", action="store_true", help="Print every tool call")
    evaluate = commands.add_parser("eval", help="Run the evaluation set")
    evaluate.add_argument("--model", action="append", help="Repeat to compare models")
    evaluate.add_argument("--only", nargs="*", help="Question ids to run")
    compare = commands.add_parser("compare", help="Compare saved evaluation reports")
    compare.add_argument("reports", nargs="+", type=Path)
    commands.add_parser("mcp", help="Serve the tools over MCP on stdio")
    monitor = commands.add_parser("monitor", help="Open the Harbour Monitor web UI on 127.0.0.1")
    monitor.add_argument("--port", type=int, default=8765)
    monitor.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)

    if args.command == "fetch-models":
        from .embed import fetch_models

        fetch_models()
        print("Models cached. Everything else now runs offline.")
        return

    netguard.install()

    if args.command == "generate":
        from .scenario import generate

        generate(DATA, ASSETS)
        print(f"Scenario written to {DATA}/")
    elif args.command == "ingest":
        from .embed import get_embedder
        from .store import ingest

        print(json.dumps(ingest(DATA, DB, get_embedder(args.embedder))))
    elif args.command == "check-offline":
        check_offline()
    elif args.command == "ask":
        from .agent import Agent
        from .llm import Ollama

        result = Agent(Ollama(args.model, OLLAMA), open_store(args.embedder), RUNS / "traces").run(args.question)
        if args.verbose:
            for call in result.tool_calls:
                print(
                    f"  -> {call['name']}({json.dumps(call['arguments'])})"
                    + (f"  [{call['error']}]" if "error" in call else "")
                )
        print(result.answer or "(no answer)")
        print(
            f"\n{result.steps} steps, {len(result.tool_calls)} tool calls, {result.seconds}s, "
            f"{result.input_tokens + result.output_tokens} tokens. Trace: {RUNS / 'traces' / result.run_id}.jsonl"
        )
        for problem in result.problems:
            print(f"WARNING: {problem}")
        if result.flags:
            print("Flags: " + ", ".join(sorted(set(result.flags))))
    elif args.command == "eval":
        from .agent import Agent
        from .evaluate import comparison_table, run_eval
        from .llm import Ollama

        store = open_store(args.embedder)
        reports = []
        for model in args.model or [MODEL]:
            print(f"\n{model}")
            agent = Agent(Ollama(model, OLLAMA), store, RUNS / "traces")
            reports.append(run_eval(agent, Path("evals/questions.yaml"), RUNS / "evals", args.only))
        print("\n" + comparison_table(reports))
    elif args.command == "compare":
        from .evaluate import comparison_table

        print(comparison_table([json.loads(p.read_text()) for p in args.reports]))
    elif args.command == "mcp":
        from .mcp_server import serve

        serve(open_store(args.embedder))
    elif args.command == "monitor":
        from .monitor import Monitor
        from .monitor import serve as serve_monitor

        serve_monitor(Monitor(open_store(args.embedder), DATA, RUNS, OLLAMA, args.model), args.port)


def check_offline():
    import httpx

    try:
        httpx.get("https://example.com", timeout=5)
        print("FAIL: reached example.com")
        sys.exit(1)
    except httpx.ConnectError as error:
        if "egress blocked" not in str(error).lower():
            raise
        print(f"OK: {error}")
    try:
        version = httpx.get(f"{OLLAMA}/api/version", timeout=5).json()["version"]
        print(f"OK: local Ollama {version} reachable at {OLLAMA}")
    except httpx.HTTPError as error:
        print(f"Ollama not reachable at {OLLAMA}: {error}")
