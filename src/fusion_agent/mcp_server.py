"""Expose the same tools over the Model Context Protocol.

Any MCP client (Claude Desktop, an IDE, another agent) can then query the fused data directly, with the same
schemas, validation and source ids as the built-in agent. Runs over stdio, so it needs no network.
"""

import json

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from .store import Store
from .tools import TOOLS


def build(store: Store) -> Server:
    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=t.name, description=t.description, input_schema=t.schema()["function"]["parameters"])
                for t in TOOLS.values()
            ]
        )

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        def reply(payload: dict, error: bool = False) -> types.CallToolResult:
            return types.CallToolResult(
                content=[types.TextContent(text=json.dumps(payload, default=str))], is_error=error
            )

        tool = TOOLS.get(params.name)
        if not tool:
            return reply({"error": f"Unknown tool {params.name}"}, error=True)
        try:
            args = tool.args.model_validate(params.arguments or {})
        except ValidationError as error:
            return reply({"error": str(error)}, error=True)
        return reply(tool.run(store, args))

    return Server("fusion-agent", on_list_tools=list_tools, on_call_tool=call_tool)


def serve(store: Store) -> None:
    server = build(store)

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
