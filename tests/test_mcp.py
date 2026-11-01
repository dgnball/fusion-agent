import json

import anyio
from mcp import Client

from fusion_agent.mcp_server import build
from fusion_agent.tools import TOOLS


def test_mcp_lists_and_calls_tools(store):
    async def scenario():
        async with Client(build(store)) as client:
            listed = await client.list_tools()
            assert {t.name for t in listed.tools} == set(TOOLS)
            result = await client.call_tool("track_anomalies", {"kind": "ais_gap"})
            payload = json.loads(result.content[0].text)
            assert payload["results"][0]["vessel_name"] == "Sable Tern"
            bad = await client.call_tool("track_anomalies", {"kind": "speeding"})
            assert bad.is_error

    anyio.run(scenario)
