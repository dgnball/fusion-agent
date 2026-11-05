"""The Harbour Monitor UI: scenario payload and streamed agent events."""

import json
from pathlib import Path

from fusion_agent.agent import Agent
from fusion_agent.llm import Reply, Scripted, ToolCall
from fusion_agent.monitor import Monitor, jsonable


def test_scenario_payload(store, tmp_path):
    payload = json.loads(Monitor(store, tmp_path, tmp_path, "http://127.0.0.1:1", "test").scenario)
    assert len(payload["reports"]) == 11
    assert [r["id"] for r in payload["reports"] if r["injection"]] == ["R-008"]
    assert any(z["vessel_name"] == "Grey Heron" for z in payload["detected"]["zone_entries"])
    assert any(g["vessel_name"] == "Sable Tern" for g in payload["detected"]["ais_gaps"])


def test_listener_sees_tool_sources(store):
    call = ToolCall("track_anomalies", {"kind": "ais_gap"})
    answer = "Sable Tern went quiet [A:V01:gap:2026-03-14T01:40]."
    events = []
    Agent(Scripted([Reply(tool_calls=[call]), Reply(content=answer)]), store).run("Gaps?", listener=events.append)
    kinds = [e["event"] for e in events]
    assert kinds == ["run_start", "model_call", "tool_call", "tool_result", "model_call", "guard", "run_end"]
    assert "A:V01:gap:2026-03-14T01:40" in events[3]["sources"]
    json.dumps(events, default=jsonable)


def test_page_has_no_external_resources():
    page = (Path(__file__).parents[1] / "src/fusion_agent/monitor.html").read_text()
    assert "http://" not in page and "https://" not in page
