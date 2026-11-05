"""The scenario's planted events must be exactly what the analytics find. Evaluation relies on it."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from fusion_agent.tools import (
    TOOLS,
    SearchReports,
    TrackAnomalies,
    VesselTrack,
    track_anomalies,
    vessel_track,
)


def test_only_planted_zone_visits(store):
    visits = [(v["vessel_name"], v["first_seen"], v["last_seen"]) for v in store.zone_entries()]
    assert visits == [
        ("Grey Heron", datetime(2026, 3, 14, 2, 10), datetime(2026, 3, 14, 2, 40)),
        ("Bright Cove", datetime(2026, 3, 14, 7, 30), datetime(2026, 3, 14, 8, 40)),
    ]


def test_only_planted_ais_gap(store):
    gaps = store.ais_gaps()
    assert [(g["vessel_name"], g["gap_minutes"]) for g in gaps] == [("Sable Tern", 90)]


def test_time_window_filters_anomalies(store):
    args = TrackAnomalies(kind="restricted_zone", start="2026-03-14T00:00Z", end="2026-03-14T06:00Z")
    results = track_anomalies(store, args)["results"]
    assert [r["vessel_name"] for r in results] == ["Grey Heron"]
    assert results[0]["source_id"] == "A:V02:zone:2026-03-14T02:10"


def test_track_labels_areas_and_silence(store):
    areas = [a["area"] for a in vessel_track(store, VesselTrack(vessel="Sable Tern"))["areas_visited"]]
    assert "No position reports (transponder silent)" in areas
    heron = [a["area"] for a in vessel_track(store, VesselTrack(vessel="grey heron"))["areas_visited"]]
    assert "Northgate Jetty restricted zone" in heron
    assert "Southern grounds" not in heron


def test_unknown_vessel_is_an_error_not_an_exception(store):
    assert "error" in vessel_track(store, VesselTrack(vessel="Black Swan"))


def test_arguments_are_validated():
    with pytest.raises(ValidationError):
        SearchReports(query="x", limit=50)
    with pytest.raises(ValidationError):
        TrackAnomalies(kind="speeding")


def test_every_tool_has_a_schema():
    for tool in TOOLS.values():
        schema = tool.schema()
        assert schema["function"]["name"] == tool.name
        assert schema["function"]["parameters"]["type"] == "object"


def test_assets_replace_rendered_images(tmp_path):
    from PIL import Image

    from fusion_agent.scenario import generate

    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGB", (64, 36), (10, 200, 30)).save(assets / "I-003.png")
    generate(tmp_path / "data", assets)
    images = tmp_path / "data" / "images"
    assert (images / "I-003.png").read_bytes() == (assets / "I-003.png").read_bytes()
    assert Image.open(images / "I-001.png").size != (64, 36)
