import pytest

from fusion_agent.embed import HashEmbedder
from fusion_agent.scenario import generate
from fusion_agent.store import Store, ingest


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    generate(data)
    ingest(data, data / "test.duckdb", HashEmbedder())
    return Store(data / "test.duckdb", HashEmbedder())
