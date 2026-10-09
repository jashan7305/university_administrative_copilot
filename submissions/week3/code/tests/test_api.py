"""API smoke tests."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_classify_validates_input():
    assert client.post("/intent/classify", json={"text": ""}).status_code == 422


def test_copilot_query_returns_service_request():
    from nlp.pipeline.copilot import INTENT_PATH
    import pytest
    if not INTENT_PATH.exists():
        pytest.skip("models not trained")
    r = client.post("/copilot/query", json={"query": "how do i apply for revaluation of my TEE paper"}).json()
    assert r["intent"] == "EXAMINATIONS" and r["service_request"]["department"]
