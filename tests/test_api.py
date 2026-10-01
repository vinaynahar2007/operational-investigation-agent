"""Minimal smoke tests for the HTTP adapter.

Four checks only, using FastAPI's in-process TestClient so no server has to be
running: health, machines, an M04 investigation, and the unknown-machine 404.

The adapter is a passthrough, so these assert the wiring and the shape of the
response, not the investigation itself (already covered by the other suites).
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "llm_configured" in payload


def test_machines_endpoint_lists_machine_ids(client):
    response = client.get("/api/machines")
    assert response.status_code == 200
    machines = response.json()["machines"]
    ids = {machine["machine_id"] for machine in machines}
    assert {"M03", "M04", "M05"} <= ids


def test_investigate_m04_returns_a_complete_result(client):
    response = client.post("/api/investigate", json={"machine_id": "M04"})
    assert response.status_code == 200
    result = response.json()

    assert result["machine_id"] == "M04"
    assert result["status"] == "complete"
    assert result["anomaly"] is not None
    assert result["root_cause"]["category"] == "equipment_efficiency"
    assert result["intervention"]["simulation"] is True
    assert result["verification"]["simulation"] is True
    assert [entry["step"] for entry in result["trace"]]


def test_investigate_accepts_trimmed_case_insensitive_machine_ids(client):
    for raw in ["M04", " m04 ", "m04"]:
        response = client.post("/api/investigate", json={"machine_id": raw})
        assert response.status_code == 200
        assert response.json()["machine_id"] == "M04"


def test_investigate_unknown_machine_returns_404(client):
    response = client.post("/api/investigate", json={"machine_id": "ZZZ"})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "ZZZ" in detail["message"]