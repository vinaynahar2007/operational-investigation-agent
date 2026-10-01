"""Minimal HTTP adapter over the existing investigation engine.

This module deliberately contains no investigation logic. It accepts a machine
id, calls ``app.agents.investigator.investigate`` unchanged, and returns that
result verbatim as JSON. The Python investigator remains the single source of
truth for detection, evidence, hypotheses, root cause, simulation and
verification.

Three endpoints only:

    GET  /api/health        liveness plus whether the LLM layer is configured
    GET  /api/machines      machine ids available to investigate
    POST /api/investigate    {"machine_id": "M04"} -> investigation result

Run with:  python -m app.api   (or: uvicorn app.api:app --port 8000)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.investigator import investigate
from app.llm.adapter import LLMAdapter
from app.tools.dataset import load_machines, normalize_machine_id

app = FastAPI(
    title="Operational Investigation Agent",
    description="Deterministic investigation engine exposed over HTTP.",
    version="1.0.0",
)

# The Vite dev server runs on a different port, so allow local origins only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class InvestigateRequest(BaseModel):
    """Request body. Only the machine id is accepted."""

    machine_id: str = Field(..., min_length=1, description="Machine identifier")


@app.get("/api/health")
def health() -> dict:
    """Liveness probe reporting whether optional LLM reasoning is configured."""
    adapter = LLMAdapter()
    return {
        "status": "ok",
        "llm_configured": adapter.is_available(),
        "llm_provider": adapter.provider_name(),
    }


@app.get("/api/machines")
def machines() -> dict:
    """Machine ids that can be investigated."""
    frame = load_machines()
    return {
        "machines": [
            {
                "machine_id": row["machine_id"],
                "machine_name": str(row["machine_name"]),
                "category": str(row["category"]),
            }
            for _, row in frame.iterrows()
        ]
    }


@app.post("/api/investigate")
def run_investigation(request: InvestigateRequest) -> dict:
    """Run the unchanged investigator and return its result as JSON.

    Investigation semantics are untouched: the same ``investigate(machine_id)``
    call is made regardless of transport. The adapter only maps an unknown
    machine onto a 404 so a client sees a clear error instead of a 200 with a
    failed body.
    """
    machine_id = normalize_machine_id(request.machine_id)
    result = investigate(machine_id)

    if not _machine_known(machine_id):
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Unknown machine '{0}'.".format(machine_id),
                "result": result,
            },
        )

    # Encoding only; no value is altered or recomputed here.
    return jsonable_encoder(result)


def _machine_known(machine_id: str) -> bool:
    """Whether the machine id exists in the dataset."""
    return normalize_machine_id(machine_id) in set(load_machines()["machine_id"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)