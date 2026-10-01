"""Operational memory: structured cases from completed investigations.

This is not a chat log. It stores one row per completed investigation, holding
the anomaly that was detected, the root cause that was inferred, the
intervention that was simulated and how that simulation verified. Future
investigations can then ask "has this shape of problem been seen before?".

Scope and honesty rules
-----------------------
Only completed, verified investigations may be stored. Current intervention and
verification are SIMULATED arithmetic projections, so every row records
``simulation = 1``. A simulated outcome must never be presented as a measured
real-world result.

Memory is evidence, never authority. A stored case may raise a question or
suggest a hypothesis; it can never override what the current dataset shows.
Similarity here is a deterministic field-by-field score, not semantic
similarity, and nothing here claims otherwise.

Storage uses Python's standard library ``sqlite3`` with parameterized SQL. No
ORM, no embeddings, no external database.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.tools.dataset import DATA_DIR

#: Default database location, overridable so tests never touch real data.
DEFAULT_DB_PATH = Path(DATA_DIR) / "operational_memory.db"
DB_PATH_ENV = "OPERATIONAL_MEMORY_DB"

#: Deterministic similarity weights. Only the fields actually compared here
#: contribute to a score; no semantic or embedding similarity is implied.
WEIGHT_METRIC = 3
WEIGHT_ROOT_CAUSE = 3
WEIGHT_MACHINE_CATEGORY = 2
WEIGHT_SYMPTOM = 2
WEIGHT_MAGNITUDE = 1

#: Two anomalies are "magnitude similar" within this many percentage points.
MAGNITUDE_TOLERANCE_PCT = 10.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    symptom TEXT,
    anomaly_metric TEXT,
    anomaly_magnitude REAL,
    root_cause_category TEXT,
    root_cause_statement TEXT,
    intervention_type TEXT,
    intervention_description TEXT,
    projected_reduction_percent REAL,
    verification_status TEXT,
    confidence REAL,
    evidence_summary TEXT,
    simulation INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'investigator'
)
"""

COLUMNS = (
    "id",
    "created_at",
    "machine_id",
    "symptom",
    "anomaly_metric",
    "anomaly_magnitude",
    "root_cause_category",
    "root_cause_statement",
    "intervention_type",
    "intervention_description",
    "projected_reduction_percent",
    "verification_status",
    "confidence",
    "evidence_summary",
    "simulation",
    "source",
)

INSERT_SQL = (
    "INSERT INTO investigations ("
    "created_at, machine_id, symptom, anomaly_metric, anomaly_magnitude, "
    "root_cause_category, root_cause_statement, intervention_type, "
    "intervention_description, projected_reduction_percent, "
    "verification_status, confidence, evidence_summary, simulation, source"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def resolve_db_path(db_path=None):
    """Database location, overridable by argument or environment."""
    if db_path is not None:
        return Path(db_path)
    import os

    from_env = os.environ.get(DB_PATH_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_DB_PATH


def connect(db_path=None) -> sqlite3.Connection:
    """Open a connection with row access by name."""
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_memory(db_path=None) -> str:
    """Create the database and table if they do not already exist.

    Returns the resolved path. Safe to call repeatedly.
    """
    path = resolve_db_path(db_path)
    with connect(path) as connection:
        connection.execute(SCHEMA)
        connection.commit()
    return str(path)


# CHUNK_ANCHOR


def store_investigation(result: dict, db_path=None, source: str = "investigator"):
    """Store a completed, verified investigation. Returns the new row id.

    Refuses (returns ``None``) anything that is not a completed, verified case:
    a missing root cause, a missing verification, or a failed verification.
    An unverified result must never enter memory as a successful resolution.
    """
    if not _is_storable(result):
        return None

    initialize_memory(db_path)
    record = _record_from_result(result, source)

    with connect(db_path) as connection:
        cursor = connection.execute(INSERT_SQL, record)
        connection.commit()
        return cursor.lastrowid


def _is_storable(result: dict) -> bool:
    """Only completed, verified investigations become operational memory."""
    if not result:
        return False
    if result.get("status") != "complete":
        return False
    root_cause = result.get("root_cause")
    verification = result.get("verification")
    if not root_cause or not verification:
        return False
    return str(verification.get("status", "")).upper() == "PASS"


def _record_from_result(result: dict, source: str) -> tuple:
    """Flatten an investigation result into schema column values."""
    anomaly = result.get("anomaly") or {}
    root_cause = result.get("root_cause") or {}
    intervention = result.get("intervention") or {}
    verification = result.get("verification") or {}
    machine = (result.get("evidence") or {}).get("machine") or {}

    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        str(result.get("machine_id", "")),
        _symptom_for(anomaly),
        anomaly.get("metric"),
        anomaly.get("magnitude_pct"),
        root_cause.get("category"),
        root_cause.get("statement"),
        intervention.get("type"),
        intervention.get("description"),
        intervention.get("projected_reduction_percent"),
        verification.get("status"),
        root_cause.get("confidence"),
        json.dumps(_evidence_summary(result), sort_keys=True),
        # Simulated unless a future phase performs real verification.
        1 if intervention.get("simulation", True) or verification.get("simulation", True) else 0,
        source,
    )


def _symptom_for(anomaly: dict):
    """A short normalised symptom label derived from the measured metric."""
    metric = anomaly.get("metric")
    if not metric:
        return None
    return "elevated_{0}".format(metric)


def _evidence_summary(result: dict) -> dict:
    """Compact, already-computed evidence snapshot for later comparison."""
    evidence = result.get("evidence") or {}
    metrics = evidence.get("metrics") or {}
    production = evidence.get("production") or {}
    return {
        "energy_change_pct": (metrics.get("energy_kwh") or {}).get("change_pct"),
        "runtime_change_pct": (metrics.get("runtime_h") or {}).get("change_pct"),
        "intensity_change_pct": (metrics.get("intensity") or {}).get("change_pct"),
        "efficiency_change_pct": (metrics.get("efficiency_factor") or {}).get("change_pct"),
        "production_change_pct": production.get("change_pct"),
        "machine_category": (evidence.get("machine") or {}).get("category"),
        "window": "{0}..{1}".format(
            (result.get("anomaly") or {}).get("start_date"),
            (result.get("anomaly") or {}).get("end_date"),
        ),
        "maintenance_records": len(evidence.get("maintenance_records") or []),
        "technician_notes": len(evidence.get("technician_notes") or []),
        "prior_incidents": len(evidence.get("historical_incidents") or []),
    }


def get_investigation(investigation_id: int, db_path=None):
    """Fetch one stored case by id, or ``None``."""
    initialize_memory(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT {0} FROM investigations WHERE id = ?".format(", ".join(COLUMNS)),
            (investigation_id,),
        ).fetchone()
    return _row_to_case(row) if row else None


def list_investigations(limit: int = 100, db_path=None) -> list:
    """All stored cases, newest first."""
    initialize_memory(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT {0} FROM investigations ORDER BY id DESC LIMIT ?".format(
                ", ".join(COLUMNS)
            ),
            (int(limit),),
        ).fetchall()
    return [_row_to_case(row) for row in rows]


def _row_to_case(row) -> dict:
    """Convert a DB row into a plain dictionary, decoding JSON columns."""
    case = {key: row[key] for key in COLUMNS}
    case["simulation"] = bool(case["simulation"])
    try:
        case["evidence_summary"] = json.loads(case["evidence_summary"] or "{}")
    except (ValueError, TypeError):
        case["evidence_summary"] = {}
    return case


def search_similar_incidents(
    anomaly_metric=None,
    root_cause_category=None,
    machine_category=None,
    symptom=None,
    anomaly_magnitude=None,
    limit: int = 5,
    exclude_id=None,
    db_path=None,
) -> list:
    """Rank stored cases by deterministic field comparison.

    Score is the sum of the weights in ``WEIGHT_*`` for the fields that match.
    Only those fields are compared: this is field similarity, not semantic
    similarity, and the returned ``reason`` lists exactly which fields matched
    so nothing stronger is implied.

    Returns a list of ``{"id", "score", "case", "reason"}`` sorted by score
    descending, then by id for a stable order.
    """
    initialize_memory(db_path)
    matches = []

    for case in list_investigations(limit=100000, db_path=db_path):
        if exclude_id is not None and case["id"] == exclude_id:
            continue

        score = 0
        reasons = []

        if anomaly_metric and case["anomaly_metric"] == anomaly_metric:
            score += WEIGHT_METRIC
            reasons.append("same anomaly metric ({0})".format(anomaly_metric))
        if (
            root_cause_category
            and case["root_cause_category"] == root_cause_category
        ):
            score += WEIGHT_ROOT_CAUSE
            reasons.append(
                "same root cause category ({0})".format(root_cause_category)
            )

        stored_category = (case["evidence_summary"] or {}).get("machine_category")
        if machine_category and stored_category and stored_category == machine_category:
            score += WEIGHT_MACHINE_CATEGORY
            reasons.append("same machine category ({0})".format(machine_category))

        if symptom and case["symptom"] == symptom:
            score += WEIGHT_SYMPTOM
            reasons.append("same symptom ({0})".format(symptom))

        if (
            anomaly_magnitude is not None
            and case["anomaly_magnitude"] is not None
            and abs(case["anomaly_magnitude"] - anomaly_magnitude)
            <= MAGNITUDE_TOLERANCE_PCT
        ):
            score += WEIGHT_MAGNITUDE
            reasons.append(
                "similar magnitude (within {0:.0f} points)".format(
                    MAGNITUDE_TOLERANCE_PCT
                )
            )

        if score > 0:
            matches.append(
                {
                    "id": case["id"],
                    "score": score,
                    "case": case,
                    "reason": reasons,
                }
            )

    matches.sort(key=lambda match: (-match["score"], match["id"]))
    return matches[: int(limit)]


def search_memory_for_result(result: dict, limit: int = 5, db_path=None) -> list:
    """Search memory using the attributes of the current investigation."""
    anomaly = result.get("anomaly") or {}
    root_cause = result.get("root_cause") or {}
    evidence = result.get("evidence") or {}

    return search_similar_incidents(
        anomaly_metric=anomaly.get("metric"),
        root_cause_category=root_cause.get("category"),
        machine_category=(evidence.get("machine") or {}).get("category"),
        symptom=_symptom_for(anomaly) if anomaly.get("metric") else None,
        anomaly_magnitude=anomaly.get("magnitude_pct"),
        limit=limit,
        db_path=db_path,
    )