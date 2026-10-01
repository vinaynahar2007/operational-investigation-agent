import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.investigator import investigate
from app.tools.memory import (
    COLUMNS,
    connect,
    get_investigation,
    initialize_memory,
    list_investigations,
    search_memory_for_result,
    search_similar_incidents,
    store_investigation,
)


@pytest.fixture()
def db(tmp_path):
    """A fresh, isolated database for each test."""
    return str(tmp_path / "memory.db")


@pytest.fixture()
def m04_result(db):
    """A real, completed M04 investigation that has NOT been stored.

    Memory is disabled so tests control exactly when storage happens.
    """
    return investigate("M04", memory_db=False)


def _fake_result(**overrides):
    """A minimal completed result, for storage-rule tests."""
    result = {
        "machine_id": "M99",
        "status": "complete",
        "anomaly": {
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
            "metric": "intensity",
            "magnitude_pct": 12.0,
        },
        "evidence": {
            "machine": {"category": "heating"},
            "metrics": {
                "energy_kwh": {"change_pct": 12.0},
                "runtime_h": {"change_pct": 0.2},
                "intensity": {"change_pct": 12.0},
                "efficiency_factor": {"change_pct": 12.0},
            },
            "production": {"change_pct": 2.0},
            "maintenance_records": [],
            "technician_notes": [],
            "historical_incidents": [],
        },
        "root_cause": {
            "category": "equipment_efficiency",
            "statement": "Probable equipment degradation.",
            "confidence": 0.7,
        },
        "intervention": {
            "type": "equipment_efficiency",
            "description": "Inspect and restore efficiency.",
            "projected_reduction_percent": 6.0,
            "simulation": True,
        },
        "verification": {
            "status": "PASS",
            "observed_reduction_percent": 6.0,
            "threshold_percent": 5.0,
            "simulation": True,
        },
    }
    result.update(overrides)
    return result


# --------------------------------------------------------------------------
# 1-4: database lifecycle and CRUD
# --------------------------------------------------------------------------


def test_database_initializes(db):
    path = initialize_memory(db)
    assert Path(path).exists()
    assert list_investigations(db_path=db) == []


def test_initialize_is_idempotent(db):
    initialize_memory(db)
    initialize_memory(db)
    assert list_investigations(db_path=db) == []


def test_investigation_can_be_stored(db, m04_result):
    record_id = store_investigation(m04_result, db_path=db)
    assert record_id is not None
    assert isinstance(record_id, int)


def test_stored_investigation_can_be_retrieved(db, m04_result):
    record_id = store_investigation(m04_result, db_path=db)
    case = get_investigation(record_id, db_path=db)
    assert case["machine_id"] == "M04"
    assert case["root_cause_category"] == "equipment_efficiency"
    assert case["verification_status"] == "PASS"
    assert set(COLUMNS) <= set(case)


def test_list_returns_investigations(db, m04_result):
    store_investigation(m04_result, db_path=db)
    store_investigation(m04_result, db_path=db)
    cases = list_investigations(db_path=db)
    assert len(cases) == 2
    assert cases[0]["id"] > cases[1]["id"]  # newest first


def test_get_missing_investigation_returns_none(db):
    assert get_investigation(9999, db_path=db) is None


# --------------------------------------------------------------------------
# 5-7: similarity
# --------------------------------------------------------------------------


def test_similar_incident_search_works(db, m04_result):
    store_investigation(m04_result, db_path=db)
    matches = search_memory_for_result(m04_result, db_path=db)
    assert len(matches) == 1
    match = matches[0]
    assert match["score"] > 0
    assert match["case"]["machine_id"] == "M04"
    assert "same anomaly metric (intensity)" in match["reason"]


def test_similarity_score_is_deterministic(db, m04_result):
    store_investigation(m04_result, db_path=db)
    first = search_memory_for_result(m04_result, db_path=db)
    second = search_memory_for_result(m04_result, db_path=db)
    assert [m["score"] for m in first] == [m["score"] for m in second]


def test_similarity_reasons_list_only_compared_fields(db, m04_result):
    store_investigation(m04_result, db_path=db)
    match = search_memory_for_result(m04_result, db_path=db)[0]
    allowed = ("anomaly metric", "root cause", "machine category", "symptom", "magnitude")
    for reason in match["reason"]:
        assert any(token in reason for token in allowed)


def test_unrelated_incident_receives_lower_similarity(db, m04_result):
    store_investigation(m04_result, db_path=db)
    unrelated = _fake_result(machine_id="M77")
    unrelated["anomaly"] = {
        "start_date": "2026-01-01", "end_date": "2026-01-05",
        "metric": "runtime", "magnitude_pct": 90.0,
    }
    unrelated["root_cause"] = {
        "category": "operational_scheduling", "statement": "s", "confidence": 0.4,
    }
    unrelated["evidence"]["machine"] = {"category": "cooling"}

    matches = search_memory_for_result(unrelated, db_path=db)
    related = search_memory_for_result(m04_result, db_path=db)
    assert not matches or matches[0]["score"] < related[0]["score"]


def test_no_matches_when_database_is_empty(db, m04_result):
    assert search_memory_for_result(m04_result, db_path=db) == []


# --------------------------------------------------------------------------
# 8: simulated flag preserved
# --------------------------------------------------------------------------


def test_simulated_flag_is_preserved(db, m04_result):
    record_id = store_investigation(m04_result, db_path=db)
    assert get_investigation(record_id, db_path=db)["simulation"] is True


def test_stored_case_records_the_projection_not_a_measurement(db, m04_result):
    record_id = store_investigation(m04_result, db_path=db)
    case = get_investigation(record_id, db_path=db)
    assert case["projected_reduction_percent"] == pytest.approx(8.03, abs=0.1)
    assert case["simulation"] is True


# --------------------------------------------------------------------------
# 9: storage rules
# --------------------------------------------------------------------------


def test_failed_verification_is_not_stored_as_success(db):
    failed = _fake_result()
    failed["verification"] = {
        "status": "FAIL",
        "observed_reduction_percent": 0.0,
        "threshold_percent": 5.0,
        "simulation": True,
    }
    assert store_investigation(failed, db_path=db) is None
    assert list_investigations(db_path=db) == []


def test_verification_failed_status_is_not_stored(db):
    assert store_investigation(_fake_result(status="verification_failed"), db_path=db) is None


def test_inconclusive_result_is_not_stored(db):
    assert store_investigation(_fake_result(status="inconclusive"), db_path=db) is None


def test_missing_verification_is_not_stored(db):
    assert store_investigation(_fake_result(verification=None), db_path=db) is None


def test_missing_root_cause_is_not_stored(db):
    assert store_investigation(_fake_result(root_cause=None), db_path=db) is None


def test_empty_result_is_not_stored(db):
    assert store_investigation({}, db_path=db) is None


# --------------------------------------------------------------------------
# 10: hostile input
# --------------------------------------------------------------------------


def test_sql_injection_like_input_does_not_break_database(db, m04_result):
    hostile = m04_result
    hostile["root_cause"] = dict(hostile["root_cause"])
    hostile["root_cause"]["statement"] = "Robert'); DROP TABLE investigations;--"
    hostile["machine_id"] = "M04'; DELETE FROM investigations;--"
    record_id = store_investigation(hostile, db_path=db)

    assert record_id is not None
    # Table intact and the hostile text stored verbatim as data.
    cases = list_investigations(db_path=db)
    assert len(cases) == 1
    assert "DROP TABLE" in cases[0]["root_cause_statement"]


def test_hostile_id_does_not_expose_other_rows(db, m04_result):
    store_investigation(m04_result, db_path=db)
    hostile = _fake_result(machine_id="x' OR '1'='1")
    store_investigation(hostile, db_path=db)
    assert len(list_investigations(db_path=db)) == 2


def test_database_remains_usable_after_hostile_input(db, m04_result):
    store_investigation(
        _fake_result(machine_id="'; DROP TABLE investigations;--"), db_path=db
    )
    store_investigation(m04_result, db_path=db)
    assert get_investigation(1, db_path=db) is not None


# --------------------------------------------------------------------------
# 11: duplicates
# --------------------------------------------------------------------------


def test_duplicate_storage_does_not_corrupt_existing_cases(db, m04_result):
    first_id = store_investigation(m04_result, db_path=db)
    second_id = store_investigation(m04_result, db_path=db)

    assert first_id != second_id
    first = get_investigation(first_id, db_path=db)
    assert first["machine_id"] == "M04"
    assert first["root_cause_category"] == "equipment_efficiency"


def test_storing_twice_leaves_both_rows_readable(db, m04_result):
    store_investigation(m04_result, db_path=db)
    store_investigation(m04_result, db_path=db)
    cases = list_investigations(db_path=db)
    assert len(cases) == 2
    assert all(case["machine_id"] == "M04" for case in cases)


def test_schema_is_stable_across_initializations(db, m04_result):
    store_investigation(m04_result, db_path=db)
    initialize_memory(db)
    with connect(db) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(investigations)")]
    assert columns == list(COLUMNS)


def test_resolve_db_path_prefers_explicit_argument(tmp_path):
    from app.tools.memory import resolve_db_path

    assert resolve_db_path(tmp_path / "x.db") == tmp_path / "x.db"