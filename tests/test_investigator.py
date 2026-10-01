import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents import investigator
from app.agents.investigator import (
    HYPOTHESIS_EQUIPMENT,
    HYPOTHESIS_PRODUCTION,
    HYPOTHESIS_RUNTIME,
    STAGE_COMPLETE,
    STAGE_DETECT,
    STAGE_EVIDENCE,
    STAGE_EVALUATE,
    STAGE_HYPOTHESES,
    STAGE_INTERVENTION,
    STAGE_MEMORY_SEARCH,
    STAGE_MEMORY_STORE,
    STAGE_VERIFY,
    STAGE_ROOT_CAUSE,
    STATUS_REJECTED,
    STATUS_SUPPORTED,
    TRACE_STAGES,
    investigate,
)
from app.tools.memory import get_investigation, list_investigations


@pytest.fixture()
def db(tmp_path):
    """A fresh, isolated memory database for each test."""
    return str(tmp_path / "memory.db")

REQUIRED_RESULT_KEYS = {
    "machine_id",
    "question",
    "anomaly",
    "evidence",
    "hypotheses",
    "root_cause",
    "recommended_next_action",
    "trace",
}


@pytest.fixture(scope="module")
def m04():
    return investigate("M04")


@pytest.fixture(scope="module")
def m03():
    return investigate("M03")


def _hypothesis(result, name):
    for hypothesis in result["hypotheses"]:
        if hypothesis["name"] == name:
            return hypothesis
    raise AssertionError("hypothesis {0} missing".format(name))


def _trace_entry(result, step):
    """Look a trace entry up by stage name, never by position."""
    for entry in result["trace"]:
        if entry["step"] == step:
            return entry
    raise AssertionError("trace step {0} missing".format(step))


# --------------------------------------------------------------------------
# M04: intensity / equipment condition
# --------------------------------------------------------------------------


def test_m04_investigation_completes(m04):
    assert m04["status"] == "complete"
    assert REQUIRED_RESULT_KEYS <= set(m04)
    assert m04["anomaly"] is not None
    assert m04["root_cause"] is not None


def test_m04_selects_equipment_efficiency_root_cause(m04):
    assert m04["root_cause"]["category"] == "equipment_efficiency"
    assert _hypothesis(m04, HYPOTHESIS_EQUIPMENT)["status"] == STATUS_SUPPORTED


def test_m04_rejects_the_runtime_explanation(m04):
    """Runtime was flat, so a scheduling cause must not be selected."""
    assert _hypothesis(m04, HYPOTHESIS_RUNTIME)["status"] == STATUS_REJECTED
    assert m04["root_cause"]["category"] != "operational_scheduling"


def test_m04_production_does_not_explain_the_rise(m04):
    assert _hypothesis(m04, HYPOTHESIS_PRODUCTION)["status"] != STATUS_SUPPORTED


def test_m04_evidence_shows_intensity_up_and_runtime_flat(m04):
    evidence = m04["evidence"]
    metrics = evidence["metrics"]
    assert metrics["intensity"]["ratio"] > 1.08
    assert metrics["runtime_h"]["ratio"] < 1.05
    assert metrics["energy_kwh"]["ratio"] > evidence["production"]["ratio"]


def test_m04_root_cause_uses_probable_language(m04):
    assert "Probable" in m04["root_cause"]["statement"]


# --------------------------------------------------------------------------
# M03: runtime / scheduling
# --------------------------------------------------------------------------


def test_m03_investigation_completes(m03):
    assert m03["status"] == "complete"
    assert m03["root_cause"] is not None


def test_m03_selects_runtime_scheduling_root_cause(m03):
    assert m03["root_cause"]["category"] == "operational_scheduling"
    assert _hypothesis(m03, HYPOTHESIS_RUNTIME)["status"] == STATUS_SUPPORTED


def test_m03_rejects_the_equipment_explanation(m03):
    """Intensity was flat, so equipment degradation must not be selected."""
    assert _hypothesis(m03, HYPOTHESIS_EQUIPMENT)["status"] == STATUS_REJECTED
    assert m03["root_cause"]["category"] != "equipment_efficiency"


def test_m03_evidence_shows_runtime_up_and_intensity_flat(m03):
    metrics = m03["evidence"]["metrics"]
    assert metrics["runtime_h"]["ratio"] > 1.08
    assert metrics["intensity"]["ratio"] < 1.05


def test_runtime_and_equipment_causes_are_distinguished(m04, m03):
    """The two failure modes must not collapse into one answer."""
    assert m04["root_cause"]["category"] != m03["root_cause"]["category"]


# --------------------------------------------------------------------------
# clean machine, unknown machine, and failure handling
# --------------------------------------------------------------------------


def test_m01_has_no_investigation_worthy_anomaly():
    result = investigate("M01")
    assert result["anomaly"] is None
    assert result["root_cause"] is None
    assert result["hypotheses"] == []
    assert result["status"] == "no_anomaly"


def test_m02_has_no_investigation_worthy_anomaly():
    result = investigate("M02")
    assert result["anomaly"] is None
    assert result["root_cause"] is None


def test_unknown_machine_fails_cleanly():
    result = investigate("M99")
    assert result["status"] == "failed"
    assert result["root_cause"] is None
    assert result["hypotheses"] == []
    assert result["recommended_next_action"] is None


def test_unknown_machine_records_a_failed_stage_and_skips_the_rest():
    """An unknown machine fails DETECT and EVIDENCE; later stages are skipped."""
    result = investigate("M99")
    statuses = {entry["step"]: entry["status"] for entry in result["trace"]}
    assert statuses[STAGE_DETECT] == "failed"
    assert statuses[STAGE_EVIDENCE] == "failed"
    assert statuses[STAGE_ROOT_CAUSE] == "skipped"
    assert statuses[STAGE_INTERVENTION] == "skipped"
    assert statuses[STAGE_VERIFY] == "skipped"


def test_inconclusive_evidence_asserts_no_root_cause():
    """M05's 6% rise is below the material band, so no cause is asserted."""
    result = investigate("M05")
    assert result["root_cause"] is None
    assert all(h["status"] != STATUS_SUPPORTED for h in result["hypotheses"])
    assert "insufficient" in result["recommended_next_action"].lower()


def test_multiple_anomalies_do_not_crash():
    """Only one window per machine exists today; extra windows must be kept."""
    result = investigate("M04")
    assert "additional_windows" in result["anomaly"]
    assert isinstance(result["anomaly"]["additional_windows"], list)


# --------------------------------------------------------------------------
# trace
# --------------------------------------------------------------------------


def test_trace_contains_all_required_stages(m04):
    steps = [entry["step"] for entry in m04["trace"]]
    for stage in (
        STAGE_DETECT,
        STAGE_EVIDENCE,
        STAGE_HYPOTHESES,
        STAGE_EVALUATE,
        STAGE_ROOT_CAUSE,
        STAGE_INTERVENTION,
        STAGE_VERIFY,
    ):
        assert stage in steps
    assert steps == list(TRACE_STAGES)


def test_trace_entries_have_step_status_and_summary(m04):
    for entry in m04["trace"]:
        assert {"step", "status", "summary"} <= set(entry)
        assert entry["status"] in {"complete", "failed", "skipped", "inconclusive"}
        assert entry["summary"]


def test_investigation_is_deterministic():
    """Identical inputs give identical results (memory disabled: no I/O)."""
    assert investigate("M04", memory_db=False) == investigate("M04", memory_db=False)


# --------------------------------------------------------------------------
# structure and evidence integrity
# --------------------------------------------------------------------------


def test_all_five_hypotheses_are_evaluated(m04):
    names = {hypothesis["name"] for hypothesis in m04["hypotheses"]}
    assert names == {
        HYPOTHESIS_PRODUCTION,
        HYPOTHESIS_RUNTIME,
        HYPOTHESIS_EQUIPMENT,
        "maintenance_operational_condition",
        "historical_similar_incident",
    }


def test_hypotheses_carry_evidence_and_status(m04):
    for hypothesis in m04["hypotheses"]:
        assert {"name", "status", "evidence", "reason", "confidence"} <= set(hypothesis)
        assert hypothesis["evidence"], hypothesis["name"]
        assert hypothesis["reason"]
        assert 0.0 <= hypothesis["confidence"] <= 1.0


def test_hypothesis_evidence_is_labelled_observed_or_inferred(m04):
    for hypothesis in m04["hypotheses"]:
        for item in hypothesis["evidence"]:
            assert item.startswith(("OBSERVED", "INFERRED")), item


def test_root_cause_contains_supporting_evidence(m04, m03):
    for result in (m04, m03):
        root_cause = result["root_cause"]
        assert root_cause["supporting_evidence"]
        assert 0.0 < root_cause["confidence"] <= 1.0
        assert root_cause["category"]
        assert root_cause["statement"]


def test_recommended_next_action_is_always_a_string(m04, m03):
    for result in (m04, m03):
        assert isinstance(result["recommended_next_action"], str)
        assert result["recommended_next_action"]


def test_evidence_values_are_derived_from_the_dataset(m04):
    """The reported baseline intensity must match a direct recomputation."""
    from app.tools.analysis import summarize_window

    metrics = m04["evidence"]["metrics"]
    baseline = m04["evidence"]["baseline_window"]
    recomputed = summarize_window("M04", baseline["start_date"], baseline["end_date"])
    assert metrics["intensity"]["baseline_value"] == pytest.approx(
        recomputed["average_intensity"]
    )


def test_evidence_separates_observed_records(m04):
    evidence = m04["evidence"]
    assert set(evidence) >= {
        "machine",
        "anomaly",
        "window",
        "baseline_window",
        "metrics",
        "production",
        "maintenance_records",
        "technician_notes",
        "historical_incidents",
        "data_quality",
    }
    assert evidence["data_quality"]["sufficient"] is True


# --------------------------------------------------------------------------
# generality: the rules must not encode M03/M04
# --------------------------------------------------------------------------


def test_no_machine_specific_branches_exist_in_the_investigator():
    """The rules must generalise, so no machine id may appear in the module."""
    source = Path(investigator.__file__).read_text(encoding="utf-8")
    for machine_id in ("M01", "M02", "M03", "M04", "M05"):
        assert machine_id not in source, machine_id


def test_no_dates_are_hard_coded_in_the_investigator():
    source = Path(investigator.__file__).read_text(encoding="utf-8")
    assert "2026-" not in source


def _synthetic_evidence(intensity_ratio, runtime_ratio, efficiency_ratio, production_ratio):
    """A complete evidence dict built purely from chosen ratios."""
    def metric(ratio):
        return {"window_value": 20.0 * ratio, "baseline_value": 20.0,
                "ratio": ratio, "change_pct": round((ratio - 1) * 100, 2)}

    return {
        "machine": {"machine_id": "SYN", "machine_name": "Synthetic", "found": True},
        "window": {"start_date": "2026-01-20", "end_date": "2026-01-30", "days": 11},
        "baseline_window": {
            "start_date": "2025-12-21",
            "end_date": "2026-01-19",
            "days": 30,
        },
        "metrics": {
            "energy_kwh": {"window_value": 116.0, "baseline_value": 100.0,
                           "ratio": 1.16, "change_pct": 16.0},
            "runtime_h": metric(runtime_ratio),
            "intensity": metric(intensity_ratio),
            "efficiency_factor": metric(efficiency_ratio),
        },
        "production": {"window_mean_units": 101.0, "baseline_mean_units": 100.0,
                       "ratio": production_ratio,
                       "change_pct": round((production_ratio - 1) * 100, 2)},
        "maintenance_records": [],
        "technician_notes": [],
        "historical_incidents": [],
        "data_quality": {"sufficient": True, "machine_known": True,
                         "metrics_available": 7, "metrics_expected": 7},
    }


def _evaluate(evidence):
    hypotheses = investigator.build_hypotheses(evidence)
    investigator.evaluate_hypotheses(hypotheses, evidence)
    root_cause = investigator.select_root_cause(hypotheses, evidence)
    return {"hypotheses": hypotheses}, root_cause


def test_evaluation_rules_depend_only_on_measured_ratios():
    """A synthetic equipment profile must select equipment, whatever its id."""
    evidence = _synthetic_evidence(
        intensity_ratio=1.16, runtime_ratio=1.00,
        efficiency_ratio=1.16, production_ratio=1.01,
    )
    result, root_cause = _evaluate(evidence)

    assert _hypothesis(result, HYPOTHESIS_EQUIPMENT)["status"] == STATUS_SUPPORTED
    assert _hypothesis(result, HYPOTHESIS_RUNTIME)["status"] == STATUS_REJECTED
    assert root_cause["category"] == "equipment_efficiency"


def test_evaluation_rules_reject_equipment_when_runtime_is_the_driver():
    """Mirror-image synthetic profile must select scheduling, not equipment."""
    evidence = _synthetic_evidence(
        intensity_ratio=1.005, runtime_ratio=1.25,
        efficiency_ratio=1.00, production_ratio=1.02,
    )
    result, root_cause = _evaluate(evidence)

    assert _hypothesis(result, HYPOTHESIS_RUNTIME)["status"] == STATUS_SUPPORTED
    assert _hypothesis(result, HYPOTHESIS_EQUIPMENT)["status"] == STATUS_REJECTED
    assert root_cause["category"] == "operational_scheduling"


def test_production_is_supported_when_it_explains_the_energy_rise():
    """A large production rise that tracks energy must be able to win."""
    evidence = _synthetic_evidence(
        intensity_ratio=1.10, runtime_ratio=1.10,
        efficiency_ratio=1.00, production_ratio=1.15,
    )
    result, root_cause = _evaluate(evidence)

    assert _hypothesis(result, HYPOTHESIS_PRODUCTION)["status"] == STATUS_SUPPORTED
    assert root_cause["category"] == "production_demand"


def test_flat_everything_yields_no_supported_hypothesis():
    """No signal anywhere must not manufacture a root cause."""
    evidence = _synthetic_evidence(
        intensity_ratio=1.00, runtime_ratio=1.00,
        efficiency_ratio=1.00, production_ratio=1.00,
    )
    result, root_cause = _evaluate(evidence)

    assert root_cause is None
    assert all(
        h["status"] != STATUS_SUPPORTED for h in result["hypotheses"]
    )


def test_missing_ratios_produce_inconclusive_not_crash():
    evidence = _synthetic_evidence(1.16, 1.0, 1.16, 1.01)
    evidence["metrics"]["intensity"]["ratio"] = None
    evidence["metrics"]["intensity"]["change_pct"] = None
    result, root_cause = _evaluate(evidence)

    assert _hypothesis(result, HYPOTHESIS_EQUIPMENT)["status"] != STATUS_SUPPORTED
    assert root_cause is None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_renders_investigation_without_error(capsys):
    from app.main import main

    assert main(["--investigate", "M04", "--skip-validation"]) == 0
    output = capsys.readouterr().out
    assert "OPERATIONAL INVESTIGATION" in output
    assert "PROBABLE ROOT CAUSE" in output
    assert "INVESTIGATION TRACE" in output
    assert "EVIDENCE (OBSERVED)" in output
    assert "HYPOTHESES (INFERRED)" in output


def test_cli_renders_unknown_machine_cleanly(capsys):
    from app.main import main

    assert main(["--investigate", "M99", "--skip-validation"]) == 0
    assert "OPERATIONAL INVESTIGATION" in capsys.readouterr().out


def test_cli_scan_still_works(capsys):
    from app.main import main

    assert main(["--scan", "--skip-validation"]) == 0
    assert "OPERATIONAL ANOMALY SCAN" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Phase 4: simulated intervention and verification
# --------------------------------------------------------------------------


def test_m04_reaches_the_intervention_stage(m04):
    statuses = {entry["step"]: entry["status"] for entry in m04["trace"]}
    assert statuses[STAGE_INTERVENTION] == "complete"
    assert m04["intervention"] is not None


def test_m04_reaches_the_verify_stage(m04):
    statuses = {entry["step"]: entry["status"] for entry in m04["trace"]}
    assert statuses[STAGE_VERIFY] == "complete"
    assert m04["verification"] is not None


def test_m04_completes_with_verified_simulation(m04):
    assert m04["status"] == "complete"
    assert m04["verification"]["status"] == "PASS"


def test_m03_reaches_the_intervention_stage(m03):
    statuses = {entry["step"]: entry["status"] for entry in m03["trace"]}
    assert statuses[STAGE_INTERVENTION] == "complete"
    assert m03["intervention"] is not None


def test_m03_reaches_the_verify_stage(m03):
    statuses = {entry["step"]: entry["status"] for entry in m03["trace"]}
    assert statuses[STAGE_VERIFY] == "complete"
    assert m03["verification"] is not None


def test_m03_completes_with_verified_simulation(m03):
    assert m03["status"] == "complete"
    assert m03["verification"]["status"] == "PASS"


def test_intervention_matches_the_selected_root_cause(m04, m03):
    assert m04["intervention"]["type"] == m04["root_cause"]["category"]
    assert m03["intervention"]["type"] == m03["root_cause"]["category"]
    assert m04["intervention"]["type"] != m03["intervention"]["type"]


def test_intervention_is_marked_simulated(m04, m03):
    for result in (m04, m03):
        assert result["intervention"]["simulation"] is True


def test_verification_is_marked_simulated(m04, m03):
    for result in (m04, m03):
        assert result["verification"]["simulation"] is True


def test_projected_energy_matches_the_formula(m04):
    intervention = m04["intervention"]
    expected = intervention["baseline_energy"] * (
        1 - intervention["projected_reduction_percent"] / 100
    )
    assert intervention["projected_energy"] == pytest.approx(expected, abs=0.02)


def test_verification_reduction_matches_the_projection(m04):
    intervention = m04["intervention"]
    verification = m04["verification"]
    expected = (
        intervention["baseline_energy"] - intervention["projected_energy"]
    ) / intervention["baseline_energy"] * 100
    assert verification["observed_reduction_percent"] == pytest.approx(
        expected, abs=0.05
    )


def test_baseline_energy_is_the_observed_window_energy(m04):
    """The simulation must start from a real measurement, not an assumption."""
    observed = m04["evidence"]["metrics"]["energy_kwh"]["window_value"]
    assert m04["intervention"]["baseline_energy"] == pytest.approx(observed, abs=0.01)


def test_expected_reduction_is_conservative(m04, m03):
    """Assumed savings must stay below the anomaly they address."""
    for result in (m04, m03):
        excess = result["evidence"]["metrics"]["energy_kwh"]["change_pct"]
        assert 0 < result["intervention"]["projected_reduction_percent"] < excess


def test_verification_compares_against_its_threshold(m04):
    verification = m04["verification"]
    assert verification["status"] == "PASS"
    assert verification["observed_reduction_percent"] >= verification["threshold_percent"]


def test_trace_order_includes_intervention_and_verify(m04):
    steps = [entry["step"] for entry in m04["trace"]]
    assert steps == list(TRACE_STAGES)
    assert STAGE_INTERVENTION in steps
    assert STAGE_VERIFY in steps


def test_trace_never_claims_a_physical_change(m04, m03):
    """The trace must not imply a real machine was modified."""
    for result in (m04, m03):
        entry = _trace_entry(result, STAGE_INTERVENTION)
        summary = entry["summary"].lower()
        assert "simulated" in summary
        assert "no physical change" in summary


def test_intervention_summary_is_labelled_simulated(m04):
    entry = _trace_entry(m04, STAGE_INTERVENTION)
    assert entry["summary"].startswith("SIMULATED")


def test_no_intervention_is_simulated_without_a_root_cause():
    """M05 is inconclusive, so nothing may be projected for it."""
    result = investigate("M05")
    assert result["root_cause"] is None
    assert result["intervention"] is None
    assert result["verification"] is None
    statuses = {entry["step"]: entry["status"] for entry in result["trace"]}
    assert statuses[STAGE_INTERVENTION] == "skipped"
    assert statuses[STAGE_VERIFY] == "skipped"


def test_inconclusive_investigation_is_not_reported_as_complete():
    """A simulation-free run must keep its inconclusive status."""
    assert investigate("M05")["status"] == "inconclusive"


def test_failed_verification_does_not_report_complete(monkeypatch):
    """A FAIL verdict must surface in the status and the VERIFY stage."""

    def _failing(*args, **kwargs):
        return {
            "status": "FAIL",
            "baseline_energy": 415.34,
            "projected_energy": 415.34,
            "observed_reduction_percent": 0.0,
            "expected_reduction_percent": 8.03,
            "threshold_percent": 5.0,
            "simulation": True,
            "reason": "SIMULATED projected reduction of 0.00% is below the "
            "5.00% threshold.",
        }

    monkeypatch.setattr(investigator, "verify_intervention", _failing)
    result = investigate("M04")
    assert result["status"] == "verification_failed"
    statuses = {entry["step"]: entry["status"] for entry in result["trace"]}
    assert statuses[STAGE_VERIFY] == "failed"


def test_rejected_simulation_fails_the_intervention_stage(monkeypatch):
    """An invalid reduction must fail INTERVENTION, not crash the run."""

    def _explode(*args, **kwargs):
        raise ValueError("expected_reduction_pct must be below 100")

    monkeypatch.setattr(investigator, "simulate_intervention", _explode)
    result = investigate("M04")
    assert result["status"] == "failed"
    assert result["intervention"] is None
    statuses = {entry["step"]: entry["status"] for entry in result["trace"]}
    assert statuses[STAGE_INTERVENTION] == "failed"
    assert statuses[STAGE_VERIFY] == "skipped"


def test_cli_shows_simulated_intervention_and_verification(capsys):
    from app.main import main

    assert main(["--investigate", "M04", "--skip-validation"]) == 0
    output = capsys.readouterr().out
    assert "INTERVENTION (SIMULATED)" in output
    assert "VERIFICATION (SIMULATED)" in output
    assert "SIMULATED: yes" in output
    assert "No physical change was made" in output


# --------------------------------------------------------------------------
# Phase 6: operational memory
# --------------------------------------------------------------------------


def test_memory_search_occurs_before_intervention(db):
    steps = [e["step"] for e in investigate("M04", memory_db=db)["trace"]]
    assert steps.index("MEMORY_SEARCH") < steps.index("INTERVENTION")


def test_memory_store_occurs_after_verify(db):
    steps = [e["step"] for e in investigate("M04", memory_db=db)["trace"]]
    assert steps.index("MEMORY_STORE") > steps.index("VERIFY")
    assert steps.index("MEMORY_STORE") < steps.index("COMPLETE")


def test_trace_includes_all_eleven_stages(db):
    steps = [e["step"] for e in investigate("M04", memory_db=db)["trace"]]
    assert steps == list(TRACE_STAGES)
    assert len(steps) == 11


def test_successful_simulated_investigation_is_stored(db):
    result = investigate("M04", memory_db=db)
    assert result["memory_record_id"] is not None
    case = get_investigation(result["memory_record_id"], db_path=db)
    assert case["simulation"] is True
    assert case["verification_status"] == "PASS"


def test_first_investigation_finds_no_memory(db):
    result = investigate("M04", memory_db=db)
    assert result["memory_matches"] == []
    entry = _trace_entry(result, "MEMORY_SEARCH")
    assert entry["summary"] == "No similar historical investigation found."


def test_second_investigation_retrieves_the_first_case(db):
    investigate("M04", memory_db=db)
    second = investigate("M04", memory_db=db)
    assert len(second["memory_matches"]) == 1
    match = second["memory_matches"][0]
    assert match["score"] > 0
    assert match["case"]["root_cause_category"] == "equipment_efficiency"
    assert "same anomaly metric" in " ".join(match["reason"])


def test_current_evidence_remains_authoritative_over_memory(db):
    """A stored M04 equipment case must not steer the M03 conclusion."""
    investigate("M04", memory_db=db)
    m03 = investigate("M03", memory_db=db)
    assert m03["root_cause"]["category"] == "operational_scheduling"
    assert m03["intervention"]["type"] == "operational_scheduling"
    runtime = _hypothesis(m03, HYPOTHESIS_RUNTIME)
    assert runtime["status"] == STATUS_SUPPORTED


def test_memory_does_not_change_hypothesis_statuses(db):
    """Storing cases must leave a repeat investigation's verdicts identical."""
    first = investigate("M04", memory_db=db)
    second = investigate("M04", memory_db=db)
    assert first["hypotheses"] == second["hypotheses"]
    assert first["evidence"] == second["evidence"]


def test_memory_can_be_disabled(db):
    result = investigate("M04", memory_db=False)
    assert result["memory_matches"] == []
    assert result["memory_record_id"] is None
    assert _trace_entry(result, "MEMORY_STORE")["status"] == "skipped"


def test_inconclusive_investigation_is_not_stored(db):
    result = investigate("M05", memory_db=db)
    assert result["memory_record_id"] is None
    assert list_investigations(db_path=db) == []


def test_cli_shows_memory_sections(capsys):
    from app.main import main

    assert main(["--investigate", "M04", "--skip-validation"]) == 0
    output = capsys.readouterr().out
    assert "MEMORY SEARCH" in output
    assert "MEMORY STORE" in output


def test_cli_shows_no_raw_sql(capsys):
    from app.main import main

    main(["--investigate", "M04", "--skip-validation"])
    output = capsys.readouterr().out
    assert "SELECT " not in output
    assert "INSERT INTO" not in output