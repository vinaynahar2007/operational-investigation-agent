import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.investigator import STAGE_LLM_REASONING, investigate
from app.llm.adapter import (
    LLMAdapter,
    LLMError,
    build_evidence_contract,
    parse_json_object,
    reconcile_reasoning,
    validate_reasoning,
)

VALID_REASONING = {
    "hypothesis_assessment": [
        {
            "name": "equipment_efficiency_degradation",
            "status": "supported",
            "reason": "Energy per runtime hour rose while runtime stayed flat.",
            "confidence": 0.8,
        }
    ],
    "root_cause_explanation": "Equipment is drawing more power for the same time.",
    "recommended_intervention_explanation": "SIMULATED inspection and restore.",
    "reasoning_summary": "Load-side condition is the best supported explanation.",
}


def mock_response(content):
    """Build a provider-shaped response wrapping model text."""
    return {"choices": [{"message": {"content": content}}]}


def stub_adapter(available=True, payload=None, raises=None):
    """Minimal adapter stand-in so investigator tests need no HTTP."""

    class StubAdapter:
        def __init__(self):
            self.calls = []

        def is_available(self):
            return available

        def describe(self):
            return {
                "provider": "stub",
                "model": "stub-model",
                "configured": available,
            }

        def reason(self, evidence):
            self.calls.append(evidence)
            if raises is not None:
                raise raises
            return dict(payload or VALID_REASONING)

    return StubAdapter()


def _llm_entry(result):
    return next(e for e in result["trace"] if e["step"] == STAGE_LLM_REASONING)


# --------------------------------------------------------------------------
# 1-2: adapter lifecycle without credentials
# --------------------------------------------------------------------------


def test_adapter_instantiates_without_credentials(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    adapter = LLMAdapter()
    assert adapter is not None
    assert adapter.is_available() is False


def test_unavailable_adapter_reports_unavailable(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert LLMAdapter().is_available() is False


def test_adapter_is_available_with_a_key():
    adapter = LLMAdapter(api_key="k", transport=lambda p: mock_response("{}"))
    assert adapter.is_available() is True


def test_unavailable_adapter_raises_on_reason(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(LLMError):
        LLMAdapter().reason({})


def test_api_key_is_never_exposed_by_describe():
    adapter = LLMAdapter(api_key="super-secret", transport=lambda p: mock_response("{}"))
    assert "super-secret" not in json.dumps(adapter.describe())


def test_no_hardcoded_secret_in_adapter_source():
    from app.llm import adapter as adapter_module

    source = Path(adapter_module.__file__).read_text(encoding="utf-8")
    assert "super-secret" not in source
    assert "sk-" not in source


# --------------------------------------------------------------------------
# 3: deterministic fallback still works
# --------------------------------------------------------------------------


def test_deterministic_fallback_still_completes(monkeypatch):
    """No credentials: the investigation must complete deterministically."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    result = investigate("M04")
    assert result["status"] == "complete"
    assert result["root_cause"]["category"] == "equipment_efficiency"
    assert result["intervention"] is not None
    assert result["verification"] is not None


def test_fallback_records_unavailable_reasoning(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    result = investigate("M04")
    assert result["llm_reasoning"]["available"] is False
    assert "deterministic" in result["llm_reasoning"]["reason"]


def test_llm_trace_is_skipped_when_unavailable(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    entry = _llm_entry(investigate("M04"))
    assert entry["status"] == "skipped"
    assert entry["summary"] == "LLM unavailable; deterministic reasoning used."


def test_unavailable_adapter_argument_also_falls_back():
    result = investigate("M04", adapter=stub_adapter(available=False))
    assert _llm_entry(result)["status"] == "skipped"
    assert result["status"] == "complete"


# --------------------------------------------------------------------------
# 4: malformed model output falls back instead of crashing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    ["this is not json at all", "{ broken json", "", "[1, 2, 3]"],
)
def test_malformed_model_json_triggers_fallback(content):
    adapter = LLMAdapter(api_key="k", transport=lambda p: mock_response(content))
    result = investigate("M04", adapter=adapter)
    assert result["status"] == "complete"
    assert _llm_entry(result)["status"] == "failed"
    assert result["llm_reasoning"]["available"] is False


def test_missing_required_field_triggers_fallback():
    incomplete = {"hypothesis_assessment": [{"name": "x", "status": "supported"}]}
    adapter = LLMAdapter(
        api_key="k", transport=lambda p: mock_response(json.dumps(incomplete))
    )
    result = investigate("M04", adapter=adapter)
    assert _llm_entry(result)["status"] == "failed"
    assert result["root_cause"]["category"] == "equipment_efficiency"


def test_transport_exception_triggers_fallback():
    def _boom(_payload):
        raise RuntimeError("connection reset")

    adapter = LLMAdapter(api_key="k", transport=_boom)
    result = investigate("M04", adapter=adapter)
    assert _llm_entry(result)["status"] == "failed"
    assert result["status"] == "complete"


def test_response_without_choices_triggers_fallback():
    adapter = LLMAdapter(api_key="k", transport=lambda p: {"unexpected": True})
    result = investigate("M04", adapter=adapter)
    assert _llm_entry(result)["status"] == "failed"


def test_model_raising_unexpected_exception_still_falls_back():
    adapter = stub_adapter(raises=RuntimeError("boom"))
    result = investigate("M04", adapter=adapter)
    assert _llm_entry(result)["status"] == "failed"
    assert result["status"] == "complete"


def test_fenced_json_is_accepted():
    fence = chr(96) * 3
    text = fence + "json\n" + json.dumps(VALID_REASONING) + "\n" + fence
    adapter = LLMAdapter(api_key="k", transport=lambda p: mock_response(text))
    result = investigate("M04", adapter=adapter)
    assert _llm_entry(result)["status"] == "complete"


# --------------------------------------------------------------------------
# 5: model output cannot replace deterministic numeric evidence
# --------------------------------------------------------------------------


def test_model_cannot_change_the_root_cause():
    """A model claiming the wrong category must not steer the result."""
    payload = dict(VALID_REASONING)
    payload["root_cause_explanation"] = "Runtime scheduling caused this."
    result = investigate("M04", adapter=stub_adapter(payload=payload))
    assert result["root_cause"]["category"] == "equipment_efficiency"


def test_contradicting_model_status_is_recorded_as_a_conflict():
    """M04 runtime is flat; a model calling it 'supported' must be flagged."""
    payload = {
        "hypothesis_assessment": [
            {
                "name": "runtime_scheduling_increase",
                "status": "supported",
                "reason": "Claims runtime rose substantially.",
                "confidence": 0.9,
            }
        ],
        "root_cause_explanation": "x",
        "recommended_intervention_explanation": "y",
        "reasoning_summary": "z",
    }
    result = investigate("M04", adapter=stub_adapter(payload=payload))
    conflicts = result["llm_reasoning"]["conflicts"]
    assert conflicts
    conflict = conflicts[0]
    assert conflict["hypothesis"] == "runtime_scheduling_increase"
    assert conflict["deterministic_status"] == "rejected"
    assert conflict["model_status"] == "supported"
    assert conflict["resolution"] == "kept_deterministic"
    runtime = next(
        h for h in result["hypotheses"] if h["name"] == "runtime_scheduling_increase"
    )
    assert runtime["status"] == "rejected"


def test_model_agreement_produces_no_conflicts():
    result = investigate("M04", adapter=stub_adapter())
    assert result["llm_reasoning"]["conflicts"] == []


def test_unknown_hypothesis_from_model_is_ignored():
    payload = {
        "hypothesis_assessment": [
            {"name": "aliens_disabled_the_boiler", "status": "supported", "confidence": 1.0}
        ],
        "root_cause_explanation": "x",
        "recommended_intervention_explanation": "y",
        "reasoning_summary": "z",
    }
    result = investigate("M04", adapter=stub_adapter(payload=payload))
    conflicts = result["llm_reasoning"]["conflicts"]
    assert conflicts[0]["resolution"] == "ignored"
    assert result["root_cause"]["category"] == "equipment_efficiency"


def test_model_never_alters_deterministic_measurements():
    """Evidence numbers must be identical with and without the model."""
    without = investigate("M04")
    with_llm = investigate("M04", adapter=stub_adapter())
    assert without["evidence"]["metrics"] == with_llm["evidence"]["metrics"]
    assert without["anomaly"] == with_llm["anomaly"]
    assert (
        without["intervention"]["projected_energy"]
        == with_llm["intervention"]["projected_energy"]
    )
    assert without["verification"] == with_llm["verification"]


def test_model_text_cannot_replace_numeric_evidence():
    """Prose claiming huge numbers must not become a measurement."""
    payload = dict(VALID_REASONING)
    payload["root_cause_explanation"] = "Energy increased by 400%."
    result = investigate("M04", adapter=stub_adapter(payload=payload))
    change = result["evidence"]["metrics"]["energy_kwh"]["change_pct"]
    assert change == pytest.approx(16.06, abs=0.1)
    assert result["evidence"]["metrics"] == investigate("M04")["evidence"]["metrics"]


# --------------------------------------------------------------------------
# 6-7: conclusions still come from the deterministic pipeline
# --------------------------------------------------------------------------


def test_m04_still_identifies_equipment_efficiency_with_llm():
    result = investigate("M04", adapter=stub_adapter())
    assert result["root_cause"]["category"] == "equipment_efficiency"
    assert _llm_entry(result)["status"] == "complete"


def test_m03_still_identifies_operational_scheduling_with_llm():
    result = investigate("M03", adapter=stub_adapter())
    assert result["root_cause"]["category"] == "operational_scheduling"


def test_m03_works_with_no_llm_at_all(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    result = investigate("M03")
    assert result["status"] == "complete"
    assert result["root_cause"]["category"] == "operational_scheduling"
    assert result["verification"]["status"] == "PASS"


def test_llm_does_not_prevent_intervention_and_verification():
    result = investigate("M04", adapter=stub_adapter())
    assert result["intervention"]["simulation"] is True
    assert result["verification"]["simulation"] is True


def test_authoritative_source_is_always_deterministic():
    assert investigate("M04")["llm_reasoning"]["authoritative_source"] == "deterministic"
    stubbed = investigate("M04", adapter=stub_adapter())
    assert stubbed["llm_reasoning"]["authoritative_source"] == "deterministic"


# --------------------------------------------------------------------------
# 8: trace reporting
# --------------------------------------------------------------------------


def test_trace_reports_complete_with_provider_and_model():
    entry = _llm_entry(investigate("M04", adapter=stub_adapter()))
    assert entry["status"] == "complete"
    assert entry["provider"] == "stub"
    assert entry["model"] == "stub-model"


def test_trace_reports_skipped_for_unavailable_llm(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    entry = _llm_entry(investigate("M04"))
    assert entry["status"] == "skipped"
    assert "provider" in entry and "model" in entry


def test_trace_reports_failed_for_broken_model_output():
    adapter = LLMAdapter(api_key="k", transport=lambda p: mock_response("nope"))
    entry = _llm_entry(investigate("M04", adapter=adapter))
    assert entry["status"] == "failed"


def test_llm_stage_sits_between_hypotheses_and_evaluate():
    steps = [e["step"] for e in investigate("M04", adapter=stub_adapter())["trace"]]
    assert steps.index("HYPOTHESES") < steps.index(STAGE_LLM_REASONING)
    assert steps.index(STAGE_LLM_REASONING) < steps.index("EVALUATE")


def test_trace_has_all_eleven_stages():
    steps = [e["step"] for e in investigate("M04", adapter=stub_adapter())["trace"]]
    assert steps == [
        "DETECT",
        "EVIDENCE",
        "HYPOTHESES",
        STAGE_LLM_REASONING,
        "EVALUATE",
        "ROOT_CAUSE",
        "MEMORY_SEARCH",
        "INTERVENTION",
        "VERIFY",
        "MEMORY_STORE",
        "COMPLETE",
    ]


def test_investigation_is_deterministic_with_llm():
    assert investigate("M04", adapter=stub_adapter(), memory_db=False) == investigate(
        "M04", adapter=stub_adapter(), memory_db=False
    )


# --------------------------------------------------------------------------
# parsing / validation / contract helpers
# --------------------------------------------------------------------------


def test_parse_json_object_handles_fenced_and_bare():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('prose {"a": 1} trailing') == {"a": 1}
    assert parse_json_object("nope") is None
    assert parse_json_object(None) is None


def test_validate_reasoning_rejects_missing_fields():
    with pytest.raises(LLMError):
        validate_reasoning({"hypothesis_assessment": []})


def test_validate_reasoning_rejects_empty_assessment():
    with pytest.raises(LLMError):
        validate_reasoning(
            {
                "hypothesis_assessment": [],
                "root_cause_explanation": "a",
                "recommended_intervention_explanation": "b",
                "reasoning_summary": "c",
            }
        )


def test_validate_reasoning_coerces_unknown_status_and_clamps_confidence():
    cleaned = validate_reasoning(
        {
            "hypothesis_assessment": [
                {"name": "x", "status": "banana", "confidence": 42}
            ],
            "root_cause_explanation": "a",
            "recommended_intervention_explanation": "b",
            "reasoning_summary": "c",
        }
    )
    entry = cleaned["hypothesis_assessment"][0]
    assert entry["status"] == "inconclusive"
    assert entry["confidence"] == 1.0


def test_evidence_contract_contains_only_computed_values():
    contract = build_evidence_contract(investigate("M04")["evidence"])
    assert set(contract) == {
        "machine",
        "anomaly",
        "energy_kwh",
        "runtime_h",
        "intensity",
        "efficiency_factor",
        "production",
        "maintenance_records",
        "technician_notes",
        "historical_incidents",
    }
    assert contract["runtime_h"]["change_pct"] == pytest.approx(0.41, abs=0.05)
    assert contract["intensity"]["change_pct"] == pytest.approx(15.58, abs=0.1)


def test_reconcile_keeps_deterministic_verdicts():
    hypotheses = [
        {"name": "a", "status": "supported"},
        {"name": "b", "status": "rejected"},
    ]
    reasoning = {
        "hypothesis_assessment": [
            {"name": "a", "status": "supported"},
            {"name": "b", "status": "supported"},
        ],
        "root_cause_explanation": "x",
        "recommended_intervention_explanation": "y",
        "reasoning_summary": "z",
    }
    merged = reconcile_reasoning(hypotheses, reasoning, {})
    assert hypotheses[1]["status"] == "rejected"  # untouched
    assert len(merged["conflicts"]) == 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_shows_llm_section_when_unavailable(monkeypatch, capsys):
    from app.main import main

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert main(["--investigate", "M04", "--skip-validation"]) == 0
    output = capsys.readouterr().out
    assert "LLM REASONING" in output
    assert "SKIPPED" in output
    assert "deterministic investigator" in output