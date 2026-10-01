"""Tests for the TypeSafe (System One) reasoning adapter.

The adapter is exercised with an injectable transport, so no network access and
no credential are required. These tests pin the contract: one Noul question per
candidate explanation, code-owned thresholds, and reconciliation that keeps the
deterministic verdicts authoritative.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.investigator import (
    HYPOTHESIS_EQUIPMENT,
    HYPOTHESIS_HISTORICAL,
    HYPOTHESIS_MAINTENANCE,
    HYPOTHESIS_PRODUCTION,
    HYPOTHESIS_RUNTIME,
    STAGE_LLM_REASONING,
    investigate,
)
from app.llm.adapter import LLMAdapter, LLMError, get_adapter, reconcile_reasoning
from app.llm.typesafe import CANDIDATES, TypeSafeAdapter


def probabilities(**overrides):
    """All candidate probabilities at an indecisive 0.5, with overrides."""
    base = {candidate["id"]: 0.5 for candidate in CANDIDATES}
    base.update(overrides)
    return base


def answers_for(values):
    """Build a System One-shaped response from per-candidate probabilities."""
    return {
        "model": "jev-test",
        "answers": {
            name: {"type": "noul", "noul": value} for name, value in values.items()
        },
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def capturing_transport(values):
    """A transport that records its payload and returns fixed judgments."""
    calls = []

    def transport(payload):
        calls.append(payload)
        return answers_for(values)

    return transport, calls


# --------------------------------------------------------------------------
# 1: configuration and availability
# --------------------------------------------------------------------------


def test_adapter_requires_a_credential(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert TypeSafeAdapter().is_available() is False


def test_adapter_is_available_with_a_key():
    adapter = TypeSafeAdapter(
        api_key="k", transport=lambda p: answers_for(probabilities())
    )
    assert adapter.is_available() is True


def test_adapter_reads_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-test-model")
    adapter = TypeSafeAdapter()
    assert adapter.is_available() is True
    assert adapter.describe()["model"] == "jev-test-model"


def test_api_key_is_never_exposed_by_describe():
    adapter = TypeSafeAdapter(
        api_key="super-secret", transport=lambda p: answers_for(probabilities())
    )
    assert "super-secret" not in json.dumps(adapter.describe())


def test_provider_name_comes_from_the_base_url():
    adapter = TypeSafeAdapter(
        api_key="k", transport=lambda p: answers_for(probabilities())
    )
    assert adapter.provider_name() == "api.typesafe.ai"


def test_no_hardcoded_secret_in_adapter_source():
    from app.llm import typesafe as typesafe_module

    source = Path(typesafe_module.__file__).read_text(encoding="utf-8")
    assert "super-secret" not in source
    assert "sk-" not in source


def test_get_adapter_prefers_typesafe_when_configured(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert isinstance(get_adapter(), TypeSafeAdapter)


def test_get_adapter_falls_back_to_the_openai_adapter(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    adapter = get_adapter()
    assert isinstance(adapter, LLMAdapter)
    assert adapter.is_available() is False


# --------------------------------------------------------------------------
# 2: the request shape
# --------------------------------------------------------------------------


def test_reason_asks_one_noul_question_per_candidate():
    transport, calls = capturing_transport(probabilities())
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    adapter.reason({"anomaly": {"metric": "intensity"}})

    payload = calls[0]
    assert payload["model"] == "jev-latest"
    assert set(payload["questions"]) == {c["id"] for c in CANDIDATES}
    for candidate in CANDIDATES:
        question = payload["questions"][candidate["id"]]
        assert question["type"] == "noul"
        assert question["instructions"] == candidate["instructions"]
        assert question["criteria"] == candidate["criteria"]


def test_question_ids_match_the_investigator_hypotheses():
    assert {c["id"] for c in CANDIDATES} == {
        HYPOTHESIS_PRODUCTION,
        HYPOTHESIS_RUNTIME,
        HYPOTHESIS_EQUIPMENT,
        HYPOTHESIS_MAINTENANCE,
        HYPOTHESIS_HISTORICAL,
    }


def test_state_is_the_evidence_passed_through_unchanged():
    transport, calls = capturing_transport(probabilities())
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    evidence = {"anomaly": {"metric": "intensity"}, "production": {"change_pct": 1.2}}
    adapter.reason(evidence)
    assert calls[0]["state"] == evidence


# --------------------------------------------------------------------------
# 3: thresholds are code decisions
# --------------------------------------------------------------------------


def test_probabilities_map_to_statuses_at_default_thresholds():
    transport, _ = capturing_transport(
        probabilities(
            equipment_efficiency_degradation=0.82,
            runtime_scheduling_increase=0.04,
        )
    )
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    reasoning = adapter.reason({})
    by_name = {a["name"]: a for a in reasoning["hypothesis_assessment"]}

    assert by_name[HYPOTHESIS_EQUIPMENT]["status"] == "supported"
    assert by_name[HYPOTHESIS_RUNTIME]["status"] == "rejected"
    assert by_name[HYPOTHESIS_PRODUCTION]["status"] == "inconclusive"
    assert "0.82" in by_name[HYPOTHESIS_EQUIPMENT]["reason"]
    assert "0.04" in by_name[HYPOTHESIS_RUNTIME]["reason"]


def test_threshold_boundaries_are_inclusive():
    transport, _ = capturing_transport(
        probabilities(
            equipment_efficiency_degradation=0.7,
            runtime_scheduling_increase=0.3,
        )
    )
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    by_name = {
        a["name"]: a for a in adapter.reason({})["hypothesis_assessment"]
    }
    assert by_name[HYPOTHESIS_EQUIPMENT]["status"] == "supported"
    assert by_name[HYPOTHESIS_RUNTIME]["status"] == "rejected"


def test_thresholds_can_be_overridden_by_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_SUPPORT_THRESHOLD", "0.9")
    monkeypatch.setenv("TYPESAFE_REJECT_THRESHOLD", "0.1")
    transport, _ = capturing_transport(
        probabilities(equipment_efficiency_degradation=0.82)
    )
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    by_name = {
        a["name"]: a for a in adapter.reason({})["hypothesis_assessment"]
    }
    # 0.82 no longer reaches the raised support threshold.
    assert by_name[HYPOTHESIS_EQUIPMENT]["status"] == "inconclusive"


def test_invalid_threshold_environment_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TYPESAFE_SUPPORT_THRESHOLD", "banana")
    transport, _ = capturing_transport(probabilities())
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    statuses = {
        a["status"] for a in adapter.reason({})["hypothesis_assessment"]
    }
    assert statuses == {"inconclusive"}


def test_deciveness_is_code_derived_and_bounded():
    transport, _ = capturing_transport(
        probabilities(equipment_efficiency_degradation=0.82)
    )
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    entry = next(
        a
        for a in adapter.reason({})["hypothesis_assessment"]
        if a["name"] == HYPOTHESIS_EQUIPMENT
    )
    assert entry["confidence"] == pytest.approx(0.64)


# --------------------------------------------------------------------------
# 4: composition and failure handling
# --------------------------------------------------------------------------


def test_summary_and_root_cause_are_code_composed():
    transport, _ = capturing_transport(
        probabilities(equipment_efficiency_degradation=0.82)
    )
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    reasoning = adapter.reason({})

    assert (
        "equipment_efficiency_degradation: supported (noul 0.82)"
        in reasoning["reasoning_summary"]
    )
    assert "deterministic" in reasoning["reasoning_summary"]
    assert (
        "equipment_efficiency_degradation"
        in reasoning["root_cause_explanation"]
    )
    assert "0.82" in reasoning["root_cause_explanation"]


def test_no_supported_candidate_yields_a_refusal_style_explanation():
    transport, _ = capturing_transport(
        probabilities(equipment_efficiency_degradation=0.55)
    )
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    reasoning = adapter.reason({})
    assert (
        "No candidate explanation reached the support threshold"
        in reasoning["root_cause_explanation"]
    )


def test_missing_answers_are_skipped_not_invented():
    response = answers_for(probabilities(equipment_efficiency_degradation=0.82))
    del response["answers"][HYPOTHESIS_RUNTIME]
    adapter = TypeSafeAdapter(api_key="k", transport=lambda p: response)
    reasoning = adapter.reason({})
    names = {a["name"] for a in reasoning["hypothesis_assessment"]}
    assert HYPOTHESIS_RUNTIME not in names
    assert HYPOTHESIS_EQUIPMENT in names


def test_all_missing_answers_raise_llm_error():
    adapter = TypeSafeAdapter(api_key="k", transport=lambda p: {"answers": {}})
    with pytest.raises(LLMError):
        adapter.reason({})


def test_malformed_noul_values_are_ignored():
    adapter = TypeSafeAdapter(
        api_key="k",
        transport=lambda p: {
            "answers": {
                HYPOTHESIS_EQUIPMENT: {"type": "noul", "noul": "banana"},
                HYPOTHESIS_RUNTIME: {"type": "noul", "noul": 1.5},
                HYPOTHESIS_PRODUCTION: {"type": "choice", "choice": "x"},
            }
        },
    )
    with pytest.raises(LLMError):
        adapter.reason({})


def test_response_missing_answers_key_raises_llm_error():
    adapter = TypeSafeAdapter(
        api_key="k", transport=lambda p: {"model": "jev-test"}
    )
    with pytest.raises(LLMError):
        adapter.reason({})


def test_unavailable_adapter_raises_on_reason(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(LLMError):
        TypeSafeAdapter().reason({})


def test_transport_failure_becomes_llm_error():
    def broken_transport(payload):
        raise OSError("network down")

    adapter = TypeSafeAdapter(api_key="k", transport=broken_transport)
    with pytest.raises(LLMError):
        adapter.reason({})


# --------------------------------------------------------------------------
# 5: reconciliation and the full investigation
# --------------------------------------------------------------------------


def test_reconcile_keeps_deterministic_verdicts_and_records_conflicts():
    hypotheses = [
        {"name": HYPOTHESIS_EQUIPMENT, "status": "supported"},
        {"name": HYPOTHESIS_RUNTIME, "status": "rejected"},
        {"name": HYPOTHESIS_PRODUCTION, "status": "inconclusive"},
        {"name": HYPOTHESIS_MAINTENANCE, "status": "inconclusive"},
        {"name": HYPOTHESIS_HISTORICAL, "status": "inconclusive"},
    ]
    transport, _ = capturing_transport(
        probabilities(
            equipment_efficiency_degradation=0.9,
            runtime_scheduling_increase=0.9,
        )
    )
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    merged = reconcile_reasoning(hypotheses, adapter.reason({}), {})

    assert hypotheses[1]["status"] == "rejected"  # untouched
    assert [c["hypothesis"] for c in merged["conflicts"]] == [HYPOTHESIS_RUNTIME]
    assert merged["conflicts"][0]["resolution"] == "kept_deterministic"
    assert merged["authoritative_source"] == "deterministic"


def test_investigation_with_typesafe_adapter_stays_deterministic():
    transport, calls = capturing_transport(
        probabilities(
            equipment_efficiency_degradation=0.9,
            runtime_scheduling_increase=0.9,
        )
    )
    adapter = TypeSafeAdapter(api_key="k", transport=transport)
    result = investigate("M04", adapter=adapter, memory_db=False)

    entry = next(
        e for e in result["trace"] if e["step"] == STAGE_LLM_REASONING
    )
    assert entry["status"] == "complete"
    assert entry["provider"] == "api.typesafe.ai"
    assert entry["model"] == "jev-latest"

    # Deterministic outcomes are untouched by the typed judgments.
    assert result["status"] == "complete"
    assert result["root_cause"]["category"] == "equipment_efficiency"

    # The runtime judgment contradicts the deterministically rejected runtime
    # hypothesis, so a conflict is recorded rather than obeyed.
    assert result["llm_reasoning"]["available"] is True
    assert result["llm_reasoning"]["conflicts"]
    runtime_conflict = next(
        c
        for c in result["llm_reasoning"]["conflicts"]
        if c["hypothesis"] == HYPOTHESIS_RUNTIME
    )
    assert runtime_conflict["model_status"] == "supported"
    assert runtime_conflict["resolution"] == "kept_deterministic"

    # One request, one question set - and the state is the evidence contract.
    assert len(calls) == 1
    assert calls[0]["state"]["machine"]["machine_id"] == "M04"


def test_investigation_is_deterministic_with_typesafe_adapter():
    transport, _ = capturing_transport(probabilities())
    one = investigate(
        "M04",
        adapter=TypeSafeAdapter(api_key="k", transport=transport),
        memory_db=False,
    )
    two = investigate(
        "M04",
        adapter=TypeSafeAdapter(api_key="k", transport=transport),
        memory_db=False,
    )
    assert one["hypotheses"] == two["hypotheses"]
