"""Tests for multi-provider reasoning configuration (OpenAI and Groq).

The OpenAI-compatible adapter learns provider presets: ``OPENAI_API_KEY`` and
``GROQ_API_KEY``, selectable with ``LLM_PROVIDER``, with an optional
``LLM_FALLBACK_PROVIDER`` that is only tried when the primary fails. All tests
use injectable transports or stub adapters, so no network access and no real
credential are required.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.investigator import STAGE_LLM_REASONING, investigate
from app.llm.adapter import (
    FallbackAdapter,
    LLMAdapter,
    LLMError,
    get_adapter,
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


def stub_adapter(available=True, raises=None, payload=None, provider="stub"):
    """Minimal adapter stand-in matching the shared reasoning interface."""

    class StubAdapter:
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return available

        def describe(self):
            return {
                "provider": provider,
                "model": "stub-model",
                "configured": available,
            }

        def provider_name(self):
            return provider

        def reason(self, evidence):
            self.calls += 1
            if raises is not None:
                raise raises
            return dict(payload or VALID_REASONING)

    return StubAdapter()


# --------------------------------------------------------------------------
# 1: provider presets and selection
# --------------------------------------------------------------------------


def test_openai_key_selects_the_openai_preset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    adapter = LLMAdapter()
    assert adapter.is_available() is True
    description = adapter.describe()
    assert description["provider"] == "api.openai.com"
    assert description["base_url"] == "https://api.openai.com/v1"
    assert description["model"] == "gpt-4o-mini"


def test_groq_key_selects_the_groq_preset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    adapter = LLMAdapter()
    assert adapter.is_available() is True
    description = adapter.describe()
    assert description["provider"] == "api.groq.com"
    assert description["base_url"] == "https://api.groq.com/openai/v1"
    assert description["model"] == "openai/gpt-oss-120b"


def test_llm_provider_overrides_key_inference(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    adapter = LLMAdapter()
    assert adapter.provider_name() == "api.groq.com"


def test_llm_model_and_base_url_override_the_preset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example/v1")
    adapter = LLMAdapter()
    description = adapter.describe()
    assert description["provider"] == "proxy.example"
    assert description["model"] == "custom-model"


def test_generic_llm_api_key_still_works(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    adapter = LLMAdapter()
    assert adapter.is_available() is True
    description = adapter.describe()
    assert description["base_url"] == "https://api.openai.com/v1"
    assert description["model"] == "gpt-4o-mini"


def test_explicit_provider_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    adapter = LLMAdapter(provider="groq", api_key="k")
    assert adapter.provider_name() == "api.groq.com"


def test_unknown_provider_falls_back_to_the_generic_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "banana")
    monkeypatch.setenv("LLM_API_KEY", "k")
    adapter = LLMAdapter()
    assert adapter.is_available() is True
    assert adapter.provider_name() == "api.openai.com"


def test_no_keys_means_unavailable(monkeypatch):
    adapter = LLMAdapter()
    assert adapter.is_available() is False


# --------------------------------------------------------------------------
# 2: the fallback chain
# --------------------------------------------------------------------------


def test_fallback_uses_the_second_adapter_when_the_first_fails():
    first = stub_adapter(raises=LLMError("rate limited"), provider="api.openai.com")
    second = stub_adapter(payload=VALID_REASONING, provider="api.groq.com")
    adapter = FallbackAdapter([first, second])

    result = adapter.reason({})
    assert result["reasoning_summary"] == VALID_REASONING["reasoning_summary"]
    assert first.calls == 1
    assert second.calls == 1


def test_fallback_short_circuits_on_the_first_success():
    first = stub_adapter(provider="api.openai.com")
    second = stub_adapter(provider="api.groq.com")
    adapter = FallbackAdapter([first, second])

    adapter.reason({})
    assert first.calls == 1
    assert second.calls == 0


def test_fallback_skips_unavailable_adapters():
    unavailable = stub_adapter(available=False, raises=LLMError("must not run"))
    second = stub_adapter(provider="api.groq.com")
    adapter = FallbackAdapter([unavailable, second])

    assert adapter.is_available() is True
    adapter.reason({})
    assert unavailable.calls == 0
    assert second.calls == 1


def test_fallback_reports_both_failures():
    first = stub_adapter(raises=LLMError("openai down"), provider="api.openai.com")
    second = stub_adapter(raises=LLMError("groq down"), provider="api.groq.com")
    adapter = FallbackAdapter([first, second])

    with pytest.raises(LLMError) as excinfo:
        adapter.reason({})
    message = str(excinfo.value)
    assert "api.openai.com" in message
    assert "api.groq.com" in message


def test_fallback_unavailable_when_nothing_is_configured():
    adapter = FallbackAdapter(
        [stub_adapter(available=False), stub_adapter(available=False)]
    )
    assert adapter.is_available() is False
    with pytest.raises(LLMError):
        adapter.reason({})


def test_fallback_describe_names_the_backup_provider():
    adapter = FallbackAdapter(
        [stub_adapter(provider="api.openai.com"), stub_adapter(provider="api.groq.com")]
    )
    description = adapter.describe()
    assert description["provider"] == "api.openai.com"
    assert description["fallback"] == ["api.groq.com"]


def test_fallback_never_exposes_keys():
    transport = lambda payload: {"choices": [{"message": {"content": "{}"}}]}
    adapter = FallbackAdapter(
        [
            LLMAdapter(api_key="super-secret", transport=transport),
            LLMAdapter(api_key="super-secret", provider="groq", transport=transport),
        ]
    )
    assert "super-secret" not in json.dumps(adapter.describe())


# --------------------------------------------------------------------------
# 3: get_adapter wiring
# --------------------------------------------------------------------------


def test_get_adapter_single_provider_stays_a_plain_adapter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    adapter = get_adapter()
    assert isinstance(adapter, LLMAdapter)
    assert adapter.provider_name() == "api.openai.com"


def test_get_adapter_infers_groq_when_only_its_key_is_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    adapter = get_adapter()
    assert isinstance(adapter, LLMAdapter)
    assert adapter.provider_name() == "api.groq.com"


def test_get_adapter_builds_the_fallback_chain(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
    adapter = get_adapter()
    assert isinstance(adapter, FallbackAdapter)
    description = adapter.describe()
    assert description["provider"] == "api.openai.com"
    assert description["fallback"] == ["api.groq.com"]


def test_get_adapter_uses_the_fallback_when_the_primary_key_is_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
    adapter = get_adapter()
    assert isinstance(adapter, FallbackAdapter)
    assert adapter.is_available() is True
    assert adapter.provider_name() == "api.groq.com"


def test_get_adapter_collapses_duplicate_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
    adapter = get_adapter()
    assert isinstance(adapter, LLMAdapter)


def test_get_adapter_no_keys_is_an_unconfigured_adapter(monkeypatch):
    adapter = get_adapter()
    assert isinstance(adapter, LLMAdapter)
    assert adapter.is_available() is False


# --------------------------------------------------------------------------
# 4: full investigations over a fallback chain
# --------------------------------------------------------------------------


def test_investigation_uses_the_fallback_when_the_primary_fails():
    first = stub_adapter(raises=LLMError("rate limited"), provider="api.openai.com")
    second = stub_adapter(provider="api.groq.com")
    result = investigate(
        "M04", adapter=FallbackAdapter([first, second]), memory_db=False
    )

    entry = next(e for e in result["trace"] if e["step"] == STAGE_LLM_REASONING)
    assert entry["status"] == "complete"
    assert result["llm_reasoning"]["available"] is True
    assert second.calls == 1


def test_investigation_survives_all_providers_failing():
    adapter = FallbackAdapter(
        [
            stub_adapter(raises=LLMError("openai down"), provider="api.openai.com"),
            stub_adapter(raises=LLMError("groq down"), provider="api.groq.com"),
        ]
    )
    result = investigate("M04", adapter=adapter, memory_db=False)

    entry = next(e for e in result["trace"] if e["step"] == STAGE_LLM_REASONING)
    assert entry["status"] == "failed"
    assert result["llm_reasoning"]["available"] is False
    # The deterministic outcome is untouched.
    assert result["status"] == "complete"
    assert result["root_cause"]["category"] == "equipment_efficiency"
