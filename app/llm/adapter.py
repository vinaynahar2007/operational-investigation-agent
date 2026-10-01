"""Optional LLM reasoning adapter.

This adapter is a thin, provider-agnostic reasoning layer. It never computes
anything: percentages, baselines, ratios, projected savings and verification
results are produced by the deterministic tools, and this module only sends
those already-computed values to a model and parses prose back.

Authority
---------
The model is not the source of numerical truth. It may interpret evidence,
compare hypotheses and explain causes. Any numeric claim it makes is ignored;
the deterministic measurements in the evidence contract are authoritative. If
the model's interpretation contradicts those measurements, the contradiction
is recorded rather than believed.

Provider configuration comes entirely from the environment:

    LLM_PROVIDER           openai | groq (default: the first provider with a key)
    OPENAI_API_KEY         credential for OpenAI (never logged, never echoed)
    GROQ_API_KEY           credential for Groq (never logged, never echoed)
    LLM_API_KEY            generic credential for any OpenAI-compatible endpoint
    LLM_MODEL              model identifier (overrides the provider default)
    LLM_BASE_URL           API base URL (overrides the provider default)
    LLM_FALLBACK_PROVIDER  second provider, tried only when the first fails

When no credential is configured the adapter reports itself unavailable and the
investigator simply continues with deterministic reasoning.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

#: Fields the model must return for its output to be usable.
REQUIRED_FIELDS = (
    "hypothesis_assessment",
    "root_cause_explanation",
    "recommended_intervention_explanation",
    "reasoning_summary",
)

VALID_STATUSES = ("supported", "rejected", "inconclusive")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 30.0

#: OpenAI-compatible providers with a known base URL, default model and the
#: environment variable that carries their credential. Any other
#: OpenAI-compatible endpoint remains usable through LLM_BASE_URL.
PROVIDERS = {
    "openai": {
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "key_env": "OPENAI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "key_env": "GROQ_API_KEY",
    },
}


def _configured_provider() -> str | None:
    """Provider chosen by LLM_PROVIDER, or the first provider with a key."""
    requested = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if requested in PROVIDERS:
        return requested
    for name, preset in PROVIDERS.items():
        if os.environ.get(preset["key_env"], "").strip():
            return name
    return None


def _fallback_provider() -> str | None:
    """Provider chosen by LLM_FALLBACK_PROVIDER, if it is a known one."""
    requested = os.environ.get("LLM_FALLBACK_PROVIDER", "").strip().lower()
    return requested if requested in PROVIDERS else None

SYSTEM_PROMPT = """You are an operational investigation reasoner for an \
industrial energy dataset.

You receive measured evidence that has already been computed by deterministic \
tools. You do not calculate anything yourself.

Rules you must follow:
- Reason only from the evidence supplied. Never invent measurements.
- Distinguish observed facts (values in the evidence) from your inference.
- Compare the competing hypotheses and say which the evidence best supports.
- Never claim any machine, equipment or process was physically changed. \
Proposed interventions are SIMULATED projections only.
- Never claim a simulated verification is a real measured verification.
- If evidence is thin or contradictory, say so and express uncertainty rather \
than forcing a confident conclusion.
- Do not recompute or restate numbers as your own calculations.

Respond with JSON only, no prose outside it, using exactly this shape:
{
  "hypothesis_assessment": [
    {"name": "...", "status": "supported|rejected|inconclusive",
     "reason": "...", "confidence": 0.0}
  ],
  "root_cause_explanation": "...",
  "recommended_intervention_explanation": "...",
  "reasoning_summary": "..."
}"""


class LLMError(RuntimeError):
    """Raised when the reasoning layer cannot produce usable output."""


class LLMAdapter:
    """Small reasoning interface used by the investigator."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport=None,
        provider: str | None = None,
    ):
        # Read provider selection from the environment, never from source.
        selected = (provider or "").strip().lower() or _configured_provider()
        preset = PROVIDERS.get(selected, {}) if selected else {}
        key_env = preset.get("key_env", "")

        self._api_key = (
            api_key
            if api_key is not None
            else (
                os.environ.get("LLM_API_KEY", "")
                or (os.environ.get(key_env, "") if key_env else "")
            )
        )
        self._model = (
            model
            or os.environ.get("LLM_MODEL", "")
            or preset.get("model", "")
            or DEFAULT_MODEL
        )
        self._base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL", "")
            or preset.get("base_url", "")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout = timeout
        # Injectable so tests can exercise the full path without a live API.
        self._transport = transport or self._http_post

    # -- interface ---------------------------------------------------------

    def is_available(self) -> bool:
        """True when a credential is configured and the adapter can be used."""
        return bool(self._api_key)

    def describe(self) -> dict:
        """Safe metadata for reporting. Never includes the API key."""
        return {
            "provider": self.provider_name(),
            "model": self._model,
            "base_url": self._base_url,
            "configured": self.is_available(),
        }

    def provider_name(self) -> str:
        """A human-readable provider label derived from the base URL."""
        match = re.search(r"https?://([^/]+)", self._base_url)
        return match.group(1) if match else "unknown"

    def reason(self, evidence: dict) -> dict:
        """Ask the model to reason over evidence.

        Returns a validated dict containing only the reasoning fields. Raises
        ``LLMError`` if the adapter is unavailable, the call fails, or the
        response cannot be parsed. Callers treat any raise as "fall back".
        """
        if not self.is_available():
            raise LLMError("No LLM credential configured.")

        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(evidence, default=str, sort_keys=True),
                },
            ],
        }

        try:
            response = self._transport(payload)
        except LLMError:
            raise
        except Exception as error:  # network, auth, malformed response
            raise LLMError("Reasoning call failed: {0}".format(error))

        return self._parse(response)

    # -- transport ---------------------------------------------------------

    def _http_post(self, payload: dict) -> dict:
        """POST to an OpenAI-compatible chat completions endpoint.

        Uses urllib from the standard library so no provider SDK is required.
        The API key is placed in a header and never logged or returned.
        """
        request = urllib.request.Request(
            url="{0}/chat/completions".format(self._base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer {0}".format(self._api_key),
                # A descriptive UA: some providers/WAFs reject the default
                # Python-urllib user agent outright with a 403.
                "User-Agent": "operational-investigation-agent/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            # Deliberately exclude the request/headers so the key cannot leak.
            raise LLMError("HTTP {0} from provider.".format(error.code))
        except Exception as error:
            raise LLMError("Request failed: {0}".format(error))
        return json.loads(body)

    # -- response handling -------------------------------------------------

    def _parse(self, response: dict) -> dict:
        """Extract and validate the model's JSON reasoning."""
        if not isinstance(response, dict):
            raise LLMError("Provider response was not a JSON object.")

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("Provider response had no message content.")

        parsed = parse_json_object(content)
        if parsed is None:
            raise LLMError("Model did not return parseable JSON.")

        validate_reasoning(parsed)
        return parsed


def parse_json_object(content: str):
    """Parse a JSON object from model output, tolerating code fences.

    Returns ``None`` when nothing parseable is present, so callers can fall
    back rather than crash on unexpected formatting.
    """
    if not isinstance(content, str):
        return None

    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None

    return parsed if isinstance(parsed, dict) else None


def validate_reasoning(parsed: dict) -> dict:
    """Ensure the model's output carries the expected structure.

    Raises ``LLMError`` when required fields are missing or malformed, which
    triggers the deterministic fallback. Model-supplied confidences are
    clamped to a sane range; statuses are coerced to known values.
    """
    for field in REQUIRED_FIELDS:
        if field not in parsed:
            raise LLMError("Model output missing field: {0}".format(field))

    assessment = parsed["hypothesis_assessment"]
    if not isinstance(assessment, list) or not assessment:
        raise LLMError("hypothesis_assessment must be a non-empty list.")

    cleaned = []
    for entry in assessment:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        status = str(entry.get("status", "inconclusive")).lower()
        if status not in VALID_STATUSES:
            status = "inconclusive"
        try:
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        cleaned.append(
            {
                "name": str(entry["name"]),
                "status": status,
                "reason": str(entry.get("reason", "")),
                "confidence": min(max(confidence, 0.0), 1.0),
            }
        )

    if not cleaned:
        raise LLMError("No usable hypothesis assessments in model output.")

    parsed["hypothesis_assessment"] = cleaned
    for field in REQUIRED_FIELDS[1:]:
        parsed[field] = str(parsed[field])
    return parsed


def build_evidence_contract(evidence: dict) -> dict:
    """Compact, already-computed evidence for the reasoning layer.

    Only values the deterministic tools actually produced are included. The
    model receives measurements, not the raw dataframes, so it cannot be
    tempted to derive its own numbers.
    """
    metrics = evidence.get("metrics", {})
    anomaly = evidence.get("anomaly", {})

    return {
        "machine": evidence.get("machine", {}),
        "anomaly": {
            "start_date": anomaly.get("start_date"),
            "end_date": anomaly.get("end_date"),
            "metric": anomaly.get("metric"),
            "anomaly_type": anomaly.get("anomaly_type"),
            "magnitude_pct": anomaly.get("magnitude_pct"),
            "severity": anomaly.get("severity"),
            "days": anomaly.get("days"),
        },
        "energy_kwh": metrics.get("energy_kwh", {}),
        "runtime_h": metrics.get("runtime_h", {}),
        "intensity": metrics.get("intensity", {}),
        "efficiency_factor": metrics.get("efficiency_factor", {}),
        "production": evidence.get("production", {}),
        "maintenance_records": evidence.get("maintenance_records", []),
        "technician_notes": evidence.get("technician_notes", []),
        "historical_incidents": evidence.get("historical_incidents", []),
    }


def reconcile_reasoning(
    deterministic_hypotheses: list,
    reasoning: dict,
    evidence: dict,
) -> dict:
    """Compare model reasoning against deterministic results.

    The deterministic verdicts are never modified. The model's per-hypothesis
    statuses are kept alongside them, and any disagreement is recorded as a
    conflict so a reviewer can see the model disagreed rather than being
    silently overruled or silently obeyed.

    Returns a dict with the model's assessment, the conflicts found, and the
    prose explanations (which are advisory only).
    """
    conflicts = []
    model_by_name = {
        entry["name"]: entry for entry in reasoning.get("hypothesis_assessment", [])
    }

    for hypothesis in deterministic_hypotheses:
        model_entry = model_by_name.get(hypothesis["name"])
        if model_entry is None:
            continue
        if model_entry["status"] != hypothesis["status"]:
            conflicts.append(
                {
                    "hypothesis": hypothesis["name"],
                    "deterministic_status": hypothesis["status"],
                    "model_status": model_entry["status"],
                    "resolution": "kept_deterministic",
                    "note": (
                        "Model interpretation conflicts with the measured "
                        "evidence; the deterministic verdict is authoritative."
                    ),
                }
            )

    for name in model_by_name:
        if name not in {h["name"] for h in deterministic_hypotheses}:
            conflicts.append(
                {
                    "hypothesis": name,
                    "deterministic_status": None,
                    "model_status": model_by_name[name]["status"],
                    "resolution": "ignored",
                    "note": "Model referenced a hypothesis that was not evaluated.",
                }
            )

    return {
        "available": True,
        "hypothesis_assessment": reasoning.get("hypothesis_assessment", []),
        "root_cause_explanation": reasoning.get("root_cause_explanation", ""),
        "recommended_intervention_explanation": reasoning.get(
            "recommended_intervention_explanation", ""
        ),
        "reasoning_summary": reasoning.get("reasoning_summary", ""),
        "conflicts": conflicts,
        "authoritative_source": "deterministic",
    }


def unavailable_reasoning(reason: str) -> dict:
    """The shape recorded when no usable reasoning was obtained."""
    return {
        "available": False,
        "hypothesis_assessment": [],
        "root_cause_explanation": "",
        "recommended_intervention_explanation": "",
        "reasoning_summary": "",
        "conflicts": [],
        "authoritative_source": "deterministic",
        "reason": reason,
    }


class FallbackAdapter:
    """Try a chain of adapters in order; the first success wins.

    Used to keep the advisory reasoning layer alive when the primary provider
    is rate-limited or unreachable (for example OpenAI with Groq as fallback).
    Every adapter shares the same ``reason`` contract, so the investigator
    needs no changes: if all of them fail, the combined error is raised and the
    deterministic results stand unchanged.
    """

    def __init__(self, adapters: list):
        self._adapters = list(adapters)

    def _active(self) -> list:
        return [adapter for adapter in self._adapters if adapter.is_available()]

    def is_available(self) -> bool:
        """True when at least one adapter in the chain is configured."""
        return bool(self._active())

    def describe(self) -> dict:
        """Safe metadata for reporting. Never includes any API key."""
        active = self._active() or self._adapters
        if not active:
            return {"provider": "unknown", "model": "unknown", "configured": False}
        description = dict(active[0].describe())
        description["fallback"] = [
            other.describe()["provider"] for other in active[1:]
        ]
        return description

    def provider_name(self) -> str:
        """Label of the first configured adapter in the chain."""
        active = self._active()
        return active[0].provider_name() if active else "unknown"

    def reason(self, evidence: dict) -> dict:
        """First successful reasoning result; raises if every adapter fails."""
        errors = []
        for adapter in self._active():
            try:
                return adapter.reason(evidence)
            except LLMError as error:
                errors.append("{0}: {1}".format(adapter.provider_name(), error))
        if errors:
            raise LLMError(
                "All reasoning providers failed ({0}).".format("; ".join(errors))
            )
        raise LLMError("No reasoning provider configured.")


def get_adapter():
    """Build the configured reasoning adapter, if any.

    A configured TypeSafe credential takes precedence, because its typed
    judgments were designed for this reasoning contract. Otherwise the
    OpenAI-compatible provider selected by LLM_PROVIDER (OpenAI or Groq) is
    used, plus the optional LLM_FALLBACK_PROVIDER which is only tried when the
    first one fails. With no credential at all a single unconfigured adapter is
    returned so callers can report availability and skip reasoning.
    """
    from app.llm.typesafe import TypeSafeAdapter

    typesafe = TypeSafeAdapter()
    if typesafe.is_available():
        return typesafe

    adapters = [LLMAdapter()]
    fallback_name = _fallback_provider()
    if fallback_name:
        adapters.append(LLMAdapter(provider=fallback_name))

    # The same provider twice (for example a fallback that resolves to the
    # primary endpoint) adds nothing, so collapse duplicates.
    unique = []
    seen = set()
    for adapter in adapters:
        label = adapter.describe()["provider"]
        if label not in seen:
            seen.add(label)
            unique.append(adapter)

    if len(unique) == 1:
        return unique[0]
    return FallbackAdapter(unique)