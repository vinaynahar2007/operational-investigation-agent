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

    LLM_API_KEY    credential (never logged, never echoed)
    LLM_MODEL      model identifier
    LLM_BASE_URL   API base URL, so any OpenAI-compatible endpoint works

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
    ):
        # Read from the environment, never from source.
        self._api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self._model = model or os.environ.get("LLM_MODEL", "") or DEFAULT_MODEL
        self._base_url = (
            base_url or os.environ.get("LLM_BASE_URL", "") or DEFAULT_BASE_URL
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
                "Authorization": "Bearer {0}".format(self._api_key),
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


def get_adapter() -> LLMAdapter:
    """Build the configured adapter, if any."""
    return LLMAdapter()