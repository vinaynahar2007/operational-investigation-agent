"""Optional TypeSafe (System One) reasoning adapter.

This adapter is an alternative reasoning layer to the OpenAI-compatible
``LLMAdapter``. TypeSafe's System One model (Jev) does not generate prose: it
returns typed judgments and calibrated probabilities that code consumes
directly. The adapter therefore asks one Noul ("does the condition hold?")
question per candidate explanation, all in a single request over the same
state, and thresholds the returned probabilities in code:

    noul >= support threshold  -> supported
    noul <= reject threshold   -> rejected
    otherwise                  -> inconclusive

Middle values stay inconclusive rather than being forced to a verdict,
mirroring the deterministic evaluator's own inconclusive band.

Like every reasoning layer in this project, this adapter is advisory. It never
computes measurements, never changes a hypothesis status and never selects the
root cause; ``reconcile_reasoning`` keeps the deterministic verdicts
authoritative and records any disagreement as a conflict.

Provider configuration comes entirely from the environment:

    TYPESAFE_API_KEY            credential (never logged, never echoed)
    TYPESAFE_MODEL              model identifier (default: jev-latest)
    TYPESAFE_BASE_URL           API base URL (default: https://api.typesafe.ai)
    TYPESAFE_SUPPORT_THRESHOLD  noul >= this -> supported (default: 0.7)
    TYPESAFE_REJECT_THRESHOLD   noul <= this -> rejected (default: 0.3)

The HTTP contract follows the published System One API reference
(``POST /v1/systemone``); the official ``typesafe-sdk`` package exposes the
same endpoint with automatic retries if a project prefers a dependency.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from app.llm.adapter import LLMError

DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_TIMEOUT = 30.0
SYSTEM_ONE_PATH = "/v1/systemone"

STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_INCONCLUSIVE = "inconclusive"

DEFAULT_SUPPORT_THRESHOLD = 0.7
DEFAULT_REJECT_THRESHOLD = 0.3

#: The five candidate explanations, mirroring the investigator's hypothesis
#: names. One Noul question is asked per candidate over the same state. The
#: question ids match the deterministic hypothesis names so that
#: ``reconcile_reasoning`` can compare each judgment against its verdict;
#: ``tests/test_typesafe.py`` asserts the two sets stay identical. Question ids
#: are chosen in code and are not sent to the model.
CANDIDATES = (
    {
        "id": "production_increase",
        "instructions": (
            "Does the measured evidence show that the rise in energy "
            "consumption (`energy_kwh`) is explained by a rise in production "
            "demand (`production`), where output units increased materially "
            "and account for most or all of the extra energy?"
        ),
        "criteria": {
            "true": (
                "Production rose materially and explains most or all of the "
                "energy increase."
            ),
            "false": (
                "Production did not rise materially, or rose too little to "
                "explain the energy increase."
            ),
        },
    },
    {
        "id": "runtime_scheduling_increase",
        "instructions": (
            "Does the measured evidence show that the energy rise is "
            "explained by the machine running longer (`runtime_h`), that is a "
            "scheduling change, rather than by energy per runtime hour "
            "increasing?"
        ),
        "criteria": {
            "true": "Runtime hours rose materially and track the energy increase.",
            "false": "Runtime hours stayed broadly flat relative to the baseline.",
        },
    },
    {
        "id": "equipment_efficiency_degradation",
        "instructions": (
            "Does the measured evidence show the machine drawing more energy "
            "for the same runtime hour (`intensity`, `efficiency_factor`) "
            "while runtime and production did not materially change, that is "
            "degraded equipment efficiency?"
        ),
        "criteria": {
            "true": (
                "Energy per runtime hour rose materially while runtime and "
                "production stayed broadly flat."
            ),
            "false": "Energy per runtime hour did not rise materially.",
        },
    },
    {
        "id": "maintenance_operational_condition",
        "instructions": (
            "Does the measured evidence connect the anomaly window to "
            "recorded maintenance state or operational conditions "
            "(`maintenance_records`, `technician_notes`), such as overdue "
            "service or technician-observed abnormalities?"
        ),
        "criteria": {
            "true": (
                "Maintenance or technician records in the window support a "
                "condition-based explanation."
            ),
            "false": (
                "No relevant maintenance or technician evidence was recorded "
                "in the window."
            ),
        },
    },
    {
        "id": "historical_similar_incident",
        "instructions": (
            "Do the recorded prior incidents (`historical_incidents`) "
            "describe a similar symptom on this machine, such that the "
            "anomaly repeats a known failure pattern with a verified "
            "intervention?"
        ),
        "criteria": {
            "true": "A prior verified incident matches the current symptom.",
            "false": "No comparable prior incident is recorded.",
        },
    },
)


class TypeSafeAdapter:
    """Typed-judgment reasoning adapter for TypeSafe's System One API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport=None,
        support_threshold: float | None = None,
        reject_threshold: float | None = None,
    ):
        # Read from the environment, never from source.
        self._api_key = (
            api_key
            if api_key is not None
            else os.environ.get("TYPESAFE_API_KEY", "")
        )
        self._model = model or os.environ.get("TYPESAFE_MODEL", "") or DEFAULT_MODEL
        self._base_url = (
            base_url or os.environ.get("TYPESAFE_BASE_URL", "") or DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout = timeout
        self._support_threshold = (
            support_threshold
            if support_threshold is not None
            else _env_float("TYPESAFE_SUPPORT_THRESHOLD", DEFAULT_SUPPORT_THRESHOLD)
        )
        self._reject_threshold = (
            reject_threshold
            if reject_threshold is not None
            else _env_float("TYPESAFE_REJECT_THRESHOLD", DEFAULT_REJECT_THRESHOLD)
        )
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
        """Ask TypeSafe for one typed judgment per candidate explanation.

        All questions go in a single request over the same state. Returns a
        reasoning dict matching the adapter contract. Raises ``LLMError`` when
        unavailable or when no usable judgment can be obtained; callers treat
        any raise as "fall back to deterministic reasoning".
        """
        if not self.is_available():
            raise LLMError("No TypeSafe credential configured.")

        payload = {
            "state": evidence,
            "model": self._model,
            "questions": {
                candidate["id"]: {
                    "type": "noul",
                    "instructions": candidate["instructions"],
                    "criteria": candidate["criteria"],
                }
                for candidate in CANDIDATES
            },
        }

        try:
            response = self._transport(payload)
        except LLMError:
            raise
        except Exception as error:  # network, auth, malformed response
            raise LLMError("TypeSafe call failed: {0}".format(error))

        return self._to_reasoning(response)

    # -- transport ---------------------------------------------------------

    def _http_post(self, payload: dict) -> dict:
        """POST to the System One evaluation endpoint.

        Uses urllib from the standard library so no provider SDK is required.
        The API key is placed in a header and never logged or returned.
        """
        request = urllib.request.Request(
            url="{0}{1}".format(self._base_url, SYSTEM_ONE_PATH),
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
            raise LLMError("HTTP {0} from TypeSafe.".format(error.code))
        except Exception as error:
            raise LLMError("Request failed: {0}".format(error))
        try:
            return json.loads(body)
        except ValueError:
            raise LLMError("TypeSafe returned a non-JSON response.")

    # -- response handling -------------------------------------------------

    def _to_reasoning(self, response: dict) -> dict:
        """Threshold the Noul answers in code and compose the reasoning dict."""
        if not isinstance(response, dict):
            raise LLMError("TypeSafe returned a malformed response.")
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise LLMError("TypeSafe response missing answers.")

        judgments = []
        for candidate in CANDIDATES:
            noul = _noul_value(answers.get(candidate["id"]))
            if noul is None:
                # A missing or malformed judgment is skipped, never invented.
                continue
            judgments.append(
                {
                    "name": candidate["id"],
                    "noul": noul,
                    "status": self._status(noul),
                }
            )

        if not judgments:
            raise LLMError("No usable TypeSafe judgments returned.")

        return {
            "hypothesis_assessment": [
                self._assessment(judgment) for judgment in judgments
            ],
            "root_cause_explanation": _root_cause_explanation(
                judgments, self._support_threshold
            ),
            "recommended_intervention_explanation": (
                "Typed judgments do not propose interventions; the "
                "deterministic intervention and its simulation remain "
                "authoritative."
            ),
            "reasoning_summary": _summary(judgments),
        }

    def _status(self, noul: float) -> str:
        """Threshold a noul probability. Code owns the decision, not the model."""
        if noul >= self._support_threshold:
            return STATUS_SUPPORTED
        if noul <= self._reject_threshold:
            return STATUS_REJECTED
        return STATUS_INCONCLUSIVE

    def _assessment(self, judgment: dict) -> dict:
        """One hypothesis_assessment entry for a Noul judgment."""
        status = judgment["status"]
        noul = judgment["noul"]
        if status == STATUS_SUPPORTED:
            reason = (
                "TypeSafe noul {0:.2f} is at or above the support threshold "
                "{1:.2f}: the typed judgment supports this explanation."
            ).format(noul, self._support_threshold)
        elif status == STATUS_REJECTED:
            reason = (
                "TypeSafe noul {0:.2f} is at or below the reject threshold "
                "{1:.2f}: the typed judgment contradicts this explanation."
            ).format(noul, self._reject_threshold)
        else:
            reason = (
                "TypeSafe noul {0:.2f} sits between the thresholds: the "
                "judgment is not decisive."
            ).format(noul)
        return {
            "name": judgment["name"],
            "status": status,
            # Decisiveness derived in code from the noul value; Noul answers
            # carry no separate model-reported confidence.
            "confidence": abs(noul - 0.5) * 2,
            "reason": reason,
        }


def _noul_value(answer):
    """Extract a trustworthy noul probability, or ``None``."""
    if not isinstance(answer, dict):
        return None
    answer_type = answer.get("type")
    if answer_type is not None and answer_type != "noul":
        return None
    try:
        value = float(answer.get("noul"))
    except (TypeError, ValueError):
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


def _root_cause_explanation(judgments: list, support_threshold: float) -> str:
    """Code-composed explanation naming the strongest typed judgment."""
    supported = [j for j in judgments if j["status"] == STATUS_SUPPORTED]
    if supported:
        best = max(supported, key=lambda j: j["noul"])
        return (
            "Strongest typed judgment: '{0}' is supported with noul "
            "probability {1:.2f} (support threshold {2:.2f})."
        ).format(best["name"], best["noul"], support_threshold)

    best = max(judgments, key=lambda j: j["noul"])
    return (
        "No candidate explanation reached the support threshold; the highest "
        "typed judgment was '{0}' at noul {1:.2f}."
    ).format(best["name"], best["noul"])


def _summary(judgments: list) -> str:
    """Factual, code-composed recap of the judgments. No invented numbers."""
    parts = [
        "{0}: {1} (noul {2:.2f})".format(j["name"], j["status"], j["noul"])
        for j in judgments
    ]
    return (
        "TypeSafe typed judgments over the measured evidence - "
        + "; ".join(parts)
        + ". The deterministic verdicts remain authoritative."
    )


def _env_float(name: str, default: float) -> float:
    """Read a float threshold from the environment, tolerating bad values."""
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default
