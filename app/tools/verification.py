"""Deterministic verification of a SIMULATED intervention.

This module checks an arithmetic projection, not a physical outcome. It
compares two numbers and reports PASS or FAIL. No equipment is measured,
contacted, or observed, so every result carries ``simulation: True``.

The observed reduction is recomputed from the two energies rather than copied
from the assumed reduction. If a projection is internally inconsistent (for
example the assumed reduction does not match the energies given), this module
reports that discrepancy instead of passing the assumption through.
"""

from __future__ import annotations

PRECISION = 2

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

#: Reduction a projection must reach, in percent, to count as verified.
DEFAULT_THRESHOLD_PCT = 5.0


def verify_intervention(
    baseline_energy: float,
    projected_energy: float,
    expected_reduction_pct: float,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> dict:
    """Check a projected reduction against a threshold.

    PASS when ``observed_reduction_percent >= threshold_percent``, where the
    observed reduction is derived from the baseline and projected energies.

    ``expected_reduction_pct`` is the assumption under test. It is reported for
    comparison only; it never becomes the observed value.
    """
    baseline = _require_number(baseline_energy, "baseline_energy")
    projected = _require_number(projected_energy, "projected_energy")
    expected = _require_number(expected_reduction_pct, "expected_reduction_pct")
    threshold = _require_number(threshold_pct, "threshold_pct")

    if baseline <= 0:
        raise ValueError(
            "baseline_energy must be greater than zero to verify a reduction, "
            "got {0}".format(baseline)
        )
    if projected < 0:
        raise ValueError(
            "projected_energy must not be negative, got {0}".format(projected)
        )

    observed = (baseline - projected) / baseline * 100.0
    observed = round(observed, PRECISION)
    passed = observed >= threshold

    if passed:
        reason = (
            "SIMULATED projected reduction of {0:.2f}% meets the {1:.2f}% "
            "threshold.".format(observed, threshold)
        )
    else:
        reason = (
            "SIMULATED projected reduction of {0:.2f}% is below the {1:.2f}% "
            "threshold.".format(observed, threshold)
        )

    # Surface any mismatch between the assumption and the arithmetic itself.
    if abs(observed - expected) > PRECISION:
        reason += (
            " Note: the projected energy does not match the assumed "
            "{0:.2f}% reduction.".format(expected)
        )

    return {
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "baseline_energy": round(baseline, PRECISION),
        "projected_energy": round(projected, PRECISION),
        "observed_reduction_percent": observed,
        "expected_reduction_percent": round(expected, PRECISION),
        "threshold_percent": round(threshold, PRECISION),
        "simulation": True,
        "reason": reason,
    }


def _require_number(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("{0} must be a number, got {1!r}".format(name, value))
    if number != number:  # NaN
        raise ValueError("{0} must be a real number".format(name))
    return number