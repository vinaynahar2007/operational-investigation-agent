"""Deterministic intervention SIMULATION.

Nothing in this module touches equipment, actuators, or any real machine. It is
arithmetic: given a baseline consumption and an assumed reduction, it projects
what consumption would look like afterwards. The output is always marked
``simulation: True`` and must be reported as SIMULATED, because no physical
action has taken place.

Expected reductions are conservative planning assumptions, not measured
savings. They are derived from the observed size of the anomaly rather than
invented, and they are labelled as assumptions wherever they surface.
"""

from __future__ import annotations

#: Number of decimals used for every numeric result, so output is stable.
PRECISION = 2

#: Category -> proposed intervention text. The investigator keys off the root
#: cause category produced by hypothesis evaluation, never a machine id.
INTERVENTION_CATALOG = {
    "equipment_efficiency": "Inspect operating condition and restore equipment to normal efficiency.",
    "operational_scheduling": "Review operating schedule and reduce unnecessary runtime.",
    "production_demand": "Review production-energy relationship and optimize energy use per unit.",
    "maintenance_condition": "Inspect maintenance/operating condition and correct the identified issue.",
    "historical_precedent": "Apply the previously verified intervention and re-measure performance.",
}

#: Share of the observed anomaly that a category is assumed to recover.
#: Deliberately under 1.0: real interventions rarely remove the whole excess,
#: and over-claiming savings would make the projection misleading.
RECOVERY_ASSUMPTION = {
    "equipment_efficiency": 0.50,
    "operational_scheduling": 0.40,
    "production_demand": 0.20,
    "maintenance_condition": 0.50,
    "historical_precedent": 0.50,
}

DEFAULT_CATEGORY = "equipment_efficiency"


def intervention_for(category: str) -> str:
    """The proposed intervention text for a root-cause category."""
    return INTERVENTION_CATALOG.get(category, INTERVENTION_CATALOG[DEFAULT_CATEGORY])


def expected_reduction_for(category: str, observed_excess_percent: float) -> float:
    """Conservative SIMULATED reduction assumption for a category.

    ``observed_excess_percent`` is the measured anomaly size. Only part of it
    is assumed recoverable, scaled by category. The result is a planning
    assumption and is labelled as such wherever it is reported.
    """
    share = RECOVERY_ASSUMPTION.get(category, RECOVERY_ASSUMPTION[DEFAULT_CATEGORY])
    if observed_excess_percent is None:
        return 0.0
    return round(max(observed_excess_percent, 0.0) * share, PRECISION)


def simulate_intervention(
    baseline_energy: float,
    intervention_type: str,
    expected_reduction_pct: float,
) -> dict:
    """Project post-intervention energy consumption.

    projected_energy = baseline_energy * (1 - expected_reduction_pct / 100)

    Raises ``ValueError`` for inputs that cannot describe a real projection:
    negative baseline energy, a negative reduction, or a reduction of 100% or
    more (which would imply zero or negative consumption).
    """
    baseline = _validate_baseline(baseline_energy)
    reduction = _validate_reduction(expected_reduction_pct)

    projected = baseline * (1.0 - reduction / 100.0)

    return {
        "intervention": intervention_type,
        "baseline_energy": round(baseline, PRECISION),
        "projected_energy": round(projected, PRECISION),
        "projected_reduction_percent": round(reduction, PRECISION),
        "simulation": True,
    }


def _validate_baseline(value) -> float:
    try:
        baseline = float(value)
    except (TypeError, ValueError):
        raise ValueError("baseline_energy must be a number, got {0!r}".format(value))
    if baseline != baseline:  # NaN
        raise ValueError("baseline_energy must be a real number")
    if baseline < 0:
        raise ValueError(
            "baseline_energy must not be negative, got {0}".format(baseline)
        )
    return baseline


def _validate_reduction(value) -> float:
    try:
        reduction = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "expected_reduction_pct must be a number, got {0!r}".format(value)
        )
    if reduction != reduction:  # NaN
        raise ValueError("expected_reduction_pct must be a real number")
    if reduction < 0:
        raise ValueError(
            "expected_reduction_pct must not be negative, got {0}".format(reduction)
        )
    if reduction >= 100:
        raise ValueError(
            "expected_reduction_pct must be below 100, got {0}".format(reduction)
        )
    return reduction