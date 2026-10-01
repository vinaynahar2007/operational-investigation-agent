"""Deterministic evidence retrieval for an investigation.

This module answers one question: *what does the data actually show?* It
assembles measured facts only. It never reasons about cause, never invents a
value, and never infers anything that is not computed from a loader in
``app.tools.dataset``.

Every number in the returned structure is either a direct dataset value or an
explicitly derived ratio between an anomaly window and its preceding baseline
window. Derived values carry the inputs that produced them so a later step (or a
human) can check the arithmetic.

Observed vs inferred
--------------------
Everything produced here is OBSERVED or COMPUTED. Cause statements belong to
``app.agents.investigator``, which marks them INFERRED.
"""

from __future__ import annotations

import pandas as pd

from app.tools.analysis import (
    INTENSITY_COLUMN,
    machine_intensity,
    summarize_window,
)
from app.tools.dataset import (
    load_incidents,
    load_maintenance,
    load_machines,
    load_production,
    load_technician_notes,
)

#: Length of the reference period immediately preceding an anomaly window.
BASELINE_WINDOW_DAYS = 30


def gather_evidence(machine_id: str, anomaly: dict) -> dict:
    """Collect measured evidence for one machine over one anomaly window.

    Returns a dictionary with machine metadata, per-metric window/baseline
    comparisons, and the qualitative records (maintenance, notes, incidents)
    that fall inside or near the window.
    """
    start_date = anomaly["start_date"]
    end_date = anomaly["end_date"]

    baseline_start, baseline_end = _baseline_window(start_date)

    window_summary = summarize_window(machine_id, start_date, end_date)
    baseline_summary = summarize_window(machine_id, baseline_start, baseline_end)

    window_dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date))

    return {
        "machine": machine_info(machine_id),
        "anomaly": dict(anomaly),
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "days": int(len(window_dates)),
        },
        "baseline_window": {
            "start_date": baseline_start,
            "end_date": baseline_end,
            "days": BASELINE_WINDOW_DAYS,
        },
        "metrics": _metric_comparisons(window_summary, baseline_summary),
        "production": production_behavior(start_date, end_date, baseline_start, baseline_end),
        "maintenance_records": maintenance_in_window(machine_id, start_date, end_date),
        "technician_notes": technician_notes_in_window(machine_id, start_date, end_date),
        "historical_incidents": incidents_for_machine(machine_id, start_date, end_date),
        "data_quality": _data_quality(
            window_summary, baseline_summary, machine_id
        ),
    }


def _baseline_window(start_date) -> tuple:
    """The reference period immediately before the anomaly window."""
    start = pd.Timestamp(start_date)
    baseline_end = (start - pd.Timedelta(days=1)).date().isoformat()
    baseline_start = (start - pd.Timedelta(days=BASELINE_WINDOW_DAYS)).date().isoformat()
    return baseline_start, baseline_end


def machine_info(machine_id: str) -> dict:
    """Metadata for one machine, or a minimal placeholder if it is unknown."""
    machines = load_machines()
    row = machines[machines["machine_id"] == machine_id]
    if row.empty:
        return {"machine_id": machine_id, "found": False}
    record = row.iloc[0]
    return {
        "machine_id": machine_id,
        "found": True,
        "machine_name": str(record["machine_name"]),
        "category": str(record["category"]),
        "rated_kw": float(record["rated_kw"]),
        "baseline_runtime_h": float(record["baseline_runtime_h"]),
    }


def _metric_comparisons(window_summary: dict, baseline_summary: dict) -> dict:
    """Window value vs baseline value and the ratio between them."""
    specs = {
        "energy_kwh": "average_energy_kwh",
        "runtime_h": "average_runtime_h",
        "intensity": "average_intensity",
        "efficiency_factor": "average_efficiency_factor",
    }
    comparisons = {}
    for name, key in specs.items():
        window_value = window_summary.get(key)
        baseline_value = baseline_summary.get(key)
        comparisons[name] = {
            "window_value": window_value,
            "baseline_value": baseline_value,
            "ratio": _ratio(window_value, baseline_value),
            "change_pct": _change_pct(window_value, baseline_value),
        }
    return comparisons


def _ratio(window_value, baseline_value):
    if window_value is None or baseline_value in (None, 0):
        return None
    return round(window_value / baseline_value, 4)


def _change_pct(window_value, baseline_value):
    if window_value is None or baseline_value in (None, 0):
        return None
    return round((window_value - baseline_value) / baseline_value * 100, 2)


def production_behavior(start_date, end_date, baseline_start, baseline_end) -> dict:
    """Plant production in the anomaly window vs the baseline window."""
    production = load_production()
    window_mean = _production_mean(production, start_date, end_date)
    baseline_mean = _production_mean(production, baseline_start, baseline_end)
    return {
        "window_mean_units": window_mean,
        "baseline_mean_units": baseline_mean,
        "ratio": _ratio(window_mean, baseline_mean),
        "change_pct": _change_pct(window_mean, baseline_mean),
    }


def _production_mean(production: pd.DataFrame, start_date, end_date):
    window = production[
        (production["date"] >= pd.Timestamp(start_date))
        & (production["date"] <= pd.Timestamp(end_date))
    ]
    values = pd.to_numeric(window["production_units"], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 2)


def maintenance_in_window(machine_id: str, start_date, end_date) -> list:
    """Maintenance records for this machine inside the anomaly window."""
    maintenance = load_maintenance()
    rows = maintenance[
        (maintenance["machine_id"] == machine_id)
        & (maintenance["date"] >= pd.Timestamp(start_date))
        & (maintenance["date"] <= pd.Timestamp(end_date))
    ]
    return [
        {
            "date": row["date"].date().isoformat(),
            "maintenance_type": str(row["maintenance_type"]),
            "status": str(row["status"]),
            "notes": str(row["notes"]),
        }
        for _, row in rows.iterrows()
    ]


def technician_notes_in_window(machine_id: str, start_date, end_date) -> list:
    """Technician observations for this machine inside the anomaly window."""
    notes = load_technician_notes()
    rows = notes[
        (notes["machine_id"] == machine_id)
        & (notes["date"] >= pd.Timestamp(start_date))
        & (notes["date"] <= pd.Timestamp(end_date))
    ]
    return [
        {
            "date": row["date"].date().isoformat(),
            "note": str(row["note"]),
            "recommended_followup": str(row["recommended_followup"]),
        }
        for _, row in rows.iterrows()
    ]


def incidents_for_machine(machine_id: str, start_date, end_date) -> list:
    """Historical incidents for this machine, flagged for whether they overlap.

    ``within_window`` separates an incident that coincided with this anomaly
    from earlier history for the same machine. Both are useful, but only the
    first is contemporaneous support.
    """
    incidents = load_incidents()
    rows = incidents[incidents["machine_id"] == machine_id].sort_values("date")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                "incident_id": str(row["incident_id"]),
                "date": row["date"].date().isoformat(),
                "symptom": str(row["symptom"]),
                "description": str(row["description"]),
                "intervention": str(row["intervention"]),
                "verified_energy_reduction": _as_float(
                    row["verified_energy_reduction"]
                ),
                "verified": _as_bool(row["verified"]),
                "within_window": bool(start <= row["date"] <= end),
            }
        )
    return records


def _as_float(value):
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _data_quality(window_summary: dict, baseline_summary: dict, machine_id: str) -> dict:
    """Flags describing how much trustworthy evidence is available.

    The investigator uses these to decide whether a conclusion is justified,
    so a thin dataset yields an inconclusive result rather than a confident
    guess.
    """
    available = [
        window_summary.get("average_energy_kwh") is not None,
        window_summary.get("average_runtime_h") is not None,
        window_summary.get("average_intensity") is not None,
        window_summary.get("average_efficiency_factor") is not None,
        baseline_summary.get("average_energy_kwh") is not None,
        baseline_summary.get("average_runtime_h") is not None,
        baseline_summary.get("average_intensity") is not None,
    ]
    return {
        "metrics_available": int(sum(available)),
        "metrics_expected": len(available),
        "sufficient": all(available),
        "machine_known": bool(machine_info(machine_id)["found"]),
    }