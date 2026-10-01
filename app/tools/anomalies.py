"""Deterministic anomaly detection over the energy dataset.

The detector answers a single question: *what changed, and which kind of change
was it?* It separates three signal types so a later investigation step can
reason about the cause instead of seeing one vague "high energy" alert:

- ``intensity``  - kWh per runtime hour is up while runtime is flat. The machine
  is drawing more power for the same amount of work: an equipment / load-side
  condition (deterioration, fouling, leaks, blocked filters).
- ``runtime``    - runtime is up while intensity is flat. The machine is running
  longer for the same power: an operational / scheduling condition (schedule
  changes, extended shifts, cycling).
- ``efficiency`` - the recorded efficiency factor drifted above its own norm.

Why the baseline is frozen inside a window
------------------------------------------
A naive trailing median quietly "heals" long anomalies: once more than half of
the 30-day window sits at the new level, the median itself moves up and the
ratio collapses back toward 1.0, truncating the reported window and hiding that
the condition is still ongoing. This module therefore anchors the baseline to
the value in force when a window *opens* and keeps comparing against that
anchor until the window closes. The reference level stays a pre-event
observation, which is what makes the window's end date trustworthy.

No machine ID, date, or expected magnitude is special-cased anywhere. Thresholds
are global constants chosen to sit in the gap between normal variation and the
real signal (see ``THRESHOLDS``).
"""

from __future__ import annotations

import math

import pandas as pd

from app.tools.analysis import (
    INTENSITY_COLUMN,
    BASELINE_WINDOW_DAYS,
    MIN_BASELINE_DAYS,
    deviation_ratio,
    machine_intensity,
    summarize_window,
    trailing_median_baseline,
)

METRIC_INTENSITY = "intensity"
METRIC_RUNTIME = "runtime"
METRIC_EFFICIENCY = "efficiency"

#: Minimum consecutive anomalous days required to open an anomaly window. A
#: single noisy day is treated as normal variation, not an operational event.
MIN_ANOMALY_DAYS = 3

#: A window closes once the metric has been back inside the band for this many
#: consecutive days, so a brief dip mid-event does not split one event in two.
RECOVERY_DAYS = 2

#: Relative deviation that counts as anomalous, per metric.
#:
#: These sit inside the measured gap between clean-machines' normal variation
#: (observed maxima of roughly +9% on runtime and +7% on intensity) and the
#: real signals (+12% or more). They are deliberately not tightened further:
#: over-sensitive thresholds turn daily noise into a stream of false alarms.
THRESHOLDS = {
    METRIC_INTENSITY: 1.12,
    METRIC_RUNTIME: 1.15,
    METRIC_EFFICIENCY: 1.03,
}

#: Magnitude bands (percent) used to grade severity.
SEVERITY_BANDS = (
    (25.0, "high"),
    (10.0, "medium"),
    (0.0, "low"),
)

#: Order of preference when several metrics fire over the same window.
#: ``intensity`` is a measured quantity (kWh actually divided by runtime hours),
#: whereas ``efficiency_factor`` is a recorded coefficient that usually just
#: restates the same physical event. When both are within ``TIE_TOLERANCE_PCT``
#: of each other the measured signal is the better label for the finding.
METRIC_PRIORITY = (METRIC_INTENSITY, METRIC_RUNTIME, METRIC_EFFICIENCY)
TIE_TOLERANCE_PCT = 2.0


def _prefers(metric: str, incumbent: str, magnitude_pct: float, incumbent_pct: float) -> bool:
    """Whether ``metric`` should replace ``incumbent`` as the primary signal."""
    if abs(magnitude_pct - incumbent_pct) > TIE_TOLERANCE_PCT:
        return magnitude_pct > incumbent_pct
    return (
        METRIC_PRIORITY.index(metric) < METRIC_PRIORITY.index(incumbent)
        if metric in METRIC_PRIORITY and incumbent in METRIC_PRIORITY
        else magnitude_pct > incumbent_pct
    )


def detect_anomalies(energy: pd.DataFrame | None = None) -> list:
    """Return one finding per anomalous event, strongest first.

    Contiguous anomalous days collapse into a single window, and overlapping
    metric signals for the same machine collapse into one finding whose
    ``metric`` names the dominant signal.
    """
    frame = _prepare(energy)
    raw: list = []

    for metric in (METRIC_INTENSITY, METRIC_RUNTIME, METRIC_EFFICIENCY):
        raw.extend(_scan_metric(frame, metric))

    findings = [
        _build_finding(window, frame) for window in _merge_findings(raw)
    ]
    findings.sort(key=lambda finding: finding["magnitude_pct"], reverse=True)
    return findings


def _prepare(energy: pd.DataFrame | None) -> pd.DataFrame:
    """Sort, add intensity, and compute every metric's trailing baseline once."""
    frame = machine_intensity(energy)
    frame = frame.sort_values(["machine_id", "date"]).reset_index(drop=True)

    frame["baseline_intensity"] = _per_machine_baseline(frame, INTENSITY_COLUMN)
    frame["baseline_runtime"] = _per_machine_baseline(frame, "runtime_h")
    frame["baseline_efficiency"] = _per_machine_baseline(frame, "efficiency_factor")

    frame["ratio_intensity"] = deviation_ratio(
        frame[INTENSITY_COLUMN], frame["baseline_intensity"]
    )
    frame["ratio_runtime"] = deviation_ratio(
        frame["runtime_h"], frame["baseline_runtime"]
    )
    frame["ratio_efficiency"] = deviation_ratio(
        frame["efficiency_factor"], frame["baseline_efficiency"]
    )
    return frame


def _per_machine_baseline(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("machine_id", sort=False)[column]
    return grouped.transform(
        lambda series: trailing_median_baseline(
            series, window=BASELINE_WINDOW_DAYS, min_periods=MIN_BASELINE_DAYS
        )
    )


def _scan_metric(frame: pd.DataFrame, metric: str) -> list:
    """Scan one metric, per machine, returning raw window dictionaries."""
    threshold = THRESHOLDS[metric]
    value_column = {
        METRIC_INTENSITY: INTENSITY_COLUMN,
        METRIC_RUNTIME: "runtime_h",
        METRIC_EFFICIENCY: "efficiency_factor",
    }[metric]
    baseline_column = "baseline_" + metric

    findings = []
    for machine_id, group in frame.groupby("machine_id", sort=True):
        windows = _scan_machine(
            machine_id,
            group.reset_index(drop=True),
            value_column,
            baseline_column,
            threshold,
        )
        for window in windows:
            if window is None:
                continue
            window["metric"] = metric
            window["supporting_metrics"] = [metric]
            window["baseline_value"] = window["anchor_baseline"]
            findings.append(window)
    return findings


def _scan_machine(
    machine_id: str,
    group: pd.DataFrame,
    value_column: str,
    baseline_column: str,
    threshold: float,
) -> list:
    """Walk one machine chronologically, freezing the baseline per window.

    State is intentionally explicit and loop-based: a single forward pass makes
    the anchor rule obvious and keeps each row's decision independent of any
    vectorised side effects.
    """
    results: list = []
    anchor = None
    current: dict | None = None
    clean_streak = 0

    for position, row in enumerate(group.itertuples(index=False)):
        value = getattr(row, value_column)
        baseline = getattr(row, baseline_column)

        # Not enough history yet, or an unusable baseline: cannot judge.
        if _is_missing(value) or _is_missing(baseline) or baseline <= 0:
            continue

        # Inside an open window the reference stays at its entry level.
        effective_baseline = anchor if anchor is not None else baseline
        ratio = value / effective_baseline

        if ratio > threshold:
            if current is None:
                # Freeze the reference level at its pre-event value so the
                # rolling median cannot absorb the anomaly and hide it.
                anchor = effective_baseline
                current = {
                    "machine_id": machine_id,
                    "start_position": position,
                    "end_position": position,
                    "anchor_baseline": effective_baseline,
                    "ratios": [ratio],
                    "values": [value],
                }
            else:
                current["end_position"] = position
                current["ratios"].append(ratio)
                current["values"].append(value)
            clean_streak = 0
            continue

        if current is not None:
            clean_streak += 1
            if clean_streak >= RECOVERY_DAYS:
                results.append(_close_window(group, current))
                current = None
                anchor = None
                clean_streak = 0
            continue

        clean_streak = 0
        anchor = None

    if current is not None:
        results.append(_close_window(group, current))
    return results


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _close_window(group: pd.DataFrame, window: dict) -> dict | None:
    """Turn an open window into a finding dict, or ``None`` if too short."""
    days = len(window["ratios"])
    if days < MIN_ANOMALY_DAYS:
        return None

    start_row = group.iloc[window["start_position"]]
    end_row = group.iloc[window["end_position"]]
    ratios = window["ratios"]

    window["mean_ratio"] = sum(ratios) / len(ratios)
    window["peak_ratio"] = max(ratios)
    window["mean_value"] = sum(window["values"]) / len(window["values"])
    window["start_date"] = start_row["date"].date().isoformat()
    window["end_date"] = end_row["date"].date().isoformat()
    window["magnitude_pct"] = round((window["mean_ratio"] - 1) * 100, 2)
    window["peak_magnitude_pct"] = round((window["peak_ratio"] - 1) * 100, 2)
    window["days"] = days
    return window


def _merge_findings(raw_windows: list) -> list:
    """Collapse windows describing the same event into one finding.

    A machine whose intensity and efficiency factor both rise over the same
    dates is one operational event, not two. The strongest metric names the
    event; the others are kept as supporting signals.
    """
    kept = [window for window in raw_windows if window is not None]
    kept.sort(
        key=lambda w: (w["machine_id"], w["start_date"], w["end_date"], w["metric"])
    )

    merged: list = []
    for window in kept:
        window = dict(window)
        window.setdefault("supporting_metrics", [window["metric"]])

        if merged:
            previous = merged[-1]
            same_event = (
                previous["machine_id"] == window["machine_id"]
                and previous["end_date"] >= window["start_date"]
            )
            if same_event:
                _absorb(previous, window)
                continue

        merged.append(window)

    return merged


def _absorb(target: dict, addition: dict) -> None:
    """Fold ``addition`` into ``target``, keeping the stronger metric primary."""
    target["end_date"] = max(target["end_date"], addition["end_date"])
    target["supporting_metrics"] = sorted(
        set(target.get("supporting_metrics", []))
        | set(addition.get("supporting_metrics", []))
        | {addition["metric"]}
    )
    target["peak_magnitude_pct"] = max(
        target["peak_magnitude_pct"], addition["peak_magnitude_pct"]
    )
    if _prefers(
        addition["metric"],
        target["metric"],
        addition["magnitude_pct"],
        target["magnitude_pct"],
    ):
        for key in (
            "metric",
            "magnitude_pct",
            "mean_ratio",
            "peak_ratio",
            "baseline_value",
        ):
            target[key] = addition[key]
def _build_finding(window: dict, frame: pd.DataFrame) -> dict:
    """Assemble the public finding structure for one anomaly window."""
    machine_id = window["machine_id"]
    start_date = window["start_date"]
    end_date = window["end_date"]

    summary = summarize_window(machine_id, start_date, end_date, energy=frame)
    machine_slice = frame[
        (frame["machine_id"] == machine_id)
        & (frame["date"] >= pd.Timestamp(start_date))
        & (frame["date"] <= pd.Timestamp(end_date))
    ]

    return {
        "machine_id": machine_id,
        "start_date": start_date,
        "end_date": end_date,
        "metric": window["metric"],
        "ratio": round(window["mean_ratio"], 4),
        "magnitude_pct": window["magnitude_pct"],
        "peak_magnitude_pct": window["peak_magnitude_pct"],
        "severity": _severity(window["magnitude_pct"]),
        "confidence": _confidence(window["magnitude_pct"], window["days"]),
        "days": window["days"],
        "anomaly_type": _anomaly_type(window["metric"]),
        "supporting_metrics": sorted(set(window.get("supporting_metrics", []))),
        "baseline_value": round(window["baseline_value"], 4),
        "mean_intensity_ratio": _mean_ratio(machine_slice, "ratio_intensity"),
        "mean_runtime_ratio": _mean_ratio(machine_slice, "ratio_runtime"),
        "average_intensity": summary["average_intensity"],
        "average_runtime_h": summary["average_runtime_h"],
        "average_energy_kwh": summary["average_energy_kwh"],
        "average_efficiency_factor": summary["average_efficiency_factor"],
        "production_change_pct": summary["production_change_pct"],
    }


def _mean_ratio(machine_slice: pd.DataFrame, column: str):
    if column not in machine_slice.columns or machine_slice.empty:
        return None
    values = pd.to_numeric(machine_slice[column], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _anomaly_type(metric: str) -> str:
    """Translate the dominant metric into a cause-oriented label.

    This distinction is the point of the module: "runtime rose" and "intensity
    rose" are different investigations, not two flavours of high energy.
    """
    return {
        METRIC_INTENSITY: "equipment_efficiency",
        METRIC_RUNTIME: "operational_scheduling",
        METRIC_EFFICIENCY: "equipment_efficiency",
    }[metric]


def _severity(magnitude_pct: float) -> str:
    for threshold, label in SEVERITY_BANDS:
        if magnitude_pct >= threshold:
            return label
    return "low"


def _confidence(magnitude_pct: float, days: int) -> float:
    """Deterministic 0-1 confidence from signal size and window length.

    Bigger and longer events score higher; the curve saturates so no window can
    claim certainty. Nothing here is tuned per machine or per date.
    """
    magnitude_score = math.tanh(max(magnitude_pct, 0.0) / 20.0)
    duration_score = min(days, 14) / 14.0
    score = 0.45 + 0.35 * magnitude_score + 0.15 * duration_score
    return round(min(score, 0.95), 3)


def findings_by_machine(findings: list) -> dict:
    """Group findings by machine_id for convenient reporting."""
    grouped: dict = {}
    for finding in findings:
        grouped.setdefault(finding["machine_id"], []).append(finding)
    return grouped


def detect_anomalies_for_machine(machine_id: str, energy: pd.DataFrame | None = None) -> list:
    """Findings for one machine, using the same global thresholds."""
    frame = machine_intensity(energy)
    subset = frame[frame["machine_id"] == machine_id]
    if subset.empty:
        return []
    return detect_anomalies(subset)