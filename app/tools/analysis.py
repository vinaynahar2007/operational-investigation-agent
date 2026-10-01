"""Deterministic metric calculations for the operational dataset.

Every function here is pure arithmetic over the loaded dataframes: no random
seeds, no learned parameters, no network calls. The same input always produces
the same output, which is what allows the anomaly detector to be trusted and
re-tested.

Baseline philosophy
-------------------
Baselines are *trailing*: an observation on day D is compared only against
earlier observations. ``shift(1)`` before the rolling median guarantees that a
value never contributes to its own reference level, so a real step change is
not silently absorbed into the baseline that is meant to expose it.
"""

from __future__ import annotations

import pandas as pd

from app.tools.dataset import load_energy, load_machines, load_production

BASELINE_WINDOW_DAYS = 30
MIN_BASELINE_DAYS = 10

INTENSITY_COLUMN = "kwh_per_runtime_h"


def machine_intensity(energy: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach ``kwh_per_runtime_h`` = energy_kwh / runtime_h.

    Zero, negative or missing runtime yields ``NaN`` instead of ``inf`` so the
    column stays usable by downstream aggregates.
    """
    frame = load_energy() if energy is None else energy.copy()
    runtime = pd.to_numeric(frame["runtime_h"], errors="coerce")
    kwh = pd.to_numeric(frame["energy_kwh"], errors="coerce")
    frame[INTENSITY_COLUMN] = kwh / runtime.where(runtime > 0)
    return frame


def trailing_median_baseline(
    series: pd.Series,
    window: int = BASELINE_WINDOW_DAYS,
    min_periods: int = MIN_BASELINE_DAYS,
) -> pd.Series:
    """Trailing median of the ``window`` observations *before* each point.

    ``min_periods`` guards the warm-up period: until a machine has enough
    history the baseline is ``NaN`` and no comparison should be attempted.
    """
    return series.shift(1).rolling(window=window, min_periods=min_periods).median()


def deviation_ratio(current: pd.Series, baseline: pd.Series) -> pd.Series:
    """``current / baseline``, with non-positive baselines as ``NaN``."""
    valid = pd.to_numeric(baseline, errors="coerce")
    return pd.to_numeric(current, errors="coerce") / valid.where(valid > 0)


def runtime_deviation(
    energy: pd.DataFrame | None = None, window: int = BASELINE_WINDOW_DAYS
) -> pd.DataFrame:
    """Per-machine runtime ratio against its own trailing baseline.

    Returns a copy of the frame with ``runtime_baseline`` and
    ``runtime_ratio`` columns, machine by machine in date order.
    """
    frame = load_energy() if energy is None else energy.copy()
    frame = frame.sort_values(["machine_id", "date"]).reset_index(drop=True)
    grouped = frame.groupby("machine_id", sort=False)["runtime_h"]
    frame["runtime_baseline"] = grouped.transform(
        lambda series: trailing_median_baseline(series, window=window)
    )
    frame["runtime_ratio"] = deviation_ratio(frame["runtime_h"], frame["runtime_baseline"])
    return frame


def plant_energy_per_unit(
    energy: pd.DataFrame | None = None, production: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Daily plant-level energy intensity: total kWh / production units.

    A plant-wide reference used as context: it separates "the whole factory
    became less efficient" from "one machine became less efficient".
    """
    energy_frame = load_energy() if energy is None else energy
    production_frame = load_production() if production is None else production

    total = (
        energy_frame.groupby("date", as_index=False)["energy_kwh"]
        .sum()
        .rename(columns={"energy_kwh": "total_energy_kwh"})
    )
    merged = total.merge(
        production_frame[["date", "production_units"]], on="date", how="left"
    )
    units = pd.to_numeric(merged["production_units"], errors="coerce")
    merged["plant_kwh_per_unit"] = merged["total_energy_kwh"] / units.where(units > 0)
    return merged.sort_values("date").reset_index(drop=True)


def summarize_window(
    machine_id: str,
    start_date,
    end_date,
    energy: pd.DataFrame | None = None,
    production: pd.DataFrame | None = None,
) -> dict:
    """Deterministic averages for one machine over an inclusive date window.

    Includes the production change across the same window when production data
    is supplied, because a production shift is the usual innocent explanation
    for higher energy and belongs in the evidence next to the anomaly.
    """
    frame = machine_intensity(energy)
    production_frame = load_production() if production is None else production

    window = frame[
        (frame["machine_id"] == machine_id)
        & (frame["date"] >= pd.Timestamp(start_date))
        & (frame["date"] <= pd.Timestamp(end_date))
    ]

    return {
        "machine_id": machine_id,
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "days": int(len(window)),
        "average_energy_kwh": _mean(window, "energy_kwh"),
        "average_runtime_h": _mean(window, "runtime_h"),
        "average_intensity": _mean(window, INTENSITY_COLUMN),
        "average_efficiency_factor": _mean(window, "efficiency_factor"),
        "production_change_pct": _production_change(
            production_frame, start_date, end_date
        ),
    }


def _mean(frame: pd.DataFrame, column: str):
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _production_change(production: pd.DataFrame, start_date, end_date):
    """Percentage change in production units across the window."""
    if "production_units" not in production.columns:
        return None
    units = pd.to_numeric(production["production_units"], errors="coerce")
    ordered = production.assign(_units=units).sort_values("date")
    window = ordered[
        (ordered["date"] >= pd.Timestamp(start_date))
        & (ordered["date"] <= pd.Timestamp(end_date))
    ]
    series = window["_units"].dropna()
    if len(series) < 2:
        return None
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    if first == 0:
        return None
    return round((last - first) / first * 100, 2)


def machine_reference() -> pd.DataFrame:
    """Machine metadata indexed by machine_id, for joining onto metrics."""
    return load_machines().set_index("machine_id")