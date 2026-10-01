"""Deterministic dataset loading and structural validation.

Loaders read the files under ``data/`` exactly as they are stored. Nothing in
this module repairs, imputes, normalises, or rewrites any CSV or JSON file.
Validation reports structural problems only; it never changes the data.

Note on ``rated_kw``: it is descriptive metadata about a machine, not a hard
ceiling on energy consumption. Measured ``energy_kwh`` can legitimately exceed
``rated_kw * runtime_h`` (for example when a machine is running inefficiently
or is degraded). No validation here enforces that ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATE_FORMAT = "%Y-%m-%d"

# app/tools/dataset.py -> app/tools -> app -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MAX_RUNTIME_H = 24.0
MIN_EFFICIENCY_FACTOR = 1.0

REQUIRED_FILES = (
    "machines.csv",
    "energy.csv",
    "production.csv",
    "maintenance.csv",
    "incidents.csv",
    "technician_notes.csv",
    "scenarios.json",
)

REQUIRED_COLUMNS = {
    "machines.csv": (
        "machine_id",
        "machine_name",
        "category",
        "rated_kw",
        "baseline_runtime_h",
    ),
    "energy.csv": (
        "date",
        "machine_id",
        "runtime_h",
        "energy_kwh",
        "efficiency_factor",
    ),
    "production.csv": ("date", "production_units"),
    "maintenance.csv": ("date", "machine_id", "maintenance_type", "status", "notes"),
    "incidents.csv": (
        "incident_id",
        "date",
        "machine_id",
        "symptom",
        "description",
        "intervention",
        "verified_energy_reduction",
        "verified",
    ),
    "technician_notes.csv": ("date", "machine_id", "note", "recommended_followup"),
}

MACHINE_REFERENCE_FILES = (
    "energy.csv",
    "maintenance.csv",
    "incidents.csv",
    "technician_notes.csv",
)


def data_dir() -> Path:
    """Absolute path of the dataset directory."""
    return DATA_DIR


def normalize_machine_id(machine_id) -> str:
    """Canonical machine identifier used throughout the investigation code."""
    if machine_id is None:
        return ""
    return str(machine_id).strip().upper()


def _normalize_machine_id_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "machine_id" not in frame.columns:
        return frame
    normalized = frame.copy()
    normalized["machine_id"] = normalized["machine_id"].map(
        lambda value: normalize_machine_id(value) if pd.notna(value) else value
    )
    return normalized


def _parse_dates(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Parse date columns explicitly as ``%Y-%m-%d``.

    ``errors="coerce"`` turns malformed values into ``NaT`` so that loading
    never raises and ``validate_dataset`` can report the offending rows.
    """
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column], format=DATE_FORMAT, errors="coerce"
            )
    return frame


def _read_csv(filename: str, date_columns=()) -> pd.DataFrame:
    return _parse_dates(pd.read_csv(DATA_DIR / filename), date_columns)


def load_machines() -> pd.DataFrame:
    """Machine metadata: identity, category, rated_kw, baseline runtime."""
    return _normalize_machine_id_column(_read_csv("machines.csv"))


def load_energy() -> pd.DataFrame:
    """Daily per-machine runtime, energy and efficiency factor."""
    return _normalize_machine_id_column(_read_csv("energy.csv", date_columns=("date",)))


def load_production() -> pd.DataFrame:
    """Daily factory production units."""
    return _read_csv("production.csv", date_columns=("date",))


def load_maintenance() -> pd.DataFrame:
    """Maintenance history and status."""
    return _normalize_machine_id_column(_read_csv("maintenance.csv", date_columns=("date",)))


def load_incidents() -> pd.DataFrame:
    """Verified historical incidents with their interventions."""
    return _normalize_machine_id_column(_read_csv("incidents.csv", date_columns=("date",)))


def load_technician_notes() -> pd.DataFrame:
    """Human operational observations and recommended follow-ups."""
    return _normalize_machine_id_column(_read_csv("technician_notes.csv", date_columns=("date",)))


def load_scenarios() -> list:
    """Demo scenarios from ``scenarios.json``."""
    with open(DATA_DIR / "scenarios.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["scenarios"]


def _check_files(errors: list) -> bool:
    missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).is_file()]
    for name in missing:
        errors.append("missing_file: {0}".format(name))
    return not missing


def _check_columns(frames: dict, errors: list) -> None:
    for filename, required in REQUIRED_COLUMNS.items():
        frame = frames[filename]
        for column in required:
            if column not in frame.columns:
                errors.append("missing_column: {0} -> {1}".format(filename, column))


def _check_nulls(frames: dict, errors: list) -> None:
    for filename, required in REQUIRED_COLUMNS.items():
        present = [c for c in required if c in frames[filename].columns]
        null_counts = frames[filename][present].isna().sum()
        for column, count in null_counts.items():
            if count:
                errors.append(
                    "null_value: {0} -> {1} ({2} rows)".format(
                        filename, column, int(count)
                    )
                )


def _check_machine_ids(frames: dict, errors: list) -> None:
    known = set(frames["machines.csv"]["machine_id"].dropna())
    for filename in MACHINE_REFERENCE_FILES:
        frame = frames[filename]
        if "machine_id" not in frame.columns:
            continue
        for machine_id in sorted(set(frame["machine_id"].dropna()) - known):
            errors.append("unknown_machine_id: {0} -> {1}".format(filename, machine_id))


def _check_energy(energy: pd.DataFrame, errors: list) -> None:
    if not {"date", "machine_id"}.issubset(energy.columns):
        return

    duplicated = energy.duplicated(subset=["date", "machine_id"], keep=False)
    keys = energy.loc[duplicated, ["date", "machine_id"]].drop_duplicates()
    for row in keys.itertuples(index=False):
        errors.append(
            "duplicate_energy_record: {0} -> {1}".format(row.date, row.machine_id)
        )

    invalid_dates = energy["date"].isna()
    if invalid_dates.any():
        errors.append(
            "invalid_date: energy.csv -> date ({0} rows)".format(
                int(invalid_dates.sum())
            )
        )

    if "runtime_h" in energy.columns:
        runtime = pd.to_numeric(energy["runtime_h"], errors="coerce")
        bad = runtime.isna() | (runtime <= 0) | (runtime > MAX_RUNTIME_H)
        if bad.any():
            errors.append(
                "invalid_runtime_h: energy.csv -> runtime_h ({0} rows, "
                "must be > 0 and <= {1})".format(int(bad.sum()), MAX_RUNTIME_H)
            )

    if "energy_kwh" in energy.columns:
        kwh = pd.to_numeric(energy["energy_kwh"], errors="coerce")
        bad = kwh.isna() | (kwh <= 0)
        if bad.any():
            errors.append(
                "invalid_energy_kwh: energy.csv -> energy_kwh ({0} rows, "
                "must be > 0)".format(int(bad.sum()))
            )

    if "efficiency_factor" in energy.columns:
        efficiency = pd.to_numeric(energy["efficiency_factor"], errors="coerce")
        bad = efficiency.isna() | (efficiency < MIN_EFFICIENCY_FACTOR)
        if bad.any():
            errors.append(
                "invalid_efficiency_factor: energy.csv -> efficiency_factor "
                "({0} rows, must be >= {1})".format(
                    int(bad.sum()), MIN_EFFICIENCY_FACTOR
                )
            )


def _check_date_ranges(frames: dict, errors: list) -> None:
    energy = frames["energy.csv"]
    production = frames["production.csv"]
    if energy["date"].isna().any() or production["date"].isna().any():
        return

    energy_start, energy_end = energy["date"].min(), energy["date"].max()
    production_start, production_end = production["date"].min(), production["date"].max()

    if (production_start, production_end) != (energy_start, energy_end):
        errors.append(
            "incoherent_date_range: production.csv {0}..{1} does not match "
            "energy.csv {2}..{3}".format(
                production_start.date(),
                production_end.date(),
                energy_start.date(),
                energy_end.date(),
            )
        )

    for filename in ("maintenance.csv", "incidents.csv", "technician_notes.csv"):
        frame = frames[filename]
        if frame.empty or frame["date"].isna().any():
            continue
        if frame["date"].min() < energy_start or frame["date"].max() > energy_end:
            errors.append(
                "incoherent_date_range: {0} {1}..{2} falls outside "
                "energy.csv {3}..{4}".format(
                    filename,
                    frame["date"].min().date(),
                    frame["date"].max().date(),
                    energy_start.date(),
                    energy_end.date(),
                )
            )


def _summary(frames: dict) -> dict:
    energy = frames["energy.csv"]
    production = frames["production.csv"]
    energy_start = energy["date"].min()
    energy_end = energy["date"].max()
    return {
        "data_dir": str(DATA_DIR),
        "machines": int(len(frames["machines.csv"])),
        "energy_rows": int(len(energy)),
        "production_rows": int(len(production)),
        "maintenance_rows": int(len(frames["maintenance.csv"])),
        "incidents": int(len(frames["incidents.csv"])),
        "technician_notes": int(len(frames["technician_notes.csv"])),
        "energy_start": None if pd.isna(energy_start) else str(energy_start.date()),
        "energy_end": None if pd.isna(energy_end) else str(energy_end.date()),
        "scenarios": len(load_scenarios()),
    }


def validate_dataset() -> dict:
    """Check basic structural integrity of the dataset.

    Returns a report dictionary with ``ok``, ``errors`` and ``summary`` keys.
    The dataset files are never modified. Problems are collected rather than
    raised so a caller can report every finding in a single pass.
    """
    errors: list = []

    if not _check_files(errors):
        return {"ok": False, "errors": errors, "summary": {}}

    frames = {
        "machines.csv": load_machines(),
        "energy.csv": load_energy(),
        "production.csv": load_production(),
        "maintenance.csv": load_maintenance(),
        "incidents.csv": load_incidents(),
        "technician_notes.csv": load_technician_notes(),
    }

    _check_columns(frames, errors)
    _check_nulls(frames, errors)
    _check_machine_ids(frames, errors)
    _check_energy(frames["energy.csv"], errors)

    if {"date"}.issubset(frames["energy.csv"].columns) and {"date"}.issubset(
        frames["production.csv"].columns
    ):
        _check_date_ranges(frames, errors)

    return {"ok": not errors, "errors": errors, "summary": _summary(frames)}