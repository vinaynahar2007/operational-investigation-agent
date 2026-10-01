import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools import dataset
from app.tools.dataset import (
    MAX_RUNTIME_H,
    MIN_EFFICIENCY_FACTOR,
    validate_dataset,
)


def test_all_loaders_return_dataframes():
    machines = dataset.load_machines()
    assert list(machines.columns) == [
        "machine_id",
        "machine_name",
        "category",
        "rated_kw",
        "baseline_runtime_h",
    ]
    assert len(dataset.load_energy()) > 0
    assert len(dataset.load_production()) > 0
    assert len(dataset.load_maintenance()) > 0
    assert len(dataset.load_incidents()) > 0
    assert len(dataset.load_technician_notes()) > 0


def test_load_scenarios_returns_scenario_list():
    scenarios = dataset.load_scenarios()
    assert isinstance(scenarios, list)
    assert len(scenarios) == 3
    assert {"id", "name", "question"} <= set(scenarios[0])


def test_dates_are_parsed_as_datetime():
    energy = dataset.load_energy()
    production = dataset.load_production()
    assert pd.api.types.is_datetime64_any_dtype(energy["date"])
    assert pd.api.types.is_datetime64_any_dtype(production["date"])
    assert energy["date"].min() == pd.Timestamp("2026-07-01")


def test_paths_are_absolute_and_cwd_independent():
    assert dataset.DATA_DIR.is_absolute()
    assert dataset.DATA_DIR.is_dir()
    assert len(dataset.load_machines()) == 5


def test_validate_dataset_passes_for_repository_data():
    report = validate_dataset()
    assert report["ok"], report["errors"]
    assert report["errors"] == []


def test_validate_dataset_summary_reports_shape():
    summary = validate_dataset()["summary"]
    assert summary["machines"] == 5
    assert summary["energy_start"] == "2026-07-01"
    assert summary["energy_end"] == "2026-09-28"
    assert summary["scenarios"] == 3
    assert summary["energy_rows"] == summary["production_rows"] * 5


def test_documented_bounds():
    assert MAX_RUNTIME_H == 24.0
    assert MIN_EFFICIENCY_FACTOR == 1.0


def test_rated_kw_is_not_an_energy_ceiling():
    """M04 exceeds rated_kw * runtime_h in the source data; that is allowed."""
    machines = dataset.load_machines().set_index("machine_id")
    energy = dataset.load_energy()
    latest = energy.sort_values("date").groupby("machine_id").tail(1)

    implied_kw = latest["energy_kwh"] / latest["runtime_h"]
    overshoot = implied_kw - machines.loc[latest["machine_id"], "rated_kw"].to_numpy()

    assert overshoot.max() > 0
    assert validate_dataset()["ok"]


def test_validate_dataset_detects_broken_structures(monkeypatch):
    stamp = pd.to_datetime
    frames = {
        "load_machines": pd.DataFrame({"machine_id": ["M01"], "rated_kw": [18]}),
        "load_energy": pd.DataFrame(
            {
                "date": stamp(["2026-07-01", "2026-07-01"], format="%Y-%m-%d"),
                "machine_id": ["M01", "M01"],
                "runtime_h": [30.0, 0.0],
                "energy_kwh": [-1.0, 5.0],
                "efficiency_factor": [0.5, None],
            }
        ),
        "load_production": pd.DataFrame(
            {
                "date": stamp(["2026-08-01"], format="%Y-%m-%d"),
                "production_units": [1],
            }
        ),
        "load_maintenance": pd.DataFrame(
            {
                "date": stamp(["2026-07-01"], format="%Y-%m-%d"),
                "machine_id": ["M99"],
            }
        ),
        "load_incidents": pd.DataFrame(
            {
                "date": stamp(["2026-07-01"], format="%Y-%m-%d"),
                "machine_id": ["M01"],
            }
        ),
        "load_technician_notes": pd.DataFrame(
            {
                "date": stamp(["2026-07-01"], format="%Y-%m-%d"),
                "machine_id": ["M01"],
            }
        ),
    }
    for name, frame in frames.items():
        monkeypatch.setattr(
            dataset, name, lambda f=frame: f.copy()
        )

    report = validate_dataset()
    codes = {error.split(":")[0] for error in report["errors"]}

    assert report["ok"] is False
    assert "missing_column" in codes
    assert "null_value" in codes
    assert "unknown_machine_id" in codes
    assert "duplicate_energy_record" in codes
    assert "invalid_runtime_h" in codes
    assert "invalid_energy_kwh" in codes
    assert "invalid_efficiency_factor" in codes
    assert "incoherent_date_range" in codes
