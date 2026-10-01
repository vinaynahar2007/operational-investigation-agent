import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools import analysis
from app.tools.anomalies import (
    METRIC_EFFICIENCY,
    METRIC_INTENSITY,
    METRIC_RUNTIME,
    MIN_ANOMALY_DAYS,
    detect_anomalies,
    detect_anomalies_for_machine,
)
from app.tools.dataset import load_energy


@pytest.fixture(scope="module")
def findings():
    return detect_anomalies()


@pytest.fixture(scope="module")
def by_machine(findings):
    return {finding["machine_id"]: finding for finding in findings}


def _synthetic_events(events, days=120, machine_id="X01"):
    """Build a synthetic machine with explicit drifted index ranges.

    ``events`` is a list of ``(start_index, end_index)`` inclusive pairs. The
    baseline is otherwise perfectly flat, so any finding must come from the
    injected steps rather than from noise.
    """
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rows = []
    for index, date in enumerate(dates):
        drifted = any(start <= index <= end for start, end in events)
        runtime = 8.0
        intensity = 24.0 if drifted else 20.0
        rows.append(
            {
                "date": date,
                "machine_id": machine_id,
                "runtime_h": runtime,
                "energy_kwh": runtime * intensity,
                "efficiency_factor": 1.0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# analysis primitives
# --------------------------------------------------------------------------


def test_machine_intensity_is_kwh_per_runtime_hour():
    frame = analysis.machine_intensity()
    assert analysis.INTENSITY_COLUMN in frame.columns
    expected = frame["energy_kwh"] / frame["runtime_h"]
    pd.testing.assert_series_equal(
        frame[analysis.INTENSITY_COLUMN], expected, check_names=False
    )


def test_machine_intensity_handles_zero_and_missing_runtime():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "machine_id": ["X", "X", "X"],
            "runtime_h": [0.0, None, 4.0],
            "energy_kwh": [10.0, 10.0, 40.0],
        }
    )
    result = analysis.machine_intensity(frame)[analysis.INTENSITY_COLUMN]
    assert result.isna().tolist() == [True, True, False]
    assert result.iloc[2] == pytest.approx(10.0)
    assert not result.isin([float("inf"), float("-inf")]).any()


def test_trailing_baseline_excludes_the_current_observation():
    series = pd.Series([1.0] * 5 + [99.0])
    baseline = analysis.trailing_median_baseline(series, window=5, min_periods=5)
    assert baseline.iloc[5] == pytest.approx(1.0)


def test_trailing_baseline_is_nan_until_enough_history():
    series = pd.Series(range(20), dtype=float)
    baseline = analysis.trailing_median_baseline(series, window=30, min_periods=10)
    assert baseline.iloc[:10].isna().all()
    assert baseline.iloc[10:].notna().all()


def test_deviation_ratio_guards_non_positive_baseline():
    ratio = analysis.deviation_ratio(pd.Series([10.0, 10.0]), pd.Series([0.0, 4.0]))
    assert pd.isna(ratio.iloc[0])
    assert ratio.iloc[1] == pytest.approx(2.5)


def test_runtime_deviation_adds_ratio_column():
    frame = analysis.runtime_deviation()
    assert "runtime_ratio" in frame.columns
    clean = frame[frame["machine_id"] == "M01"]["runtime_ratio"].dropna()
    assert clean.between(0.8, 1.2).all()


def test_plant_energy_per_unit_matches_manual_division():
    plant = analysis.plant_energy_per_unit()
    energy = load_energy()
    expected_total = energy.groupby("date")["energy_kwh"].sum()
    merged = plant.set_index("date")["total_energy_kwh"]
    pd.testing.assert_series_equal(
        merged.sort_index(), expected_total.sort_index(), check_names=False
    )
    assert plant["plant_kwh_per_unit"].notna().all()


def test_summarize_window_returns_deterministic_averages():
    summary = analysis.summarize_window("M03", "2026-09-14", "2026-09-28")
    assert summary["start_date"] == "2026-09-14"
    assert summary["end_date"] == "2026-09-28"
    assert summary["days"] == 15
    assert summary["average_runtime_h"] > 9.0
    assert summary["average_intensity"] > 0
    assert summary["production_change_pct"] is not None


# --------------------------------------------------------------------------
# ground-truth detections
# --------------------------------------------------------------------------


def test_m04_intensity_anomaly_is_detected(by_machine):
    finding = by_machine["M04"]
    assert finding["metric"] == METRIC_INTENSITY
    assert 13.0 <= finding["magnitude_pct"] <= 18.0
    assert finding["anomaly_type"] == "equipment_efficiency"


def test_m04_window_matches_ground_truth(by_machine):
    finding = by_machine["M04"]
    assert finding["start_date"] == "2026-08-30"
    assert finding["end_date"] == "2026-09-28"
    assert finding["days"] == 30


def test_m04_runtime_stays_flat_so_it_is_not_a_scheduling_anomaly(by_machine):
    assert by_machine["M04"]["mean_runtime_ratio"] == pytest.approx(1.0, abs=0.06)


def test_m03_runtime_anomaly_is_detected(by_machine):
    finding = by_machine["M03"]
    assert finding["metric"] == METRIC_RUNTIME
    assert finding["magnitude_pct"] >= 20.0
    assert finding["anomaly_type"] == "operational_scheduling"


def test_m03_window_matches_ground_truth(by_machine):
    finding = by_machine["M03"]
    assert finding["start_date"] == "2026-09-14"
    assert finding["end_date"] == "2026-09-28"


def test_m03_intensity_stays_flat_so_it_is_not_an_equipment_anomaly(by_machine):
    assert by_machine["M03"]["mean_intensity_ratio"] == pytest.approx(1.0, abs=0.05)


def test_m05_weak_efficiency_anomaly_is_detected_and_graded_low(by_machine):
    finding = by_machine["M05"]
    assert finding["metric"] == METRIC_EFFICIENCY
    assert finding["severity"] == "low"
    assert 4.0 <= finding["magnitude_pct"] <= 8.0
    assert finding["start_date"] == "2026-08-10"
    assert finding["end_date"] == "2026-08-24"


def test_runtime_and_intensity_anomalies_are_not_collapsed_together(by_machine):
    """The whole point: different metric and different cause label."""
    assert by_machine["M03"]["metric"] == METRIC_RUNTIME
    assert by_machine["M04"]["metric"] == METRIC_INTENSITY
    assert by_machine["M03"]["anomaly_type"] != by_machine["M04"]["anomaly_type"]


def test_m01_and_m02_remain_clean(findings):
    detected = {finding["machine_id"] for finding in findings}
    assert "M01" not in detected
    assert "M02" not in detected


def test_each_machine_yields_a_single_finding(findings):
    machine_ids = [finding["machine_id"] for finding in findings]
    assert len(machine_ids) == len(set(machine_ids))


def test_finding_structure_contains_required_keys(findings):
    required = {
        "machine_id",
        "start_date",
        "end_date",
        "metric",
        "ratio",
        "magnitude_pct",
        "severity",
        "confidence",
    }
    for finding in findings:
        assert required <= set(finding)
        assert 0.0 <= finding["confidence"] <= 1.0
        assert finding["severity"] in {"low", "medium", "high"}
        assert finding["start_date"] <= finding["end_date"]


def test_baseline_freezing_keeps_long_anomaly_open_to_its_end():
    """A 35-day anomaly must not be truncated as the median absorbs it."""
    frame = _synthetic_events([(40, 74)])
    detected = detect_anomalies_for_machine("X01", frame)
    assert len(detected) == 1
    assert detected[0]["start_date"] == "2026-02-10"
    assert detected[0]["end_date"] == "2026-03-16"
    assert detected[0]["days"] == 35


# --------------------------------------------------------------------------
# synthetic data behaviour
# --------------------------------------------------------------------------


def test_synthetic_clean_dataset_produces_no_anomaly():
    assert detect_anomalies_for_machine("X01", _synthetic_events([])) == []


def test_synthetic_intensity_step_is_detected_as_intensity():
    detected = detect_anomalies_for_machine("X01", _synthetic_events([(40, 74)]))
    assert len(detected) == 1
    assert detected[0]["metric"] == METRIC_INTENSITY
    assert detected[0]["magnitude_pct"] == pytest.approx(20.0, abs=2.0)


def test_synthetic_runtime_step_is_detected_as_runtime_not_intensity():
    """Runtime up, intensity flat: this must read as scheduling, not equipment."""
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    rows = [
        {
            "date": date,
            "machine_id": "X01",
            "runtime_h": 11.0 if 40 <= index <= 74 else 8.0,
            "energy_kwh": (11.0 if 40 <= index <= 74 else 8.0) * 20.0,
            "efficiency_factor": 1.0,
        }
        for index, date in enumerate(dates)
    ]
    detected = detect_anomalies_for_machine("X01", pd.DataFrame(rows))
    assert len(detected) == 1
    assert detected[0]["metric"] == METRIC_RUNTIME
    assert detected[0]["anomaly_type"] == "operational_scheduling"


def test_contiguous_rows_collapse_into_one_window():
    """35 consecutive bad days must not become 35 separate findings."""
    detected = detect_anomalies_for_machine("X01", _synthetic_events([(40, 74)]))
    assert len(detected) == 1
    assert detected[0]["days"] == 35


def test_two_separated_events_stay_separate():
    frame = _synthetic_events([(40, 55), (80, 95)])
    detected = detect_anomalies_for_machine("X01", frame)
    assert len(detected) == 2
    starts = sorted(finding["start_date"] for finding in detected)
    assert starts == ["2026-02-10", "2026-03-22"]


def test_single_anomalous_day_is_not_reported_as_an_event():
    frame = _synthetic_events([(45, 46)])
    assert detect_anomalies_for_machine("X01", frame) == []


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------


def test_zero_runtime_rows_do_not_crash_the_detector():
    frame = _synthetic_events([(40, 74)])
    frame.loc[40:45, "runtime_h"] = 0.0
    detected = detect_anomalies_for_machine("X01", frame)
    assert isinstance(detected, list)


def test_missing_values_do_not_crash_the_detector():
    frame = _synthetic_events([(40, 74)])
    frame.loc[50:55, "energy_kwh"] = None
    detected = detect_anomalies_for_machine("X01", frame)
    assert isinstance(detected, list)


def test_insufficient_history_produces_no_anomaly():
    """Shorter than the baseline warm-up, there is nothing to judge against."""
    frame = _synthetic_events([(1, 3)], days=5)
    assert detect_anomalies_for_machine("X01", frame) == []


def test_empty_machine_returns_no_findings():
    frame = _synthetic_events([])
    assert detect_anomalies_for_machine("NOPE", frame) == []


def test_detection_is_deterministic_across_runs():
    first = detect_anomalies()
    second = detect_anomalies()
    assert first == second


def test_unknown_machine_filter_returns_empty(findings):
    assert detect_anomalies_for_machine("M99") == []