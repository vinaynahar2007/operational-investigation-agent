import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.intervention import (
    RECOVERY_ASSUMPTION,
    expected_reduction_for,
    intervention_for,
    simulate_intervention,
)


# --------------------------------------------------------------------------
# valid calculations
# --------------------------------------------------------------------------


def test_valid_intervention_calculation():
    result = simulate_intervention(400.0, "Restore equipment efficiency", 10.0)
    assert result["intervention"] == "Restore equipment efficiency"
    assert result["baseline_energy"] == 400.0
    assert result["projected_energy"] == 360.0
    assert result["projected_reduction_percent"] == 10.0
    assert result["simulation"] is True


def test_projected_energy_calculation():
    """projected = baseline * (1 - reduction / 100)."""
    result = simulate_intervention(500.0, "anything", 25.0)
    assert result["projected_energy"] == pytest.approx(375.0)


def test_projected_energy_rounds_consistently():
    result = simulate_intervention(415.34, "anything", 8.03)
    assert result["projected_energy"] == pytest.approx(381.99, abs=0.01)
    assert result["projected_energy"] == round(result["projected_energy"], 2)


def test_zero_reduction_leaves_energy_unchanged():
    result = simulate_intervention(250.0, "anything", 0.0)
    assert result["projected_reduction_percent"] == 0.0
    assert result["projected_energy"] == 250.0
    assert result["simulation"] is True


def test_zero_baseline_is_allowed():
    result = simulate_intervention(0.0, "anything", 10.0)
    assert result["baseline_energy"] == 0.0
    assert result["projected_energy"] == 0.0


def test_near_full_reduction_leaves_a_small_remainder():
    result = simulate_intervention(1000.0, "anything", 99.9)
    assert result["projected_energy"] == pytest.approx(1.0, abs=0.01)


def test_intervention_is_deterministic():
    first = simulate_intervention(415.34, "x", 8.03)
    second = simulate_intervention(415.34, "x", 8.03)
    assert first == second


# --------------------------------------------------------------------------
# invalid inputs
# --------------------------------------------------------------------------


def test_negative_reduction_is_rejected():
    with pytest.raises(ValueError):
        simulate_intervention(100.0, "anything", -5.0)


def test_reduction_of_exactly_100_is_rejected():
    with pytest.raises(ValueError):
        simulate_intervention(100.0, "anything", 100.0)


def test_reduction_above_100_is_rejected():
    with pytest.raises(ValueError):
        simulate_intervention(100.0, "anything", 150.0)


def test_negative_baseline_is_rejected():
    with pytest.raises(ValueError):
        simulate_intervention(-100.0, "anything", 10.0)


def test_non_numeric_baseline_is_rejected():
    with pytest.raises(ValueError):
        simulate_intervention("abc", "anything", 10.0)


def test_non_numeric_reduction_is_rejected():
    with pytest.raises(ValueError):
        simulate_intervention(100.0, "anything", "abc")


def test_none_inputs_are_rejected():
    with pytest.raises(ValueError):
        simulate_intervention(None, "anything", 10.0)
    with pytest.raises(ValueError):
        simulate_intervention(100.0, "anything", None)


# --------------------------------------------------------------------------
# category selection and conservative assumptions
# --------------------------------------------------------------------------


def test_each_root_cause_category_maps_to_an_intervention():
    expected = {
        "equipment_efficiency": "Inspect operating condition and restore equipment to normal efficiency.",
        "operational_scheduling": "Review operating schedule and reduce unnecessary runtime.",
        "production_demand": "Review production-energy relationship and optimize energy use per unit.",
        "maintenance_condition": "Inspect maintenance/operating condition and correct the identified issue.",
    }
    for category, text in expected.items():
        assert intervention_for(category) == text


def test_unknown_category_falls_back_to_a_generic_intervention():
    assert intervention_for("something_new") == intervention_for("equipment_efficiency")


def test_expected_reduction_is_a_conservative_fraction_of_the_excess():
    """Assumptions must stay below the observed excess, never exceed it."""
    for category, share in RECOVERY_ASSUMPTION.items():
        assert 0.0 < share < 1.0
        reduction = expected_reduction_for(category, 20.0)
        assert 0.0 < reduction < 20.0


def test_expected_reduction_scales_with_observed_excess():
    small = expected_reduction_for("equipment_efficiency", 10.0)
    large = expected_reduction_for("equipment_efficiency", 40.0)
    assert large > small


def test_expected_reduction_handles_missing_excess():
    assert expected_reduction_for("equipment_efficiency", None) == 0.0


def test_negative_observed_excess_is_clamped_to_zero():
    assert expected_reduction_for("equipment_efficiency", -5.0) == 0.0