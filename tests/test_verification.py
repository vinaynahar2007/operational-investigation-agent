import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.intervention import simulate_intervention
from app.tools.verification import (
    DEFAULT_THRESHOLD_PCT,
    STATUS_FAIL,
    STATUS_PASS,
    verify_intervention,
)


# --------------------------------------------------------------------------
# PASS / FAIL
# --------------------------------------------------------------------------


def test_pass_when_threshold_is_met():
    result = verify_intervention(400.0, 360.0, 10.0, 5.0)
    assert result["status"] == STATUS_PASS


def test_fail_when_threshold_is_not_met():
    result = verify_intervention(400.0, 396.0, 1.0, 5.0)
    assert result["status"] == STATUS_FAIL


def test_pass_exactly_at_threshold():
    """Meeting the threshold exactly must pass (>= not >)."""
    result = verify_intervention(100.0, 95.0, 5.0, 5.0)
    assert result["status"] == STATUS_PASS


def test_fail_just_below_threshold():
    result = verify_intervention(100.0, 95.01, 4.99, 5.0)
    assert result["status"] == STATUS_FAIL


def test_zero_projection_reduction_fails_a_positive_threshold():
    result = verify_intervention(500.0, 500.0, 0.0, 5.0)
    assert result["status"] == STATUS_FAIL
    assert result["observed_reduction_percent"] == 0.0


def test_worse_than_baseline_projection_fails():
    """Energy above baseline is a negative reduction and must not pass."""
    result = verify_intervention(400.0, 450.0, 10.0, 5.0)
    assert result["status"] == STATUS_FAIL
    assert result["observed_reduction_percent"] < 0


# --------------------------------------------------------------------------
# arithmetic correctness
# --------------------------------------------------------------------------


def test_observed_reduction_is_calculated_not_copied():
    """Assumed 30%, actual energies imply 10% -> must report 10%."""
    result = verify_intervention(200.0, 180.0, 30.0, 5.0)
    assert result["observed_reduction_percent"] == 10.0
    assert result["expected_reduction_percent"] == 30.0


def test_calculated_reduction_matches_manual_arithmetic():
    baseline, projected = 415.34, 381.99
    expected = (baseline - projected) / baseline * 100
    result = verify_intervention(baseline, projected, 8.03, 5.0)
    assert result["observed_reduction_percent"] == pytest.approx(expected, abs=0.01)


def test_mismatch_between_assumption_and_energies_is_flagged():
    result = verify_intervention(200.0, 180.0, 30.0, 5.0)
    assert "does not match" in result["reason"]


def test_simulated_projection_verifies_against_its_own_energies():
    """A real simulation output must verify consistently."""
    projection = simulate_intervention(415.34, "Restore efficiency", 8.03)
    result = verify_intervention(
        projection["baseline_energy"],
        projection["projected_energy"],
        projection["projected_reduction_percent"],
        DEFAULT_THRESHOLD_PCT,
    )
    assert result["status"] == STATUS_PASS
    assert result["observed_reduction_percent"] == pytest.approx(8.03, abs=0.05)


def test_default_threshold_is_used_when_omitted():
    result = verify_intervention(400.0, 396.0, 1.0)
    assert result["threshold_percent"] == DEFAULT_THRESHOLD_PCT
    assert result["status"] == STATUS_FAIL


# --------------------------------------------------------------------------
# epistemic labelling
# --------------------------------------------------------------------------


def test_simulation_flag_is_always_true():
    result = verify_intervention(400.0, 360.0, 10.0, 5.0)
    assert result["simulation"] is True


def test_reason_is_labelled_simulated():
    result = verify_intervention(400.0, 360.0, 10.0, 5.0)
    assert result["reason"].startswith("SIMULATED")


def test_required_keys_are_present():
    result = verify_intervention(400.0, 360.0, 10.0, 5.0)
    assert {
        "status",
        "baseline_energy",
        "projected_energy",
        "observed_reduction_percent",
        "threshold_percent",
        "simulation",
        "reason",
    } <= set(result)


# --------------------------------------------------------------------------
# invalid inputs
# --------------------------------------------------------------------------


def test_zero_baseline_is_rejected():
    with pytest.raises(ValueError):
        verify_intervention(0.0, 0.0, 10.0, 5.0)


def test_negative_projected_energy_is_rejected():
    with pytest.raises(ValueError):
        verify_intervention(100.0, -5.0, 10.0, 5.0)


def test_non_numeric_inputs_are_rejected():
    with pytest.raises(ValueError):
        verify_intervention("x", 10.0, 10.0, 5.0)
    with pytest.raises(ValueError):
        verify_intervention(100.0, "x", 10.0, 5.0)


def test_zero_threshold_passes_any_reduction():
    result = verify_intervention(100.0, 99.9, 0.1, 0.0)
    assert result["status"] == STATUS_PASS