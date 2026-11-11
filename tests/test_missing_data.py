"""Tests for how missing atomic data is handled.

xraylib raises ValueError when it has no data (e.g. no K shell for H).  Only
that error may mean "skip this element"; any other error is a real problem
and must stop the run instead of silently dropping an element.
"""
import pytest

from xrf_model.pipeline import _k_shell_params


def test_hydrogen_has_no_k_shell_data():
    assert _k_shell_params(1) is None


def test_iron_has_k_shell_data():
    p = _k_shell_params(26)
    assert p["E_edge"] == pytest.approx(7.112, abs=1e-3)
    assert 0 < p["wK"] < 1 and 0 < p["R_jump"] < 1


def test_a_real_error_is_not_swallowed():
    with pytest.raises(TypeError):
        _k_shell_params("Fe")            # element symbol instead of atomic number
