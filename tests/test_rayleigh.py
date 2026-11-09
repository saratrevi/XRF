"""Regression tests for the coherent (Rayleigh) cross section.

xraylib.Fi returns the anomalous correction f' (at most a few electrons), not
the FFAST-style f1 = Z + f' this code used before moving to xraylib. The atomic
amplitude is therefore f0 + f' + i*f'', with no -Z term.
"""
import numpy as np
import pytest
import xraylib

from xrf_model.physics import (
    compute_coherent_cross_section,
    precompute_anomalous_factors,
)

Z_VALUES = [8, 14, 26, 47]  # O, Si, Fe, Ag
ENERGIES = np.array([5.0, 10.0, 20.0, 40.0])  # keV
ANGLE = np.radians(90.0)


def _pure_element(Z):
    params = {"X": {"Atomic Number": Z, "Atomic Mass": xraylib.AtomicWeight(Z)}}
    return params, {"X": 1.0}


@pytest.mark.parametrize("Z", Z_VALUES)
def test_matches_dcs_rayl_without_anomalous_terms(Z):
    """With f' = f'' = 0 the result must equal xraylib's own Rayleigh DCS."""
    params, conc = _pure_element(Z)
    zeros = np.zeros_like(ENERGIES)
    cache = {"X": {"f1": zeros, "f2": zeros}}

    result = compute_coherent_cross_section(ENERGIES, ANGLE, params, conc, cache)

    expected = np.array([xraylib.DCS_Rayl(Z, float(E), ANGLE) for E in ENERGIES])
    assert result == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("Z", Z_VALUES)
def test_anomalous_terms_correct_the_form_factor(Z, use_cache):
    """The DCS must scale by |f0 + f' + i*f''|^2 / f0^2, on both code paths."""
    params, conc = _pure_element(Z)
    cache = precompute_anomalous_factors(ENERGIES, params) if use_cache else None

    result = compute_coherent_cross_section(ENERGIES, ANGLE, params, conc, cache)

    expected = []
    for E in map(float, ENERGIES):
        f0 = xraylib.FF_Rayl(Z, xraylib.MomentTransf(E, ANGLE))
        scale = ((f0 + xraylib.Fi(Z, E)) ** 2 + xraylib.Fii(Z, E) ** 2) / f0 ** 2
        expected.append(xraylib.DCS_Rayl(Z, E, ANGLE) * scale)
    assert result == pytest.approx(np.array(expected), rel=1e-3)
