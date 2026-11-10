"""Tests for the Si detection efficiency used by the two detector paths.

The efficiency must count the interactions that the response places in the
spectrum: photoelectric only for Gaussian peaks (every photon at its full
energy), photoelectric + Compton for a matrix normalised per photon that left
energy in the crystal. Rayleigh scattering leaves no energy and never counts.
"""
import numpy as np
import pandas as pd
import pytest
import xraylib

from xrf_model.detector_response import build_spectrum_with_response, load_response
from xrf_model.spectrum import build_spectrum, qe_active_si, transmission_be

RHO_SI, D_UM, BE_MM = 2.329, 450.0, 0.0127
E = np.array([5.0, 15.75, 22.0, 30.0, 40.0])
BROADENING = dict(fano=0.118, noise_ev=12.7, w_ev=3.65)


def beer(mu):
    return 1.0 - np.exp(-mu * RHO_SI * D_UM * 1e-4)


def test_photoelectric_efficiency_leaves_out_compton_and_rayleigh():
    expected = beer(np.array([xraylib.CS_Photo(14, e) for e in E]))
    np.testing.assert_allclose(qe_active_si(E, D_UM), expected, rtol=1e-12)


def test_matrix_efficiency_adds_compton_but_not_rayleigh():
    expected = beer(np.array([xraylib.CS_Photo(14, e) + xraylib.CS_Compt(14, e) for e in E]))
    np.testing.assert_allclose(qe_active_si(E, D_UM, include_compton=True), expected,
                               rtol=1e-12)


def test_gaussian_peak_holds_only_the_photoabsorbed_photons():
    grid = 20.0 + 0.01 * np.arange(401)                 # 20 … 24 keV
    line = pd.DataFrame({"E_line_keV": [22.0], "Intensity_ph_s": [100.0]})
    S = build_spectrum(grid, np.zeros_like(grid), line,
                       be_mm=BE_MM, si_active_um=D_UM, **BROADENING)
    # expected value from xraylib directly, not from the functions under test
    window = np.exp(-xraylib.CS_Total(4, 22.0) * 1.848 * BE_MM / 10.0)
    photoabsorbed = beer(xraylib.CS_Photo(14, 22.0))
    assert np.trapezoid(S, grid) == pytest.approx(100.0 * window * photoabsorbed, rel=1e-4)


def test_matrix_path_counts_compton_events_too(tmp_path):
    # delta matrix: every detected photon lands at its own energy
    e_in = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
    n_ch = 9000
    resp = np.zeros((e_in.size, n_ch))
    resp[np.arange(e_in.size), np.round(e_in / 0.005).astype(int)] = 1.0
    np.savez(tmp_path / "m.npz", energy_in_keV=e_in,
             channel_edges_keV=(np.arange(n_ch + 1) - 0.5) * 0.005, response=resp)
    line = pd.DataFrame({"E_line_keV": [30.0], "Intensity_ph_s": [100.0]})
    no_continuum = np.zeros(3)
    _, S = build_spectrum_with_response(
        np.array([10.0, 10.03, 10.06]), no_continuum, line, load_response(tmp_path / "m.npz"),
        rows="per_detected_photon", be_mm=BE_MM, si_active_um=D_UM, **BROADENING)
    e = np.array([30.0])
    T = transmission_be(e, BE_MM)[0]
    total = np.sum(S) * 0.005
    assert total == pytest.approx(100.0 * T * qe_active_si(e, D_UM, include_compton=True)[0],
                                  rel=1e-9)
    assert total > 1.1 * 100.0 * T * qe_active_si(e, D_UM)[0]   # clearly not photoelectric only
