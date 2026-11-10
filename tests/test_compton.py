"""Tests for the Doppler broadening of Compton scattering.

The kernel spreads each scattered photon over E' with the sample's Compton
profile. It must keep every photon, put the peak at the Compton energy, have
the width the profile predicts, and leave out electrons whose binding energy
is larger than the energy the photon gives up.
"""
import numpy as np
import pytest
import xraylib

from xrf_model.compton import DP, _P, compton_kernel, element_shells, sample_profile
from xrf_model.energy_grid import bin_edges

THETA = np.pi / 2
MC2 = 510.99895


def compton_energy(E, theta=THETA):
    return E / (1 + E / MC2 * (1 - np.cos(theta)))


def soil_like():
    params = {el: {"Atomic Number": xraylib.SymbolToAtomicNumber(el)} for el in ("O", "Si", "Fe")}
    return sample_profile(params, {"O": 0.5, "Si": 0.3, "Fe": 0.2})


def single_row(profile, E, lo, hi, step=0.002, theta=THETA):
    centres = np.arange(lo, hi, step)
    return centres, compton_kernel(profile, np.array([E]), bin_edges(centres), theta)[0]


def fwhm(x, y):
    h = y.max() / 2
    i = np.where(y > h)[0]
    return (np.interp(h, [y[i[-1] + 1], y[i[-1]]], [x[i[-1] + 1], x[i[-1]]])
            - np.interp(h, [y[i[0] - 1], y[i[0]]], [x[i[0] - 1], x[i[0]]]))


@pytest.mark.parametrize("Z", [8, 13, 14, 26])
def test_shells_hold_all_the_electrons(Z):
    shells = element_shells(Z)
    assert sum(n for _, n, _ in shells) == pytest.approx(Z, abs=1e-6)
    for _, _, cdf in shells:
        assert cdf[0] == 0.0 and cdf[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("Z", [8, 14, 26])
def test_shell_profiles_add_up_to_the_element_profile(Z):
    total_cdf = sum(n * cdf for _, n, cdf in element_shells(Z))
    J = np.gradient(total_cdf, DP)
    for p in (0.0, 0.5, 1.0, 2.0, 4.0):
        i = int(np.argmin(np.abs(_P - p)))
        assert J[i] == pytest.approx(xraylib.ComptonProfile(Z, p), rel=1e-2)


def test_every_photon_is_kept():
    energy = 1.025 + 0.03 * np.arange(1000)                 # 1 … 31 keV
    K = compton_kernel(soil_like(), energy, bin_edges(energy), THETA)
    np.testing.assert_allclose((K * np.diff(bin_edges(energy))).sum(axis=1), 1.0, rtol=1e-9)


def test_peak_sits_at_the_compton_energy():
    E_c = compton_energy(22.1)
    x, k = single_row(soil_like(), 22.1, 19.0, 23.0)
    assert x[np.argmax(k)] == pytest.approx(E_c, abs=0.005)


def test_width_follows_the_profile():
    # oxygen only: at 22.1 keV and 90° the photon gives up ~0.9 keV, more than
    # the O K binding (0.53 keV), so all electrons take part near the peak
    oxygen = sample_profile({"O": {"Atomic Number": 8}}, {"O": 1.0})
    p = np.linspace(-10, 10, 20001)
    J = np.array([xraylib.ComptonProfile(8, abs(v)) for v in p])
    above = p[J > J.max() / 2]
    fwhm_p = above.max() - above.min()
    E, E_c = 22.1, compton_energy(22.1)
    pz = lambda Ep: (-137.035999 * (E - Ep - E * Ep / MC2)
                     / np.sqrt(E ** 2 + Ep ** 2))                # θ = 90°
    dE_dp = 2e-4 / (pz(E_c + 1e-4) - pz(E_c - 1e-4))
    x, k = single_row(oxygen, E, 20.0, 22.5, step=0.001)
    assert fwhm(x, k) == pytest.approx(fwhm_p * dE_dp, rel=2e-2)


def test_bound_electrons_need_enough_energy():
    # Fe K electrons only (binding 7.11 keV): a 30 keV photon must give up more
    # than 7.11 keV, so nothing can appear above 30 − 7.11 keV
    U, n, cdf = element_shells(26)[0]
    k_shell = {"U": np.array([U]), "C": np.vstack([np.zeros_like(cdf), cdf])}
    x, k = single_row(k_shell, 30.0, 10.0, 31.0, step=0.01)
    assert np.all(k[x > 30.0 - U + 0.01] == 0.0)
    assert np.sum(k * 0.01) == pytest.approx(1.0, rel=1e-9)


def test_no_open_shell_puts_the_photon_at_the_compton_energy():
    U, n, cdf = element_shells(26)[0]
    k_shell = {"U": np.array([U]), "C": np.vstack([np.zeros_like(cdf), cdf])}
    x, k = single_row(k_shell, 3.0, 2.5, 3.5, step=0.01)    # 3 keV < Fe K binding
    assert np.count_nonzero(k) == 1
    assert x[np.argmax(k)] == pytest.approx(compton_energy(3.0), abs=0.01)
