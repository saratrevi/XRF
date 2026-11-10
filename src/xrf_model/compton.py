# src/xrf_model/compton.py
"""
Doppler broadening of Compton scattering (impulse approximation).

Electrons in atoms move, so a photon of energy E scattered at angle θ does not
leave at the single Compton energy E_C but over a spread of energies.  In the
impulse approximation the spread follows the Compton profile J(p_z), the
distribution of the electron momentum along the scattering vector:

    p_z(E, E') = −137.036 · (E − E' − E·E'·(1 − cos θ)/mc²)
                 / √(E² + E'² − 2·E·E'·cos θ)                       [a.u.]

and an electron with binding energy U takes part only if E − E' > U.

Shell profiles come from xraylib (Biggs et al. 1975, one electron per
profile), weighted by the electrons in each shell and by the sample's
electrons per gram.  Only the shape comes from here: the total Compton
intensity per incident energy stays the Klein–Nishina × S(x, Z) value from
physics.py.  The change of the Klein–Nishina factor across the peak (a few
percent) is neglected.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import xraylib

MC2_KEV = 510.99895
INV_ALPHA = 137.035999      # 1/α: converts momentum from units of mc to atomic units
P_MAX = 100.0               # a.u., end of the xraylib profile tables
DP = 0.01                   # a.u.
_P = np.linspace(-P_MAX, P_MAX, int(round(2 * P_MAX / DP)) + 1)
_ROWS_PER_CHUNK = 256


@lru_cache(maxsize=None)
def element_shells(Z: int) -> tuple:
    """
    Shells of element Z as (binding energy U [keV], electrons, CDF over _P).

    The CDF is that of one electron's profile, going from 0 to 1.  Electrons
    missing from xraylib's shell profiles (e.g. one 3p sub-shell of Al) are
    given to the outermost shell, which is where they belong.
    """
    p_pos = _P[_P >= 0]
    shells = []
    for s in range(40):
        try:
            J = np.array([xraylib.ComptonProfile_Partial(Z, s, float(p)) for p in p_pos])
            n = xraylib.ElectronConfig(Z, s)
        except ValueError:
            continue
        try:
            U = xraylib.EdgeEnergy(Z, s)
        except ValueError:
            U = 0.0                      # valence shell without tabulated edge
        J_full = np.concatenate([J[:0:-1], J])          # J(|p|), mirrored
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (J_full[1:] + J_full[:-1]) * DP)])
        shells.append([U, n, cdf / cdf[-1]])
    missing = Z - sum(sh[1] for sh in shells)
    if missing > 1e-6:
        min(shells, key=lambda sh: sh[0])[1] += missing
    return tuple((U, n, cdf) for U, n, cdf in shells)


def sample_profile(param_dict: dict, conc: dict) -> dict:
    """
    Compton profile table of a sample.

    Returns {"U": binding energies sorted ascending, "C": cumulative table}
    where C[k] is the CDF over p_z of the k lowest-binding shells together,
    weighted by electrons per gram and normalised so that C[-1] ends at 1.
    """
    shells = []
    for el, w in conc.items():
        if w <= 0:
            continue
        Z = param_dict[el]["Atomic Number"]
        A = xraylib.AtomicWeight(Z)
        for U, n, cdf in element_shells(Z):
            shells.append((U, w / A * n, cdf))
    shells.sort(key=lambda sh: sh[0])
    weights = np.array([sh[1] for sh in shells])
    weights /= weights.sum()
    C = np.zeros((len(shells) + 1, _P.size))
    for k, ((_, _, cdf), wt) in enumerate(zip(shells, weights)):
        C[k + 1] = C[k] + wt * cdf
    return {"U": np.array([sh[0] for sh in shells]), "C": C}


def _lookup(C: np.ndarray, k: np.ndarray, p: np.ndarray) -> np.ndarray:
    """C[k, p] with linear interpolation in p."""
    x = np.clip((p + P_MAX) / DP, 0.0, _P.size - 1 - 1e-9)
    i0 = x.astype(np.intp)
    f = x - i0
    return C[k, i0] * (1.0 - f) + C[k, i0 + 1] * f


def compton_kernel(profile: dict, energy_in: np.ndarray, out_edges: np.ndarray,
                   theta: float) -> np.ndarray:
    """
    K[i, j] [1/keV]: probability density that a photon of energy_in[i],
    scattered at angle theta, leaves in output bin j (edges out_edges).
    Each row integrates to 1 over the output bins.

    Each bin gets the profile's weight between the p_z of its two edges,
    with the shells open at the bin centre (binding energy below E − E').
    If no shell is open anywhere (very low energy), the photon goes to the
    bin of the free-electron Compton energy.
    """
    cos_t = np.cos(theta)
    widths = np.diff(out_edges)
    centres = 0.5 * (out_edges[:-1] + out_edges[1:])
    K = np.zeros((energy_in.size, widths.size))
    for start in range(0, energy_in.size, _ROWS_PER_CHUNK):
        E = energy_in[start:start + _ROWS_PER_CHUNK, None]
        Ee = out_edges[None, :]
        p_edges = (-INV_ALPHA * (E - Ee - E * Ee * (1.0 - cos_t) / MC2_KEV)
                   / np.sqrt(E ** 2 + Ee ** 2 - 2.0 * E * Ee * cos_t))
        k_open = np.searchsorted(profile["U"], E - centres[None, :], side="left")
        prob = np.clip(_lookup(profile["C"], k_open, p_edges[:, 1:])
                       - _lookup(profile["C"], k_open, p_edges[:, :-1]), 0.0, None)
        total = prob.sum(axis=1)
        rows = np.arange(start, start + E.shape[0])
        ok = total > 0
        K[rows[ok]] = prob[ok] / total[ok, None]
        if not ok.all():
            E_c = E[~ok, 0] / (1.0 + E[~ok, 0] / MC2_KEV * (1.0 - cos_t))
            j = np.clip(np.searchsorted(out_edges, E_c) - 1, 0, widths.size - 1)
            K[rows[~ok], j] = 1.0
    return K / widths[None, :]


def bin_edges(centres: np.ndarray) -> np.ndarray:
    """Edges halfway between bin centres, ends extended by half a bin."""
    mid = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[2 * centres[0] - mid[0]], mid, [2 * centres[-1] - mid[-1]]])
