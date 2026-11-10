# src/xrf_model/detector_response.py
"""
Detector spectrum from a Monte Carlo response matrix.

The matrix gives, for photons of a set of energies, the distribution of the
energy they deposit in the detector, before electronic broadening.  It holds
the physics of the detector (escape peaks, Compton scattering in the Si,
partial deposits).  The electronics (Fano and noise broadening) are applied
here from the config, so changing them does not need a new matrix.

File format: a NumPy .npz with three arrays

    energy_in_keV      (N_in,)       photon energy of each row, increasing
    channel_edges_keV  (N_ch + 1,)   edges of the deposited-energy channels,
                                     all of the same width
    response           (N_in, N_ch)  probability that a photon of energy
                                     energy_in_keV[i] deposits an energy in
                                     channel j

Rows are either per detected photon, meaning per photon that left energy in
the crystal (each row sums to 1; the code applies the Be window and the Si
efficiency for photoelectric + Compton interactions), or per incident photon
(each row sums to the detection efficiency, so the window and efficiency are
already inside the matrix).

Not modelled: pile-up and dead time, which depend on the count rate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .spectrum import qe_active_si, transmission_be

ROW_OPTIONS = ("per_detected_photon", "per_incident_photon")


def load_response(path) -> dict:
    """Read and check a response matrix file (format in the module docstring)."""
    with np.load(Path(path), allow_pickle=False) as f:
        missing = {"energy_in_keV", "channel_edges_keV", "response"} - set(f.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        e_in = np.asarray(f["energy_in_keV"], dtype=float)
        edges = np.asarray(f["channel_edges_keV"], dtype=float)
        resp = np.asarray(f["response"], dtype=float)

    if resp.shape != (e_in.size, edges.size - 1):
        raise ValueError(f"{path}: response has shape {resp.shape}, expected "
                         f"({e_in.size}, {edges.size - 1}) from the energy arrays")
    if np.any(np.diff(e_in) <= 0):
        raise ValueError(f"{path}: energy_in_keV must be strictly increasing")
    widths = np.diff(edges)
    if np.any(widths <= 0) or not np.allclose(widths, widths[0], rtol=1e-6):
        raise ValueError(f"{path}: channels must all have the same width")
    if not np.all(np.isfinite(resp)) or np.any(resp < 0):
        raise ValueError(f"{path}: response must be finite and non-negative")

    return {
        "energy_in_keV": e_in,
        "channel_centres_keV": 0.5 * (edges[:-1] + edges[1:]),
        "channel_width_keV": float(widths[0]),
        "response": resp,
    }


def _shift(centres: np.ndarray, row: np.ndarray, delta_kev: float) -> np.ndarray:
    """Move a whole row by delta_kev along the deposited-energy axis."""
    return np.interp(centres - delta_kev, centres, row, left=0.0, right=0.0)


def fold(response: dict, energy_kev, counts) -> np.ndarray:
    """
    Deposited-energy spectrum [counts/s per channel] for photons of the given
    energies and rates.

    A photon between two rows uses both rows, each shifted to the photon's own
    energy and weighted by distance, so peaks land at the exact energy without
    being widened.  Outside the row range the nearest row is shifted.
    """
    e_in = response["energy_in_keV"]
    rows = response["response"]
    centres = response["channel_centres_keV"]
    out = np.zeros(centres.size)
    for e, n in zip(np.atleast_1d(energy_kev), np.atleast_1d(counts)):
        if n == 0:
            continue
        i = int(np.searchsorted(e_in, e))
        if i == 0 or i == e_in.size:
            k = 0 if i == 0 else e_in.size - 1
            out += n * _shift(centres, rows[k], e - e_in[k])
            continue
        lo, hi = i - 1, i
        t = (e - e_in[lo]) / (e_in[hi] - e_in[lo])
        out += n * ((1.0 - t) * _shift(centres, rows[lo], e - e_in[lo])
                    + t * _shift(centres, rows[hi], e - e_in[hi]))
    return out


def broaden(centres: np.ndarray, counts: np.ndarray,
            fano: float, noise_kev: float, w_kev: float) -> np.ndarray:
    """
    Gaussian electronic broadening of a deposited-energy spectrum.

    Each channel at deposited energy E is spread with
        σ² = F · w · E + σ_el²
    and its counts are kept (weights normalised per channel).
    """
    if fano == 0.0 and noise_kev == 0.0:
        return counts.copy()
    width = float(centres[1] - centres[0])
    sig = np.sqrt(fano * w_kev * np.clip(centres, 0.0, None) + noise_kev ** 2)
    sig = np.maximum(sig, 1e-9)
    K = int(np.ceil(6.0 * sig.max() / width))
    k = np.arange(-K, K + 1)
    g = np.exp(-0.5 * (k[:, None] * width / sig[None, :]) ** 2)
    g /= g.sum(axis=0, keepdims=True)
    out = np.zeros_like(counts)
    n = counts.size
    for kk, weights in zip(k, g):
        src = counts * weights
        if kk >= 0:
            out[kk:] += src[:n - kk]
        else:
            out[:kk] += src[-kk:]
    return out


def build_spectrum_with_response(
    energy_grid: np.ndarray,
    I_coherent: np.ndarray,
    df_fluo,
    response: dict,
    rows: str,
    be_mm: float,
    si_active_um: float,
    fano: float,
    noise_ev: float,
    w_ev: float,
    I_incoherent: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulated detector spectrum through a response matrix.

    Inputs are what reaches the detector face: the scattered continua
    [ph/s/keV] on energy_grid and the lines [ph/s] in df_fluo.

    Returns (channel_centres_keV, spectrum [ph/s/keV]) on the matrix channels.
    """
    if rows not in ROW_OPTIONS:
        raise ValueError(f"Unknown detector_response rows: {rows!r} "
                         f"(options: {', '.join(ROW_OPTIONS)})")

    continuum = I_coherent if I_incoherent is None else I_coherent + I_incoherent
    lines = df_fluo[np.isfinite(df_fluo["E_line_keV"]) & (df_fluo["Intensity_ph_s"] > 0)]
    energy = np.concatenate([energy_grid, lines["E_line_keV"].to_numpy(float)])
    rate = np.concatenate([continuum * np.gradient(energy_grid),       # ph/s per bin
                           lines["Intensity_ph_s"].to_numpy(float)])   # ph/s per line

    if rows == "per_detected_photon":
        # rows are per photon that left energy in the crystal, so the
        # efficiency counts photoelectric and Compton interactions
        rate = (rate * transmission_be(energy, be_mm)
                * qe_active_si(energy, si_active_um, include_compton=True))

    deposited = fold(response, energy, rate)
    centres = response["channel_centres_keV"]
    counts = broaden(centres, deposited, fano, noise_ev / 1000.0, w_ev / 1000.0)
    return centres, counts / response["channel_width_keV"]
