"""
Detector-response simulation for the XRF pipeline.

Assembles a simulated spectrum in ph/s/keV on an arbitrary energy grid by:
  - Applying Be entrance-window transmission
  - Convolving the coherent-scatter continuum with the SDD Gaussian response
  - Adding Gaussian-broadened fluorescence lines
  - Multiplying by the active-Si quantum efficiency

This is the simple response, used when the config has no detector_response
matrix: peaks are pure Gaussians, with no escape peaks, tailing or shelf.
With a matrix, detector_response.py is used instead.  Neither models pile-up
or dead time.

All distance inputs (be_mm, si_active_um) are converted to cm internally.
df_fluo must be the DataFrame written to the Fluorescence sheet by the
pipeline, with at minimum the columns E_line_keV and Intensity_ph_s.
"""
from __future__ import annotations

import numpy as np
import xraylib


def transmission_be(energy_kev: np.ndarray, be_mm: float) -> np.ndarray:
    """
    Transmission of the Be entrance window.

    Parameters
    ----------
    energy_kev : 1D array of photon energies [keV]
    be_mm      : Be window thickness [mm]

    Returns
    -------
    T_Be : array, same shape as energy_kev, values in (0, 1]
    """
    be_cm  = be_mm / 10.0
    rho_be = 1.848  # g/cm³
    mu_be  = np.array([xraylib.CS_Total(4, E) for E in energy_kev])
    return np.exp(-mu_be * rho_be * be_cm)


def qe_active_si(energy_kev: np.ndarray, si_active_um: float,
                 include_compton: bool = False) -> np.ndarray:
    """
    Quantum efficiency of the active Si detector bulk.

    Fraction of photons that interact in a Si layer of thickness
    si_active_um and leave energy there.  Follows Beer-Lambert:
        QE = 1 - exp(-μ_Si * ρ_Si * d)

    Which interactions count depends on how the response places photons:
      include_compton=False  μ = photoelectric only.  For Gaussian peaks,
                             which put every detected photon at its full
                             energy: only photoabsorbed photons do that.
      include_compton=True   μ = photoelectric + Compton.  For a response
                             matrix normalised per photon that left energy in
                             the crystal: the matrix itself puts Compton
                             events in the low-energy shelf.
    Rayleigh scattering is never counted: it leaves no energy.

    Low-energy photons are absorbed near the surface (QE→1); high-energy
    photons may pass through without interacting (QE<1 above ~15 keV for a
    typical 450 µm crystal).

    Parameters
    ----------
    energy_kev      : 1D array of photon energies [keV]
    si_active_um    : active Si thickness [µm]
    include_compton : also count Compton interactions (see above)

    Returns
    -------
    QE : array, same shape as energy_kev, values in (0, 1)
    """
    si_cm  = si_active_um / 1.0e4
    rho_si = 2.329  # g/cm³
    mu_si  = np.array([xraylib.CS_Photo(14, E)
                       + (xraylib.CS_Compt(14, E) if include_compton else 0.0)
                       for E in energy_kev])
    return 1.0 - np.exp(-mu_si * rho_si * si_cm)


def sigma_kev(energy_kev, fano: float, noise_ev: float, w_ev: float):
    """
    Gaussian sigma of the SDD response in keV.

    Uses the two-term formula:
        sigma² = fano · w · E + sigma_el²

    Parameters
    ----------
    energy_kev : scalar or array of photon energies [keV]
    fano       : effective Fano factor, measured together with w
    noise_ev   : electronic noise sigma_el [eV rms]
    w_ev       : energy per electron-hole pair used with fano [eV]

    Returns
    -------
    sigma in keV, same type/shape as energy_kev
    """
    w_kev, noise_kev = w_ev / 1000.0, noise_ev / 1000.0
    return np.sqrt(fano * w_kev * energy_kev + noise_kev ** 2)


def _smear_continuum(
    energy_grid: np.ndarray,
    spectrum: np.ndarray,
    fano: float,
    noise_ev: float,
    w_ev: float,
) -> np.ndarray:
    """
    Convolve a continuum spectrum with the energy-dependent SDD Gaussian.

    Each input bin at energy E' contributes to output energy E as:
        I_smeared(E) = ∫ I(E') G(E - E'; sigma(E')) dE'

    where sigma(E') is the Fano+ENC resolution at the *source* energy.
    Implemented as a dense (N × N) matrix multiply; fine for N ≲ 2000.

    Parameters
    ----------
    energy_grid   : sorted 1D array [keV]
    spectrum      : 1D array [ph/s/keV] on energy_grid
    fano, noise_ev, w_ev : detector resolution parameters (see sigma_kev)

    Returns
    -------
    smeared : 1D array [ph/s/keV]
    """
    dE  = np.gradient(energy_grid)
    sig = sigma_kev(energy_grid, fano, noise_ev, w_ev)         # (N,) at source energies

    # diff[i, j] = energy_grid[i] - energy_grid[j]
    diff  = energy_grid[:, np.newaxis] - energy_grid[np.newaxis, :]   # (N_out, N_in)
    sig_j = sig[np.newaxis, :]                                         # (1,    N_in)

    G = np.exp(-0.5 * (diff / sig_j) ** 2) / (sig_j * np.sqrt(2.0 * np.pi))
    return G @ (spectrum * dE)


def build_spectrum(
    energy_grid: np.ndarray,
    I_coherent: np.ndarray,
    df_fluo,
    be_mm: float,
    si_active_um: float,
    fano: float,
    noise_ev: float,
    w_ev: float,
    I_incoherent: np.ndarray | None = None,
) -> np.ndarray:
    """
    Assemble the full simulated detector spectrum in ph/s/keV.

    Steps
    -----
    1. Be-window transmission T applied to both continuum and lines.
    2. Active-Si QE applied (fraction of photons actually absorbed/detected).
    3. Coherent + incoherent scatter continuum convolved with the Gaussian
       detector response before adding to the output spectrum.
    4. Each fluorescence line (ph/s) is spread into a Gaussian peak [ph/s/keV]
       using the energy-dependent Fano+ENC sigma; T and QE are applied at
       the line energy.

    Parameters
    ----------
    energy_grid   : 1D array of energies [keV] — must be sorted ascending
    I_coherent    : 1D array, coherent scatter intensity [ph/s/keV]
    df_fluo       : DataFrame with columns E_line_keV and Intensity_ph_s
    be_mm         : Be window thickness [mm]
    si_active_um  : active Si crystal thickness [µm]
    fano          : effective Fano factor, measured together with w_ev
    noise_ev      : electronic noise sigma_el [eV rms]
    w_ev          : energy per electron-hole pair used with fano [eV]
    I_incoherent  : 1D array, incoherent (Compton) scatter intensity [ph/s/keV],
                    already redistributed to scattered-photon energies.
                    If None, Compton scatter is not included.

    Returns
    -------
    spectrum : 1D array [ph/s/keV] on energy_grid
    """
    T  = transmission_be(energy_grid, be_mm)
    QE = qe_active_si(energy_grid, si_active_um)

    continuum = I_coherent
    if I_incoherent is not None:
        continuum = continuum + I_incoherent

    # Continuum: apply Be transmission and Si efficiency, then smear with the
    # detector resolution
    continuum_pre = continuum * T * QE
    spectrum = _smear_continuum(energy_grid, continuum_pre, fano, noise_ev, w_ev)

    for _, row in df_fluo.iterrows():
        E0 = row["E_line_keV"]
        I0 = row["Intensity_ph_s"]
        if not np.isfinite(E0) or I0 <= 0.0:
            continue

        sig = float(sigma_kev(E0, fano, noise_ev, w_ev))
        # Gaussian PDF [1/keV]: integrates to 1 over energy → I0 * gauss has units ph/s/keV ✓
        gauss       = np.exp(-0.5 * ((energy_grid - E0) / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))
        T_at_line   = float(np.interp(E0, energy_grid, T))
        QE_at_line  = float(np.interp(E0, energy_grid, QE))
        spectrum   += I0 * T_at_line * QE_at_line * gauss

    return spectrum
