# src/xrf_model/physics.py
from __future__ import annotations

import numpy as np
import xraylib


# ── L-shell line table ────────────────────────────────────────────────────────
# (display name, xraylib attribute name, originating subshell: 0=L1 1=L2 2=L3)
def _build_l_lines():
    _candidates = [
        ("Lα1",  "LA1_LINE",  2),   # L3 → M5
        ("Lα2",  "LA2_LINE",  2),   # L3 → M4
        ("Lβ1",  "LB1_LINE",  1),   # L2 → M4
        ("Lβ2",  "LB2_LINE",  2),   # L3 → N4,5
        ("Lβ3",  "LB3_LINE",  0),   # L1 → M2,3
        ("Lβ4",  "LB4_LINE",  0),   # L1 → M2
        ("Lβ15", "LB15_LINE", 2),   # L3 → N4
        ("Lβ17", "LB17_LINE", 1),   # L2 → M3
        ("Lγ1",  "LG1_LINE",  1),   # L2 → N4
        ("Lγ2",  "LG2_LINE",  0),   # L1 → N2,3
        ("Lγ3",  "LG3_LINE",  0),   # L1 → N1
        ("Ll",   "LL_LINE",   2),   # L3 → M1
    ]
    return [
        (name, getattr(xraylib, attr), idx)
        for name, attr, idx in _candidates
        if getattr(xraylib, attr, None) is not None
    ]

_L_LINES = _build_l_lines()


# ── Element properties ────────────────────────────────────────────────────────

class ElementProperties:
    """Stores per-element atomic numbers/masses (from xraylib) and the run config."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.param_dict = {}
        for el in cfg["all_matrix_elements"]:
            Z = xraylib.SymbolToAtomicNumber(el)
            self.param_dict[el] = {
                "Atomic Number": Z,
                "Atomic Mass":   xraylib.AtomicWeight(Z),
            }


# ── Fluorescence integrals ────────────────────────────────────────────────────

def compute_k_fluorescence(
    energy_array,
    flux_array,
    Z,
    cos_theta,
    cos_phi,
    total_atten_sample_array,
    total_atten_sample_at_line,
    absorption_edge,
):
    """
    Raw excitation integral for K-shell fluorescence at a single grid cell.

    Returns
        ∫_{E_edge}  φ(E) · μ_ph(Z,E) / [μ(E) + (cosθ/cosφ) · μ_line]  dE

    flux_array must already contain the cosθ factor (φ = P·cosθ/R²), so that
    the denominator recovers μ(E)/sinΦ + μ_line/sinΨ after cancellation.

    cos_theta, cos_phi : per-cell sinΦ and sinΨ from GridGeometry — required,
                         no cfg fallback.
    """
    mask = energy_array >= absorption_edge
    if not mask.any():
        return 0.0
    E_sub     = energy_array[mask]
    flux_sub  = flux_array[mask]
    mu_ph     = np.array([xraylib.CS_Photo(Z, E) for E in E_sub])
    mu_tot_E  = total_atten_sample_array[mask]
    denom     = mu_tot_E + (cos_theta / cos_phi) * total_atten_sample_at_line
    integrand = np.where(denom > 0.0, flux_sub * mu_ph / denom, 0.0)
    return np.trapezoid(integrand, x=E_sub)


def compute_l_fluorescence(
    energy_array,
    flux_array,
    Z,
    cos_theta,
    cos_phi,
    total_atten_sample_array,
    f_atten_interp,
):
    """
    L-shell fluorescence with Coster-Kronig vacancy-cascade corrections.

    cos_theta, cos_phi : per-cell values — required, no cfg fallback.

    Returns a list of dicts — one per L line with E_line > 0 and rad_rate > 0:
        line_name, E_line, rad_rate, I_raw
    I_raw = ω_Li · N_Li_eff · RadRate / (4π).  Caller multiplies by concentration w.

    Cascade equations
    -----------------
    N_L1 = ∫ φ · σ_L1(E) / denom dE
    N_L2 = ∫ φ · σ_L2(E) / denom dE  +  f12 · N_L1
    N_L3 = ∫ φ · σ_L3(E) / denom dE  +  f23 · N_L2  +  f13 · N_L1

    xraylib calls: CS_Photo_Partial, FluorYield, CosKronTransProb, RadRate, LineEnergy.
    """
    geom = cos_theta / cos_phi

    def _safe(fn, *args):
        try:
            v = fn(*args)
            return float(v) if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    f12 = _safe(xraylib.CosKronTransProb, Z, xraylib.FL12_TRANS)
    f13 = _safe(xraylib.CosKronTransProb, Z, xraylib.FL13_TRANS)
    f23 = _safe(xraylib.CosKronTransProb, Z, xraylib.FL23_TRANS)

    omega_L = [
        _safe(xraylib.FluorYield, Z, xraylib.L1_SHELL),
        _safe(xraylib.FluorYield, Z, xraylib.L2_SHELL),
        _safe(xraylib.FluorYield, Z, xraylib.L3_SHELL),
    ]

    # Partial photoionisation arrays — depend on Z and E only, not on cell geometry.
    sigma_L = [
        np.array([_safe(xraylib.CS_Photo_Partial, Z, sh, E) for E in energy_array])
        for sh in [xraylib.L1_SHELL, xraylib.L2_SHELL, xraylib.L3_SHELL]
    ]

    rows = []
    for line_name, line_const, shell_idx in _L_LINES:
        E_line   = _safe(xraylib.LineEnergy, Z, line_const)
        rad_rate = _safe(xraylib.RadRate,    Z, line_const)
        if E_line <= 0.0 or rad_rate <= 0.0:
            continue

        denom = total_atten_sample_array + geom * float(f_atten_interp(E_line))
        raw_L = [
            np.trapezoid(
                np.where(denom > 0.0, flux_array * sigma / denom, 0.0),
                x=energy_array,
            )
            for sigma in sigma_L
        ]

        N_L1 = raw_L[0]
        N_L2 = raw_L[1] + f12 * N_L1
        N_L3 = raw_L[2] + f23 * N_L2 + f13 * N_L1
        N_eff = [N_L1, N_L2, N_L3]

        I_raw = omega_L[shell_idx] * N_eff[shell_idx] * rad_rate / (4.0 * np.pi)
        rows.append({
            "line_name": line_name,
            "E_line":    E_line,
            "rad_rate":  rad_rate,
            "I_raw":     I_raw,
        })

    return rows


# ── Coherent (Rayleigh) scattering ────────────────────────────────────────────

def precompute_anomalous_factors(energy_array, param_dict):
    """Precompute f1, f2 anomalous scattering factors for all elements.

    These depend only on energy, not on scattering angle or cell position, so
    they are computed once per run and reused across all grid cells.

    Returns a dict: {el_name: {"f1": array, "f2": array}}
    """
    cache = {}
    for el_name, params in param_dict.items():
        Z = params["Atomic Number"]
        cache[el_name] = {
            "f1": np.array([xraylib.Fi(Z, E)  for E in energy_array]),
            "f2": np.array([xraylib.Fii(Z, E) for E in energy_array]),
        }
    return cache


def compute_coherent_cross_section(energy_array, scattering_angle, param_dict, conc,
                                   anomalous_cache=None):
    """
    Coherent (Rayleigh) scattering cross section of the sample mixture [cm²/g].

    anomalous_cache: output of precompute_anomalous_factors — pass it from the
                     grid loop to avoid recomputing f1/f2 per cell.
    """
    r_e = 2.818e-13   # cm
    N_A = 6.022e23
    ang_factor = 1.0 + np.cos(scattering_angle) ** 2
    # FF_Rayl expects sin(θ/2)·E[keV] / 12.398  [Å⁻¹]
    x_angstrom = np.sin(scattering_angle / 2.0) * energy_array / 12.398

    sigma_coh = np.zeros_like(energy_array, dtype=float)
    for el_name, params in param_dict.items():
        Z_val = params["Atomic Number"]
        f0 = np.array([xraylib.FF_Rayl(Z_val, q) for q in x_angstrom])
        if anomalous_cache is not None:
            f1 = anomalous_cache[el_name]["f1"]
            f2 = anomalous_cache[el_name]["f2"]
        else:
            f1 = np.array([xraylib.Fi(Z_val, E)  for E in energy_array])
            f2 = np.array([xraylib.Fii(Z_val, E) for E in energy_array])
        f_eff = np.sqrt((f0 + f1 - Z_val) ** 2 + f2 ** 2)
        dsigma = (r_e ** 2 / 2.0) * ang_factor * f_eff ** 2
        sigma_coh += conc[el_name] * (N_A / params["Atomic Mass"]) * dsigma

    return sigma_coh
