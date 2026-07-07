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

    def __init__(self, cfg: dict, elements: list):
        self.cfg = cfg
        self.param_dict = {}
        for el in elements:
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
    sinPhi,
    sinPsi,
    total_atten_sample_array,
    total_atten_sample_at_line,
    absorption_edge,
):
    """
    Raw excitation integral for K-shell fluorescence at a single grid cell.

    Returns
        ∫_{E_edge}  φ(E) · μ_ph(Z,E) / [μ(E) + (sinΦ/sinΨ) · μ_line]  dE

    flux_array must already contain the sinΦ factor (φ = P·sinΦ/R²), so that
    the denominator recovers μ(E)/sinΦ + μ_line/sinΨ after cancellation.

    sinPhi, sinPsi : per-cell sinΦ and sinΨ from GridGeometry — required,
                     no cfg fallback.
    """
    mask = energy_array >= absorption_edge
    if not mask.any():
        return 0.0
    E_sub     = energy_array[mask]
    flux_sub  = flux_array[mask]
    mu_ph     = np.array([xraylib.CS_Photo(Z, E) for E in E_sub])
    mu_tot_E  = total_atten_sample_array[mask]
    denom     = mu_tot_E + (sinPhi / sinPsi) * total_atten_sample_at_line
    integrand = np.where(denom > 0.0, flux_sub * mu_ph / denom, 0.0)
    return np.trapezoid(integrand, x=E_sub)


def compute_l_fluorescence(
    energy_array,
    flux_array,
    Z,
    sinPhi,
    sinPsi,
    total_atten_sample_array,
    f_atten_interp,
):
    """
    L-shell fluorescence with Coster-Kronig vacancy-cascade corrections.

    sinPhi, sinPsi : per-cell sinΦ and sinΨ from GridGeometry — required,
                     no cfg fallback.

    Returns a list of dicts — one per L line with E_line > 0 and rad_rate > 0:
        line_name, E_line, rad_rate, I_raw
    I_raw = ω_Li · N_Li_eff · RadRate.  Caller multiplies by w and (Ω/4π)·ΔS.

    Cascade equations
    -----------------
    N_L1 = ∫ φ · σ_L1(E) / denom dE
    N_L2 = ∫ φ · σ_L2(E) / denom dE  +  f12 · N_L1
    N_L3 = ∫ φ · σ_L3(E) / denom dE  +  f23 · N_L2  +  f13 · N_L1

    xraylib calls: CS_Photo_Partial, FluorYield, CosKronTransProb, RadRate, LineEnergy.
    """
    if Z < 29:
        return []

    geom = sinPhi / sinPsi

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

    edge_L = [
        _safe(xraylib.EdgeEnergy, Z, xraylib.L1_SHELL),
        _safe(xraylib.EdgeEnergy, Z, xraylib.L2_SHELL),
        _safe(xraylib.EdgeEnergy, Z, xraylib.L3_SHELL),
    ]
    sigma_L = [
        np.where(energy_array >= edge_L[i], sigma_L[i], 0.0)
        for i in range(3)
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

        I_raw = omega_L[shell_idx] * N_eff[shell_idx] * rad_rate
        rows.append({
            "line_name": line_name,
            "E_line":    E_line,
            "rad_rate":  rad_rate,
            "I_raw":     I_raw,
        })

    return rows


# ── Secondary (inter-element) fluorescence ────────────────────────────────────

def compute_k_secondary_fluorescence(
    energy_array,
    flux_array,
    sinPhi,
    sinPsi,
    total_atten_sample_array,
    total_atten_sample_at_line,
    absorption_edge,
    exciters,
):
    """
    Raw excitation integral for SECONDARY K-shell fluorescence at one grid cell.

    The analyte A (K-edge = ``absorption_edge``, emission line attenuated by
    ``total_atten_sample_at_line`` = μ(E_A)) is excited a second time by the
    characteristic lines of the other elements B in the sample.  The result is
    the ``raw_sec`` quantity that plugs into the SAME accumulator, with the SAME
    multiplier (cell_factor · w_A · ω_K,A · R_jump,A · p_A,line), as the primary
    K term produced by ``compute_k_fluorescence``.

    Physics (Sherman / Shiraiwa-Fujino, semi-infinite homogeneous sample,
    polychromatic beam)
    -------------------------------------------------------------------------
        raw_sec = ½ · Σ_B Σ_b  C_Bb · ∫_{E ≥ E_edge,B}
                       φ(E) · τ_B(E) · L_b(E) / [ μ(E) + (sinΦ/sinΨ)·μ(E_A) ]  dE

        C_Bb  = w_B · ω_K,B · R_jump,B · p_b · τ_A(E_Bb)      (B line-b yield ×
                                                               A photo-absorption
                                                               at the B line)
        τ_X(E) = CS_Photo(Z_X, E)                             [cm²/g]
        L_b(E) = (sinΦ/μ(E))·ln(1 + (μ(E)/sinΦ)/μ(E_Bb))      ← primary path in
               + (sinΨ/μ(E_A))·ln(1 + (μ(E_A)/sinΨ)/μ(E_Bb))  ← A-line path out

    The ½ is the isotropic-emission solid-angle factor of the intermediate B
    atom.  The two logarithms are the closed form of the double depth integral
    ∫∫ e^{-a t'} e^{-b t} · ½·E₁(μ_B|t−t'|) dt dt', with a = μ(E)/sinΦ,
    b = μ(E_A)/sinΨ.  Note the denominator is identical to the primary
    ``compute_k_fluorescence`` denominator.

    A B-line contributes only when its energy exceeds the analyte K edge
    (``E_Bb > absorption_edge``); self-pairs (B == A) must already be excluded
    when building ``exciters``.

    Parameters
    ----------
    flux_array : φ(E) = P·sinΦ/R²  (already carries the sinΦ factor, as in
                 ``compute_k_fluorescence``).
    total_atten_sample_array       : μ(E) sample mass attenuation array [cm²/g].
    total_atten_sample_at_line     : μ(E_A) at the analyte emission line [cm²/g].
    absorption_edge                : analyte K-edge energy [keV].
    exciters : list of per-B-line dicts, each with keys
        'E_line_B'         : B line energy E_Bb [keV]
        'mu_B'             : μ(E_Bb) sample attenuation at the B line [cm²/g]
        'E_edge_B'         : B K-edge energy [keV] (primary mask)
        'yield_B'          : w_B · ω_K,B · R_jump,B · p_b  (dimensionless)
        'cs_photo_B'       : CS_Photo(Z_B, E) over energy_array [cm²/g]
        'cs_photo_A_at_B'  : CS_Photo(Z_A, E_Bb) = τ_A(E_Bb) [cm²/g]
    """
    if not exciters:
        return 0.0

    geom  = sinPhi / sinPsi
    mu_E  = total_atten_sample_array
    denom = mu_E + geom * total_atten_sample_at_line
    inv_denom = np.where(denom > 0.0, 1.0 / denom, 0.0)

    # A-line escape (path-out) log term depends only on E_A and E_B → scalar/B.
    b_out = total_atten_sample_at_line / sinPsi          # μ(E_A)/sinΨ

    total = 0.0
    for ex in exciters:
        E_Bb = ex["E_line_B"]
        if E_Bb <= absorption_edge:          # B line can't reach analyte K edge
            continue
        mu_B = ex["mu_B"]
        if mu_B <= 0.0:
            continue
        mask = energy_array >= ex["E_edge_B"]
        if not mask.any():
            continue

        # L_b(E): primary path-in (array) + A-line path-out (scalar)
        term_in  = (sinPhi / mu_E) * np.log1p((mu_E / sinPhi) / mu_B)
        term_out = (sinPsi / total_atten_sample_at_line) * np.log1p(b_out / mu_B)
        L        = term_in + term_out

        integrand = np.where(
            mask, flux_array * ex["cs_photo_B"] * L * inv_denom, 0.0
        )
        I = np.trapezoid(integrand, x=energy_array)
        total += ex["yield_B"] * ex["cs_photo_A_at_B"] * I

    return 0.5 * total


# ── Incoherent (Compton) scattering ──────────────────────────────────────────

def compute_incoherent_cross_section(energy_array, scattering_angle, param_dict, conc):
    """
    Incoherent (Compton) differential cross section of the sample mixture [cm²/g/sr].

    Uses xraylib.DCS_Compt(Z, E, θ) which implements the Klein-Nishina formula
    weighted by the incoherent scattering function S(x, Z) from Hubbell et al.
    (1975), J. Phys. Chem. Ref. Data 4, 471.  S accounts for the bound-electron
    binding corrections to the free-electron Klein-Nishina result.

    scattering_angle : scalar [rad], per-cell value from GridGeometry
    """
    sigma_incoh = np.zeros_like(energy_array, dtype=float)
    for el_name, params in param_dict.items():
        w = conc.get(el_name, 0.0)
        if w <= 0:
            continue
        Z_val = params["Atomic Number"]
        sigma_incoh += w * np.array(
            [xraylib.DCS_Compt(Z_val, float(E), float(scattering_angle))
             for E in energy_array]
        )
    return sigma_incoh


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
        w = conc.get(el_name, 0.0)
        if w <= 0:
            continue
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
        sigma_coh += w * (N_A / params["Atomic Mass"]) * dsigma

    return sigma_coh
