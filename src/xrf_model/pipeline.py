# src/xrf_model/pipeline.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xraylib
from scipy.interpolate import interp1d

from .io import load_flux_data
from .physics import (
    ElementProperties,
    compute_k_fluorescence,
    compute_l_fluorescence,
    compute_coherent_cross_section,
    precompute_anomalous_factors,
    _L_LINES,
)
from .geometry import GridGeometry


def _k_group(Z: int, line_attr_names: list) -> tuple:
    """
    Energy-weighted mean line energy and total branching fraction for a K-line group.

    line_attr_names: list of xraylib attribute name strings, e.g. ["KA1_LINE", "KA2_LINE"].
    Lines absent from the installed xraylib version are silently skipped.
    Returns (mean_energy_keV, total_branching_fraction).
    """
    total, e_sum = 0.0, 0.0
    for name in line_attr_names:
        lc = getattr(xraylib, name, None)
        if lc is None:
            continue
        try:
            r = xraylib.RadRate(Z, lc)
            e = xraylib.LineEnergy(Z, lc)
            if r > 0 and e > 0:
                total += r
                e_sum += r * e
        except Exception:
            pass
    return (e_sum / total if total > 0 else 0.0), total


def run_one_grid(flux_file, cfg: dict, debugging: bool = False) -> dict:
    """
    Spatially-resolved K + L fluorescence using a 2D surface grid.

    Implements:
        I_det(ip) = Σ_{m,n}  dΩ/(4π) · ΔS
                    · ∫ Q_ip(E) φ(E,r_mn) / [μ(E)/sinΦ + μ(ip)/sinΨ] dE

    where φ(E, r_mn) = P(E) · sinΦ_mn / R_src_mn².

    Flux conversion
    ---------------
    The flux file delivers P_raw(E) in ph/cm²/s/keV at reference distance
    src_reference_distance_cm [cm] from the source.  The code converts:

        P_eff(E) = P_raw(E) × d_ref²

    so that phi_mn = P_eff × cos_th / R_mm² is in ph/mm²/s/keV and the
    accumulated output intensity is in ph/s.

    All atomic parameters are fetched from xraylib (no spreadsheet look-ups).
    All distances are in mm.  Cross-sections from xraylib are in cm²/g.
    """
    flux_file = Path(flux_file)
    run_name  = flux_file.stem

    # ── Flux spectrum ─────────────────────────────────────────────────────────
    energy_kev, flux_raw = load_flux_data(str(flux_file))
    # Convert ph/cm²/s/keV at d_ref [cm] to effective spectral radiance:
    #   P_eff × cos_th / R_mm²  →  ph/mm²/s/keV  (irradiance at grid point)
    d_ref = float(cfg["src_reference_distance_cm"])
    P_eff = flux_raw * (d_ref ** 2)

    # ── Concentrations ────────────────────────────────────────────────────────
    df_params = pd.read_excel(cfg["paths"]["element_data"], index_col=0)
    df_conc   = df_params.loc[:, 'mantle_mars':]
    conc_cols = cfg["concentration_cols"]

    concentrations = {c: {} for c in conc_cols}
    for el in cfg["all_matrix_elements"]:
        for c in conc_cols:
            concentrations[c][el] = df_conc.at[el, c]

    element_properties = ElementProperties(cfg)

    # ── K-shell atomic parameters (xraylib, precomputed once per element) ─────
    # Elements that lack valid K-shell data in xraylib are silently skipped.
    k_params: dict = {}
    for el in cfg["fluorescence_elements"]:
        Z = element_properties.param_dict[el]["Atomic Number"]
        try:
            E_edge = xraylib.EdgeEnergy(Z, xraylib.K_SHELL)
            wK     = xraylib.FluorYield(Z, xraylib.K_SHELL)
            jump_r = xraylib.JumpFactor(Z, xraylib.K_SHELL)
        except Exception:
            continue
        if E_edge <= 0 or wK <= 0 or jump_r <= 1.0:
            continue
        R_jump = 1.0 - 1.0 / jump_r   # (r-1)/r  absorption jump factor

        E_Ka, P_Ka = _k_group(Z, ["KA1_LINE", "KA2_LINE"])
        E_Kb, P_Kb = _k_group(Z, ["KB1_LINE", "KB2_LINE", "KB3_LINE"])
        k_params[el] = dict(Z=Z, E_edge=E_edge, wK=wK, R_jump=R_jump,
                            E_Ka=E_Ka, P_Ka=P_Ka, E_Kb=E_Kb, P_Kb=P_Kb)

    # ── Grid geometry ─────────────────────────────────────────────────────────
    geo = GridGeometry(cfg)
    cos_theta_arr, cos_phi_arr, R_src_arr, R_det_arr, Omega_arr, valid_arr = \
        geo.compute_local_geometry()

    n_valid = int(valid_arr.sum())
    print(f"Grid: {cfg['grid_n_x']} × {cfg['grid_n_y']}  "
          f"({cfg['grid_n_x'] * cfg['grid_n_y']} points, {n_valid} valid)")

    out_dir = Path(cfg["paths"]["output_dir"]) / (run_name + "_grid")
    out_dir.mkdir(parents=True, exist_ok=True)

    # f1/f2 depend only on energy — precompute once, reuse across all cells
    anomalous_cache = precompute_anomalous_factors(energy_kev,
                                                   element_properties.param_dict)
    results_dict = {}

    for conc in conc_cols:
        print('=' * 70)
        print(f'SAMPLE: {conc}')
        print('=' * 70)

        sample_conc = concentrations[conc]

        # Sample total mass attenuation μ(E)/ρ  [cm²/g]
        # Σ_j  w_j · CS_Total(Z_j, E)
        sample_mu = np.zeros_like(energy_kev, dtype=float)
        for el_name, w in sample_conc.items():
            if w > 0:
                Z_el = element_properties.param_dict[el_name]["Atomic Number"]
                sample_mu += w * np.array([xraylib.CS_Total(Z_el, E)
                                           for E in energy_kev])

        f_mu_interp = interp1d(energy_kev, sample_mu,
                               bounds_error=False, fill_value="extrapolate")

        # Accumulators keyed by (element_symbol, line_label_string)
        line_totals: dict = {}
        I_coh_total = np.zeros_like(energy_kev)

        # ── Grid loop ─────────────────────────────────────────────────────────
        ny, nx = valid_arr.shape
        for iy in range(ny):
            for ix in range(nx):
                if not valid_arr[iy, ix]:
                    continue

                cos_th = float(cos_theta_arr[iy, ix])   # sinΦ_mn
                cos_ph = float(cos_phi_arr[iy, ix])      # sinΨ_mn
                R_src  = float(R_src_arr[iy, ix])        # mm
                R_det  = float(R_det_arr[iy, ix])        # mm
                Omega  = float(Omega_arr[iy, ix])        # sr

                # φ(E, r_mn) = P_eff · sinΦ / R_src²  [ph/mm²/s/keV]
                phi_mn = P_eff * cos_th / (R_src ** 2)

                # dΩ/(4π) · ΔS  — dimensionless fraction × mm²
                cell_factor = (Omega / (4.0 * np.pi)) * geo.dS

                # ── Coherent (Rayleigh) scattering ────────────────────────────
                p_mn  = np.array([geo.XX[iy, ix], geo.YY[iy, ix], 0.0])
                k_in  = (p_mn - geo.r_src) / R_src
                k_out = (geo.r_det - p_mn) / R_det
                psi   = np.arccos(np.clip(np.dot(k_in, k_out), -1.0, 1.0))
                sigma_coh = compute_coherent_cross_section(
                    energy_kev, psi, element_properties.param_dict,
                    sample_conc, anomalous_cache,
                )
                denom_coh = sample_mu * (1.0 + cos_th / cos_ph)
                I_coh_total += cell_factor * np.where(
                    denom_coh > 0, phi_mn * sigma_coh / denom_coh, 0.0)

                # ── K + L fluorescence ────────────────────────────────────────
                for el, kp in k_params.items():
                    w = sample_conc.get(el, 0.0)
                    if w <= 0:
                        continue

                    Z = kp["Z"]

                    # K-alpha (KA1 + KA2 group)
                    if kp["E_Ka"] > 0 and kp["P_Ka"] > 0:
                        mu_Ka = float(f_mu_interp(kp["E_Ka"]))
                        raw   = compute_k_fluorescence(
                            energy_kev, phi_mn, Z, cos_th, cos_ph,
                            sample_mu, mu_Ka, kp["E_edge"],
                        )
                        key = (el, "Ka")
                        line_totals[key] = (line_totals.get(key, 0.0)
                                            + cell_factor * raw * w
                                            * kp["wK"] * kp["R_jump"] * kp["P_Ka"])

                    # K-beta (KB1 + KB2 + KB3 group)
                    if kp["E_Kb"] > 0 and kp["P_Kb"] > 0:
                        mu_Kb = float(f_mu_interp(kp["E_Kb"]))
                        raw   = compute_k_fluorescence(
                            energy_kev, phi_mn, Z, cos_th, cos_ph,
                            sample_mu, mu_Kb, kp["E_edge"],
                        )
                        key = (el, "Kb")
                        line_totals[key] = (line_totals.get(key, 0.0)
                                            + cell_factor * raw * w
                                            * kp["wK"] * kp["R_jump"] * kp["P_Kb"])

                    # L-shell (all lines with non-zero rate for this Z)
                    for r in compute_l_fluorescence(
                        energy_kev, phi_mn, Z, cos_th, cos_ph,
                        sample_mu, f_mu_interp,
                    ):
                        key = (el, r["line_name"])
                        line_totals[key] = (line_totals.get(key, 0.0)
                                            + cell_factor * w * r["I_raw"])

        # ── Build output DataFrame ────────────────────────────────────────────
        rows = []
        for el, kp in k_params.items():
            w = sample_conc.get(el, 0.0)
            Z = kp["Z"]

            # K lines
            for line_key, E_line, P_line, display in [
                ("Ka", kp["E_Ka"], kp["P_Ka"], "Kα"),
                ("Kb", kp["E_Kb"], kp["P_Kb"], "Kβ"),
            ]:
                rows.append({
                    "Element":        el,
                    "Concentration":  w,
                    "Line":           display,
                    "E_line_keV":     E_line if E_line > 0 else np.nan,
                    "P_line":         P_line,
                    "Intensity_ph_s": line_totals.get((el, line_key), 0.0),
                })

            # L lines — only emit rows for lines that were actually computed
            for line_name, line_const, _ in _L_LINES:
                key = (el, line_name)
                if key not in line_totals:
                    continue
                try:
                    E_line   = xraylib.LineEnergy(Z, line_const)
                    rad_rate = xraylib.RadRate(Z, line_const)
                except Exception:
                    E_line, rad_rate = np.nan, np.nan
                rows.append({
                    "Element":        el,
                    "Concentration":  w,
                    "Line":           line_name,
                    "E_line_keV":     E_line,
                    "P_line":         rad_rate,
                    "Intensity_ph_s": line_totals[key],
                })

        df_fluo = pd.DataFrame(rows).sort_values(["Element", "Line"])

        xlsx_path = out_dir / f"{run_name}_{conc}_grid_results.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame({"Energy_keV": energy_kev,
                          "I_coherent": I_coh_total}).to_excel(
                writer, sheet_name="Coherent", index=False)
            df_fluo.to_excel(writer, sheet_name="Fluorescence", index=False)
        print(f"→ Saved to {xlsx_path}")

        results_dict[conc] = {
            "energy":        energy_kev,
            "I_coherent":    I_coh_total,
            "fluorescence":  df_fluo,
            "out_dir":       str(out_dir),
            "n_valid_cells": n_valid,
        }

    return results_dict
