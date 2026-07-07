# src/xrf_model/pipeline.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xraylib
from scipy.interpolate import interp1d

from .io import load_flux_data
from .output import save_mca
from .spectrum import build_spectrum
from .plot_spectrum import plot_and_save_spectrum
from .physics import (
    ElementProperties,
    compute_k_fluorescence,
    compute_k_secondary_fluorescence,
    compute_l_fluorescence,
    compute_coherent_cross_section,
    compute_incoherent_cross_section,
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

    so that phi_mn = P_eff × sinΦ / R_cm² is in ph/cm²/s/keV and the
    accumulated output intensity is in ph/s.

    All atomic parameters are fetched from xraylib (no spreadsheet look-ups).
    All distances are in cm.  Cross-sections from xraylib are in cm²/g.
    """
    flux_file = Path(flux_file)
    run_name  = flux_file.stem

    # ── Flux spectrum ─────────────────────────────────────────────────────────
    energy_kev, flux_raw = load_flux_data(str(flux_file))

    flux_units = cfg.get("flux_units", "ph_per_cm2_per_s_per_keV")
    if flux_units == "ph_per_cm2_per_s_per_keV":
        # flux_raw: ph/cm²/s/keV at d_ref [cm] from source (SpekPy z_cm)
        # P_eff: ph/s/keV  (source spectral luminosity, geometry-independent)
        d_ref = float(cfg["src_reference_distance_cm"])
        P_eff = flux_raw * (d_ref ** 2)
    elif flux_units == "ph_per_s_per_keV":
        # flux_raw is already a source luminosity — no reference distance needed
        P_eff = flux_raw                   # ph/s/keV
    else:
        raise ValueError(f"Unknown flux_units: {flux_units!r}")

    # ── Concentrations ────────────────────────────────────────────────────────
    df_params = pd.read_excel(cfg["paths"]["element_data"], index_col=0)
    df_conc   = df_params.loc[:, 'mantle_mars':]
    conc_cols = cfg["concentration_cols"]

    # Build per-sample dicts from non-zero, non-NaN rows only
    concentrations = {c: {} for c in conc_cols}
    for el in df_conc.index:
        for c in conc_cols:
            w = df_conc.at[el, c]
            if pd.notna(w) and float(w) > 0:
                concentrations[c][el] = float(w)

    # Elements active in at least one sample (preserves Excel row order)
    active_elements = [
        el for el in df_conc.index
        if any(el in concentrations[c] for c in conc_cols)
    ]
    print(f"Active elements: {active_elements}")

    element_properties = ElementProperties(cfg, active_elements)

    # ── K-shell atomic parameters (xraylib, precomputed once per element) ─────
    # Elements that lack valid K-shell data in xraylib are silently skipped.
    k_params: dict = {}
    for el in active_elements:
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

    # Secondary fluorescence toggle (default on)
    include_secondary = bool(cfg.get("include_secondary", True))

    # CS_Photo(Z, E) over the beam grid — depends only on Z & E (not sample),
    # so precompute once and reuse for the secondary-fluorescence integral.
    cs_photo_cache: dict = {}
    if include_secondary:
        for el, kp in k_params.items():
            Z = kp["Z"]
            cs_photo_cache[el] = np.array([xraylib.CS_Photo(Z, E)
                                           for E in energy_kev])

    # ── Grid geometry ─────────────────────────────────────────────────────────
    geo = GridGeometry(cfg)
    sinPhi_arr, sinPsi_arr, R_src_arr, R_det_arr, Omega_arr, valid_arr = \
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

        # ── Secondary-fluorescence exciter tables (per sample) ─────────────────
        # For each analyte A, the list of B-lines (B ≠ A, present in this sample)
        # that can re-excite it.  Depends only on composition, not on cell/angle,
        # so it is built once here and reused across all grid cells.
        exc_by_analyte: dict = {}
        if include_secondary:
            # Master list of every K-line emitter present in this sample.
            exciter_master = []
            for el_B, kp_B in k_params.items():
                w_B = sample_conc.get(el_B, 0.0)
                if w_B <= 0:
                    continue
                for E_line, P_line in [(kp_B["E_Ka"], kp_B["P_Ka"]),
                                       (kp_B["E_Kb"], kp_B["P_Kb"])]:
                    if E_line <= 0 or P_line <= 0:
                        continue
                    exciter_master.append(dict(
                        element   = el_B,
                        E_line_B  = E_line,
                        mu_B      = float(f_mu_interp(E_line)),
                        E_edge_B  = kp_B["E_edge"],
                        yield_B   = w_B * kp_B["wK"] * kp_B["R_jump"] * P_line,
                        cs_photo_B= cs_photo_cache[el_B],
                    ))
            # Per analyte A: keep only higher-energy exciters (E_Bb > A K-edge),
            # attach τ_A(E_Bb) = CS_Photo(Z_A, E_Bb).
            for el_A, kp_A in k_params.items():
                if sample_conc.get(el_A, 0.0) <= 0:
                    continue
                Z_A     = kp_A["Z"]
                E_edge_A = kp_A["E_edge"]
                lst = []
                for ex in exciter_master:
                    if ex["element"] == el_A:
                        continue
                    if ex["E_line_B"] <= E_edge_A:
                        continue
                    lst.append(dict(
                        E_line_B        = ex["E_line_B"],
                        mu_B            = ex["mu_B"],
                        E_edge_B        = ex["E_edge_B"],
                        yield_B         = ex["yield_B"],
                        cs_photo_B      = ex["cs_photo_B"],
                        cs_photo_A_at_B = xraylib.CS_Photo(Z_A, ex["E_line_B"]),
                    ))
                if lst:
                    exc_by_analyte[el_A] = lst

        # Accumulators keyed by (element_symbol, line_label_string)
        line_totals: dict = {}
        I_coh_total   = np.zeros_like(energy_kev)
        I_incoh_total = np.zeros_like(energy_kev)

        # ── Grid loop ─────────────────────────────────────────────────────────
        ny, nx = valid_arr.shape
        for iy in range(ny):
            for ix in range(nx):
                if not valid_arr[iy, ix]:
                    continue

                sinPhi = float(sinPhi_arr[iy, ix])
                sinPsi = float(sinPsi_arr[iy, ix])
                R_src  = float(R_src_arr[iy, ix])        # cm
                R_det  = float(R_det_arr[iy, ix])        # cm
                Omega  = float(Omega_arr[iy, ix])        # sr

                # φ(E, r_mn) = P_eff · sinΦ / R_src²  [ph/cm²/s/keV]
                phi_mn = P_eff * sinPhi / (R_src ** 2)

                # (Ω/4π)·ΔS — for isotropic fluorescence emission
                cell_factor = (Omega / (4.0 * np.pi)) * geo.dS
                # Ω·ΔS — for coherent scattering: sigma_coh is dσ/dΩ [cm²/g/sr],
                # already evaluated at the exact scattering angle, so multiply by Ω directly
                coh_factor = Omega * geo.dS

                # ── Coherent (Rayleigh) scattering ────────────────────────────
                p_mn  = np.array([geo.XX[iy, ix], geo.YY[iy, ix], 0.0])
                k_in  = (p_mn - geo.r_src) / R_src
                k_out = (geo.r_det - p_mn) / R_det
                psi   = np.arccos(np.clip(np.dot(k_in, k_out), -1.0, 1.0))
                sigma_coh = compute_coherent_cross_section(
                    energy_kev, psi, element_properties.param_dict,
                    sample_conc, anomalous_cache,
                )
                denom_coh = sample_mu * (1.0 + sinPhi / sinPsi)
                I_coh_total += coh_factor * np.where(
                    denom_coh > 0, phi_mn * sigma_coh / denom_coh, 0.0)

                # ── Incoherent (Compton) scattering ───────────────────────────
                # Scattered photon energy for each incident energy bin [keV].
                # Compton formula: E' = E / (1 + (E/511)(1 - cos θ))
                E_compton = energy_kev / (
                    1.0 + (energy_kev / 511.0) * (1.0 - np.cos(psi))
                )
                # Differential cross section: Klein-Nishina × S(x,Z) [cm²/g/sr]
                # Ref: Hubbell et al. (1975), J. Phys. Chem. Ref. Data 4, 471.
                sigma_incoh = compute_incoherent_cross_section(
                    energy_kev, psi, element_properties.param_dict, sample_conc
                )
                # Attenuation: incident beam at E, scattered beam at E'.
                # Separate μ values because E ≠ E' (unlike Rayleigh).
                # Ref: Tee et al. (2025), X-Ray Spectrometry 54, 133.
                mu_compton  = f_mu_interp(E_compton)
                denom_incoh = sample_mu + (sinPhi / sinPsi) * mu_compton
                # Raw spectral density [ph/s/keV] at incident energy E_i.
                incoh_raw = coh_factor * np.where(
                    denom_incoh > 0, phi_mn * sigma_incoh / denom_incoh, 0.0
                )
                # Jacobian |dE/dE'| = (E/E')²: converts the density from
                # incident-energy coordinates to scattered-energy coordinates.
                jacobian = (energy_kev / E_compton) ** 2
                incoh_at_scattered = incoh_raw * jacobian
                # Redistribute onto the output energy grid using the
                # (monotone) mapping E_i → E'_i via linear interpolation.
                I_incoh_total += np.interp(
                    energy_kev, E_compton, incoh_at_scattered,
                    left=0.0, right=0.0,
                )

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
                            energy_kev, phi_mn, Z, sinPhi, sinPsi,
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
                            energy_kev, phi_mn, Z, sinPhi, sinPsi,
                            sample_mu, mu_Kb, kp["E_edge"],
                        )
                        key = (el, "Kb")
                        line_totals[key] = (line_totals.get(key, 0.0)
                                            + cell_factor * raw * w
                                            * kp["wK"] * kp["R_jump"] * kp["P_Kb"])

                    # Secondary (inter-element) K fluorescence — adds into the
                    # same Kα/Kβ accumulators with the identical A-side
                    # multiplier (cell_factor · w · ωK · R_jump · p_line).
                    exciters = exc_by_analyte.get(el) if include_secondary else None
                    if exciters:
                        if kp["E_Ka"] > 0 and kp["P_Ka"] > 0:
                            mu_Ka = float(f_mu_interp(kp["E_Ka"]))
                            raw_s = compute_k_secondary_fluorescence(
                                energy_kev, phi_mn, sinPhi, sinPsi,
                                sample_mu, mu_Ka, kp["E_edge"], exciters,
                            )
                            key = (el, "Ka")
                            line_totals[key] = (line_totals.get(key, 0.0)
                                                + cell_factor * raw_s * w
                                                * kp["wK"] * kp["R_jump"] * kp["P_Ka"])
                        if kp["E_Kb"] > 0 and kp["P_Kb"] > 0:
                            mu_Kb = float(f_mu_interp(kp["E_Kb"]))
                            raw_s = compute_k_secondary_fluorescence(
                                energy_kev, phi_mn, sinPhi, sinPsi,
                                sample_mu, mu_Kb, kp["E_edge"], exciters,
                            )
                            key = (el, "Kb")
                            line_totals[key] = (line_totals.get(key, 0.0)
                                                + cell_factor * raw_s * w
                                                * kp["wK"] * kp["R_jump"] * kp["P_Kb"])

                    # L-shell (all lines with non-zero rate for this Z)
                    for r in compute_l_fluorescence(
                        energy_kev, phi_mn, Z, sinPhi, sinPsi,
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
            pd.DataFrame({"Energy_keV":   energy_kev,
                          "I_coherent":   I_coh_total,
                          "I_incoherent": I_incoh_total}).to_excel(
                writer, sheet_name="Scattering", index=False)
            df_fluo.to_excel(writer, sheet_name="Fluorescence", index=False)
        print(f"→ Saved to {xlsx_path}")

        # ── Simulated detector spectrum ───────────────────────────────────────
        be_mm         = float(cfg.get("be_window_mm",            0.125))
        si_active_um  = float(cfg.get("si_active_layer_um",     450.0))
        fano          = float(cfg.get("detector_fano",           0.115))
        enc_electrons = float(cfg.get("detector_enc_electrons",  8.0))

        spectrum = build_spectrum(
            energy_kev, I_coh_total, df_fluo,
            be_mm=be_mm, si_active_um=si_active_um,
            fano=fano, enc_electrons=enc_electrons,
            I_incoherent=I_incoh_total,
        )

        spec_xlsx_path = out_dir / f"{run_name}_{conc}_spectrum.xlsx"
        pd.DataFrame({"Energy_keV": energy_kev,
                      "Intensity_ph_s_per_keV": spectrum}).to_excel(
            spec_xlsx_path, index=False)
        print(f"→ Saved spectrum to {spec_xlsx_path}")

        # PyMca-compatible ASCII export (File → Open in PyMca)
        dat_path = out_dir / f"{run_name}_{conc}_spectrum.dat"
        with open(dat_path, "w") as fh:
            fh.write(f"#F {dat_path.name}\n")
            fh.write(f"#S 1  XRF simulated spectrum — {conc}\n")
            fh.write("#N 2\n")
            fh.write("#L Energy[keV]  Intensity[ph/s/keV]\n")
            for e, s in zip(energy_kev, spectrum):
                fh.write(f"{e:.6f}  {s:.6e}\n")
        print(f"→ Saved PyMca file to {dat_path}")

        # ORTEC PMCA .mca export — integer counts per channel, readable by PyMCA
        live_time = float(cfg.get("mca_live_time_s", 1.0))
        mca_path  = out_dir / f"{run_name}_{conc}_spectrum.mca"
        save_mca(energy_kev, spectrum, str(mca_path),
                 description=f"XRF simulated spectrum — {conc}",
                 live_time=live_time)

        plot_path = out_dir / f"{run_name}_{conc}_spectrum.png"
        src = geo.r_src
        det = geo.r_det
        src_dist = float(src[2])
        det_dist = float(det[2])
        src_angle = float(np.degrees(np.arctan2(np.sqrt(src[0]**2 + src[1]**2), src[2])))
        det_angle = float(np.degrees(np.arctan2(np.sqrt(det[0]**2 + det[1]**2), det[2])))
        geo_info = dict(src_dist_cm=src_dist, src_angle_deg=src_angle,
                        det_dist_cm=det_dist, det_angle_deg=det_angle)
        plot_and_save_spectrum(energy_kev, spectrum, df_fluo, plot_path,
                               conc_label=conc, geo_info=geo_info)

        results_dict[conc] = {
            "energy":        energy_kev,
            "I_coherent":    I_coh_total,
            "I_incoherent":  I_incoh_total,
            "fluorescence":  df_fluo,
            "out_dir":       str(out_dir),
            "n_valid_cells": n_valid,
        }

    return results_dict
