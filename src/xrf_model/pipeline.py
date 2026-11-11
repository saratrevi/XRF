# src/xrf_model/pipeline.py
from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path
import numpy as np
import pandas as pd
import xraylib

from .io import load_flux_data
from .output import save_mca
from .spectrum import build_spectrum
from .plot_spectrum import plot_and_save_spectrum
from .physics import (
    ElementProperties,
    compute_k_fluorescence,
    compute_k_secondary_fluorescence,
    compute_l_fluorescence,
    l_line_cross_sections,
    mass_attenuation,
    compute_coherent_cross_section,
    compute_incoherent_cross_section,
    precompute_anomalous_factors,
)
from .geometry import GridGeometry
from .tube import tube_source_term
from .atmosphere import attenuation_coefficient
from .detector_response import load_response, build_spectrum_with_response
from .compton import compton_kernel, sample_profile
from .energy_grid import bin_edges
from .output import read_calibration


def _k_group(Z: int, line_attr_names: list) -> tuple:
    """
    Energy-weighted mean line energy and total branching fraction for a K-line group.

    Used only for an element's role as an exciter in secondary fluorescence,
    where splitting the group changes almost nothing (it would matter only if
    a group straddled the absorption edge of the excited element).

    line_attr_names: list of xraylib attribute name strings, e.g. ["KA1_LINE", "KA2_LINE"].
    Lines absent from the installed xraylib version are skipped.
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
        except ValueError:            # line not tabulated for this element
            continue
        if r > 0 and e > 0:
            total += r
            e_sum += r * e
    return (e_sum / total if total > 0 else 0.0), total


def _k_shell_params(Z: int) -> dict | None:
    """
    K-shell edge [keV], fluorescence yield and absorption-jump factor of Z.

    Returns None if xraylib has no K-shell data for Z (it raises ValueError,
    e.g. for H) or the values cannot give K lines.  Any other error is a real
    problem and is not caught.
    """
    try:
        E_edge = xraylib.EdgeEnergy(Z, xraylib.K_SHELL)
        wK = xraylib.FluorYield(Z, xraylib.K_SHELL)
        jump_r = xraylib.JumpFactor(Z, xraylib.K_SHELL)
    except ValueError:
        return None
    if E_edge <= 0 or wK <= 0 or jump_r <= 1.0:
        return None
    return {"E_edge": E_edge, "wK": wK, "R_jump": 1.0 - 1.0 / jump_r}   # (r-1)/r


# Individual K lines: (display name, xraylib attribute name)
_K_LINES = [("Kα1", "KA1_LINE"), ("Kα2", "KA2_LINE"),
            ("Kβ1", "KB1_LINE"), ("Kβ3", "KB3_LINE"), ("Kβ2", "KB2_LINE")]


def _k_lines(Z: int) -> list:
    """K lines of element Z as dicts name, E [keV], P (radiative rate); missing lines skipped."""
    lines = []
    for name, attr in _K_LINES:
        lc = getattr(xraylib, attr, None)
        if lc is None:
            continue
        try:
            E, P = xraylib.LineEnergy(Z, lc), xraylib.RadRate(Z, lc)
        except ValueError:            # line not tabulated for this element
            continue
        if E > 0 and P > 0:
            lines.append({"name": name, "E": E, "P": P})
    return lines


def run_one_grid(flux_file, cfg: dict, debugging: bool = False) -> dict:
    """
    Spatially-resolved K + L fluorescence using a 2D surface grid.

    Implements:
        I_det(ip) = Σ_{m,n}  dΩ/(4π) · ΔS
                    · ∫ Q_ip(E) φ(E,r_mn) / [μ(E)/sinΦ + μ(ip)/sinΨ] dE

    where φ(E, r_mn) = P(E) · sinΦ_mn / R_src_mn², with R_src_mn the
    straight-line distance from the focal spot to the point (not the height).

    Atmosphere
    ----------
    With atmosphere = air, the incoming spectrum is multiplied by
    exp(−μ_air(E) · L_in) and every outgoing photon by exp(−μ_air(E_out) · L_out),
    at its own energy (line energy, Rayleigh E, Compton E').  L_in and L_out
    are the path lengths outside the housings: R_src − src_vacuum_path_cm and
    R_det − det_vacuum_path_cm.  With atmosphere = vacuum (default) nothing is
    absorbed.

    Source spectrum
    ---------------
    P(E) is the source term in ph/s/keV/sr, which does not depend on distance.

    If the config has a ``tube:`` block, P(E) is generated with SpekPy (see
    tube.py) and flux_file must be None.

    Otherwise flux_file is read.  It delivers P_raw(E) in ph/cm²/s/keV at
    reference distance src_reference_distance_cm [cm] from the source, on the
    beam axis (flux_units = ph_per_cm2_per_s_per_keV, the only option).  The
    code converts it to the source term:

        P_eff(E) = P_raw(E) × d_ref²   [ph/s/keV/sr]

    so that phi_mn = P_eff × sinΦ / R_cm² is in ph/cm²/s/keV and the
    accumulated output intensity is in ph/s.

    All atomic parameters are fetched from xraylib (no spreadsheet look-ups).
    All distances are in cm.  Cross-sections from xraylib are in cm²/g.
    """
    if "detector_enc_electrons" in cfg:
        raise ValueError("detector_enc_electrons was replaced by detector_noise_eV "
                         "(= ENC × w) and detector_w_eV; see configs/lab_xrf.yml.")

    # .mca channels: a real MCA's calibration, read now so a bad file fails early
    mca_calibration = (read_calibration(cfg["mca_calibration"])
                       if cfg.get("mca_calibration") else None)
    mca_channels = cfg.get("mca_channels")
    if mca_calibration is not None and not mca_channels:
        raise ValueError("mca_calibration needs mca_channels (the number of MCA channels)")

    # ── Source spectrum ───────────────────────────────────────────────────────
    tube_cfg = cfg.get("tube")
    if tube_cfg is not None:
        if flux_file is not None:
            raise ValueError("The config has a `tube:` block, so the spectrum is "
                             "generated: pass flux_file=None, or remove the "
                             "block to use the file.")
        energy_kev, P_eff = tube_source_term(tube_cfg)   # ph/s/keV/sr
        run_name = f"tube_{float(tube_cfg['kvp']):g}kV_{tube_cfg['target']}"
    elif flux_file is None:
        raise ValueError("No source spectrum: pass a flux file or add a "
                         "`tube:` block to the config.")
    else:
        flux_file = Path(flux_file)
        run_name  = flux_file.stem
        energy_kev, flux_raw = load_flux_data(str(flux_file))

        flux_units = cfg.get("flux_units", "ph_per_cm2_per_s_per_keV")
        if flux_units != "ph_per_cm2_per_s_per_keV":
            raise ValueError(f"Unknown flux_units: {flux_units!r} "
                             "(the only option is ph_per_cm2_per_s_per_keV)")
        # flux_raw: ph/cm²/s/keV at d_ref [cm] from the source, on the beam
        # axis.  The fluence falls as 1/d², so × d_ref² gives ph/s/keV/sr.
        d_ref = float(cfg["src_reference_distance_cm"])
        P_eff = flux_raw * (d_ref ** 2)                  # ph/s/keV/sr

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
    # Elements without K-shell data in xraylib (e.g. H) get no lines; they are
    # named in the output and still count in the attenuation and scattering.
    k_params: dict = {}
    no_k_data = []
    for el in active_elements:
        Z = element_properties.param_dict[el]["Atomic Number"]
        shell = _k_shell_params(Z)
        if shell is None:
            no_k_data.append(el)
            continue

        # Individual lines for this element's own K emission; the Kα and Kβ
        # groups are kept only for its role as an exciter of other elements.
        E_Ka, P_Ka = _k_group(Z, ["KA1_LINE", "KA2_LINE"])
        E_Kb, P_Kb = _k_group(Z, ["KB1_LINE", "KB2_LINE", "KB3_LINE"])
        k_params[el] = dict(Z=Z, **shell,
                            E_Ka=E_Ka, P_Ka=P_Ka, E_Kb=E_Kb, P_Kb=P_Kb,
                            lines=_k_lines(Z))
    if no_k_data:
        print(f"No K-shell data in xraylib, no lines computed for: {', '.join(no_k_data)}")

    # L-line cross sections (with Coster-Kronig and K→L cascade) depend only on
    # Z and E: compute once per element, reuse in every cell and sample.
    l_lines_by_el = {el: l_line_cross_sections(kp["Z"], energy_kev)
                     for el, kp in k_params.items()}

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

    # ── Atmosphere between tube, sample and detector ──────────────────────────
    mu_air = attenuation_coefficient(cfg.get("atmosphere", "vacuum"))   # 1/cm
    mu_air_grid = mu_air(energy_kev)
    src_vacuum = float(cfg.get("src_vacuum_path_cm", 0.0))
    det_vacuum = float(cfg.get("det_vacuum_path_cm", 0.0))
    # ── Detector response matrix (optional; otherwise Gaussian peaks) ─────────
    response_cfg = cfg.get("detector_response")
    response = load_response(response_cfg["file"]) if response_cfg else None

    for kp in k_params.values():
        for line in kp["lines"]:
            line["mu_air"] = float(mu_air(line["E"]))

    out_dir = Path(cfg["paths"]["output_dir"]) / (run_name + "_grid")
    out_dir.mkdir(parents=True, exist_ok=True)

    # f1/f2 depend only on energy — precompute once, reuse across all cells
    anomalous_cache = precompute_anomalous_factors(energy_kev,
                                                   element_properties.param_dict)
    # Compton: incident bin widths and output bin edges on the same grid
    dE_in = np.gradient(energy_kev)
    out_edges = bin_edges(energy_kev)
    results_dict = {}

    for conc in conc_cols:
        print('=' * 70)
        print(f'SAMPLE: {conc}')
        print('=' * 70)

        sample_conc = concentrations[conc]

        # Sample total mass attenuation μ(E)/ρ  [cm²/g]: Σ_j  w_j · CS_Total(Z_j, E)
        # on the beam grid, and at the exact energy of each line (cached),
        # never interpolated across an absorption edge or outside the grid.
        sample_mu = mass_attenuation(element_properties.param_dict, sample_conc,
                                     energy_kev)
        mu_at_line = lru_cache(maxsize=None)(
            partial(mass_attenuation, element_properties.param_dict, sample_conc))

        # Compton profile of this sample (shape of the Doppler broadening)
        compton_profile = sample_profile(element_properties.param_dict, sample_conc)

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
                        mu_B      = mu_at_line(E_line),
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

                # Air path: only the part outside the tube and detector
                # housings (vacuum inside) is in air.
                L_in  = max(R_src - src_vacuum, 0.0)
                L_out = max(R_det - det_vacuum, 0.0)
                T_out = np.exp(-mu_air_grid * L_out)   # for outgoing photons at E

                # φ(E, r_mn) = P_eff · sinΦ / R_src² · T_in(E)  [ph/cm²/s/keV]
                phi_mn = (P_eff * sinPhi / (R_src ** 2)
                          * np.exp(-mu_air_grid * L_in))

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
                I_coh_total += coh_factor * T_out * np.where(
                    denom_coh > 0, phi_mn * sigma_coh / denom_coh, 0.0)

                # ── Incoherent (Compton) scattering ───────────────────────────
                # Total per incident energy: Klein-Nishina × S(x,Z) [cm²/g/sr]
                # Ref: Hubbell et al. (1975), J. Phys. Chem. Ref. Data 4, 471.
                sigma_incoh = compute_incoherent_cross_section(
                    energy_kev, psi, element_properties.param_dict, sample_conc
                )
                # Photons per incident bin [ph/s], before attenuation in the sample
                rate_in = coh_factor * phi_mn * sigma_incoh * dE_in
                # Doppler-broadened spread over the scattered energy E' (the
                # output grid): kernel[i, j] [1/keV] from the sample's Compton
                # profile, impulse approximation with binding cutoffs.
                kernel = compton_kernel(compton_profile, energy_kev, out_edges, psi)
                # Attenuation: incident beam at E, scattered beam at E'.
                # Ref: Tee et al. (2025), X-Ray Spectrometry 54, 133.
                denom_incoh = (sample_mu[:, None]
                               + (sinPhi / sinPsi) * sample_mu[None, :])
                # The output grid is the scattered energy, so T_out applies there.
                I_incoh_total += T_out * (rate_in[:, None] * kernel
                                          / denom_incoh).sum(axis=0)

                # ── K + L fluorescence ────────────────────────────────────────
                for el, kp in k_params.items():
                    w = sample_conc.get(el, 0.0)
                    if w <= 0:
                        continue

                    Z = kp["Z"]
                    exciters = exc_by_analyte.get(el) if include_secondary else None

                    # K lines, each at its own energy (Kα1, Kα2, Kβ1, Kβ3, Kβ2):
                    # primary + secondary (inter-element) fluorescence, with the
                    # multiplier cell_factor · T_line · w · ωK · R_jump · p_line.
                    for line in kp["lines"]:
                        mu_line = mu_at_line(line["E"])
                        raw = compute_k_fluorescence(
                            energy_kev, phi_mn, Z, sinPhi, sinPsi,
                            sample_mu, mu_line, kp["E_edge"],
                        )
                        if exciters:
                            raw += compute_k_secondary_fluorescence(
                                energy_kev, phi_mn, sinPhi, sinPsi,
                                sample_mu, mu_line, kp["E_edge"], exciters,
                            )
                        T_line = np.exp(-line["mu_air"] * L_out)   # air to the detector
                        key = (el, line["name"])
                        line_totals[key] = (line_totals.get(key, 0.0)
                                            + cell_factor * T_line * raw * w
                                            * kp["wK"] * kp["R_jump"] * line["P"])

                    # L-shell (all lines with non-zero rate for this Z)
                    for r in compute_l_fluorescence(
                        energy_kev, phi_mn, sinPhi, sinPsi,
                        sample_mu, mu_at_line, l_lines_by_el[el],
                    ):
                        key = (el, r["line_name"])
                        T_L = np.exp(-float(mu_air(r["E_line"])) * L_out)
                        line_totals[key] = (line_totals.get(key, 0.0)
                                            + cell_factor * T_L * w * r["I_raw"])

        # ── Build output DataFrame ────────────────────────────────────────────
        rows = []
        for el, kp in k_params.items():
            w = sample_conc.get(el, 0.0)
            Z = kp["Z"]

            # K lines, one row per line
            for line in kp["lines"]:
                rows.append({
                    "Element":        el,
                    "Concentration":  w,
                    "Line":           line["name"],
                    "E_line_keV":     line["E"],
                    "P_line":         line["P"],
                    "Intensity_ph_s": line_totals.get((el, line["name"]), 0.0),
                })

            # L lines — energy and rate as already found by l_line_cross_sections
            for line in l_lines_by_el[el]:
                key = (el, line["line_name"])
                if key not in line_totals:
                    continue
                rows.append({
                    "Element":        el,
                    "Concentration":  w,
                    "Line":           line["line_name"],
                    "E_line_keV":     line["E_line"],
                    "P_line":         line["rad_rate"],
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
        be_mm        = float(cfg.get("be_window_mm",       0.125))
        si_active_um = float(cfg.get("si_active_layer_um", 450.0))
        fano         = float(cfg.get("detector_fano",      0.115))
        w_ev         = float(cfg.get("detector_w_eV",      3.62))
        noise_ev     = float(cfg.get("detector_noise_eV",  8.0 * 3.62))

        if response is not None:
            # spectrum on the matrix channels
            spec_energy, spectrum = build_spectrum_with_response(
                energy_kev, I_coh_total, df_fluo, response,
                rows=response_cfg.get("rows", "per_detected_photon"),
                be_mm=be_mm, si_active_um=si_active_um,
                fano=fano, noise_ev=noise_ev, w_ev=w_ev,
                I_incoherent=I_incoh_total,
            )
        else:
            spec_energy = energy_kev
            spectrum = build_spectrum(
                energy_kev, I_coh_total, df_fluo,
                be_mm=be_mm, si_active_um=si_active_um,
                fano=fano, noise_ev=noise_ev, w_ev=w_ev,
                I_incoherent=I_incoh_total,
            )

        spec_xlsx_path = out_dir / f"{run_name}_{conc}_spectrum.xlsx"
        pd.DataFrame({"Energy_keV": spec_energy,
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
            for e, s in zip(spec_energy, spectrum):
                fh.write(f"{e:.6f}  {s:.6e}\n")
        print(f"→ Saved PyMca file to {dat_path}")

        # ORTEC PMCA .mca export — integer counts per channel, readable by PyMCA
        live_time = float(cfg.get("mca_live_time_s", 1.0))
        mca_path  = out_dir / f"{run_name}_{conc}_spectrum.mca"
        save_mca(spec_energy, spectrum, str(mca_path),
                 description=f"XRF simulated spectrum — {conc}",
                 live_time=live_time,
                 calibration=mca_calibration, n_channels=mca_channels)

        plot_path = out_dir / f"{run_name}_{conc}_spectrum.png"
        src = geo.r_src
        det = geo.r_det
        src_dist = float(src[2])
        det_dist = float(det[2])
        src_angle = float(np.degrees(np.arctan2(np.sqrt(src[0]**2 + src[1]**2), src[2])))
        det_angle = float(np.degrees(np.arctan2(np.sqrt(det[0]**2 + det[1]**2), det[2])))
        geo_info = dict(src_dist_cm=src_dist, src_angle_deg=src_angle,
                        det_dist_cm=det_dist, det_angle_deg=det_angle)
        plot_and_save_spectrum(spec_energy, spectrum, df_fluo, plot_path,
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
