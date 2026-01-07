# src/xrf_model/pipeline.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .io import (
    load_flux_data,
    load_element_mass_absorption_data,
    interpolate_element_full_data,
    load_form_factor_data,
)
from .physics import compute_coherent_scattering_sample, compute_k_fluorescence
from .output import save_coherent_spectrum_txt


def run_one(flux_file: str | Path, cfg: dict) -> dict:
    flux_file = Path(flux_file)
    flare_name = flux_file.stem

    # --- Load flare flux
    energy_solar_flare, flux_solar_flare_0 = load_flux_data(str(flux_file))

    # --- Load element parameters
    df = pd.read_excel(cfg["paths"]["element_data"], index_col=0)
    concentration_cols = cfg["concentration_cols"]

    theta = cfg["theta"]
    phi = cfg["phi"]
    altitude = cfg["altitude"]
    distance_sun_AU = cfg["distance_sun_AU"]
    footprint = cfg["footprint"]

    # Your code uses 1/altitude^2 as "solid angle" factor
    solid_angle = 1 / (altitude ** 2)
    scattering_angle = np.pi - (theta + phi)

    # --- Build param_dict
    param_dict = {}
    for el in cfg["all_matrix_elements"]:
        pdata = {
            "f_rel": df.at[el, "f_REL"],
            "fNT": df.at[el, "f_NT"],
            "Atomic Number": df.at[el, "Atomic Number"],
            "Atomic Mass": df.at[el, "Atomic Mass"],
        }
        for key in concentration_cols:
            pdata[key] = df.at[el, key]
        param_dict[el] = pdata

    # --- Load & interpolate mass absorption data
    element_xs_data = {}
    data_dir = cfg["paths"]["absorption_dir"]
    for el in param_dict:
        txt_path = os.path.join(data_dir, f"{el}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Missing element file: {txt_path}")
        element_xs_data[el] = load_element_mass_absorption_data(txt_path)

    common_energy = energy_solar_flare.copy()
    full_data_dict = {
        el: interpolate_element_full_data(xs_data, common_energy)
        for el, xs_data in element_xs_data.items()
    }

    # --- Load form factors
    form_factor_dict = load_form_factor_data(cfg["paths"]["form_factors"])

    # --- Output dir per flare (prevents overwriting)
    out_dir = Path(cfg["paths"]["output_dir"]) / flare_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {}

    # --- Loop concentrations
    for conc in concentration_cols:
        sample_mass_atten_array = np.zeros_like(common_energy, dtype=float)
        for el_name, params in param_dict.items():
            w = params[conc]
            if w > 0:
                sample_mass_atten_array += (
                    w * full_data_dict[el_name]["total_mass_attenuation_interp"]
                )

        scale = (footprint * np.cos(theta)) / (distance_sun_AU ** 2)
        flux_scaled = flux_solar_flare_0 * scale * solid_angle

        I_coh = compute_coherent_scattering_sample(
            energy_array=common_energy,
            flux_array=flux_scaled,
            sample_mass_atten_array=sample_mass_atten_array,
            param_dict=param_dict,
            form_factor_dict=form_factor_dict,
            full_data_dict=full_data_dict,
            concentration_key=conc,
            scattering_angle=scattering_angle,
            theta=theta,
            phi=phi,
        )

        fluorescence_rows = []
        f_atten_interp = interp1d(common_energy, sample_mass_atten_array,
                                  bounds_error=False, fill_value="extrapolate")

        for el, params in param_dict.items():
            w = params[conc]
            if w <= 0 or el not in cfg["fluorescence_elements"]:
                continue

            # Use your existing df columns (same logic as notebook)
            E_edge = float(df.at[el, "K-edge Line"])
            wK = float(df.at[el, "wK"])
            R = float(df.at[el, "K alpha ratio"])

            # --- Kα ---
            E_Ka = float(df.at[el, "K-alpha Line"])
            P_Ka = float(df.at[el, "P_kalpha"])
            mu_Ka = float(f_atten_interp(E_Ka))

            raw_Ka = compute_k_fluorescence(
                energy_array=common_energy,
                flux_array=flux_scaled,
                photoelectric_array_element=full_data_dict[el]["photoelectric_absorption_interp"],
                total_atten_sample_array=sample_mass_atten_array,
                total_atten_sample_at_line=mu_Ka,
                absorption_edge=E_edge,
                theta=theta,
                phi=phi,
            )
            I_Ka = (raw_Ka * w * wK * R * P_Ka) / (4 * np.pi)

            fluorescence_rows.append({
                "Element": el, "Concentration": w, "Line": "Kα",
                "E_line_keV": E_Ka, "P_line": P_Ka, "Intensity_ph_cm2_s": I_Ka
            })

            # --- Kβ (optional if present)
            E_Kb = float(df.at[el, "K-beta Line"]) if "K-beta Line" in df.columns else 0.0
            P_Kb = float(df.at[el, "P_kbeta"])     if "P_kbeta"     in df.columns else 0.0
            if E_Kb > 0.0 and P_Kb > 0.0:
                mu_Kb = float(f_atten_interp(E_Kb))
                raw_Kb = compute_k_fluorescence(
                    energy_array=common_energy,
                    flux_array=flux_scaled,
                    photoelectric_array_element=full_data_dict[el]["photoelectric_absorption_interp"],
                    total_atten_sample_array=sample_mass_atten_array,
                    total_atten_sample_at_line=mu_Kb,
                    absorption_edge=E_edge,
                    theta=theta,
                    phi=phi,
                )
                I_Kb = (raw_Kb * w * wK * R * P_Kb) / (4 * np.pi)
            else:
                I_Kb = 0.0

            fluorescence_rows.append({
                "Element": el, "Concentration": w, "Line": "Kβ",
                "E_line_keV": E_Kb if E_Kb > 0 else np.nan,
                "P_line": P_Kb, "Intensity_ph_cm2_s": I_Kb
            })

        df_fluo = pd.DataFrame(fluorescence_rows).sort_values(["Element", "Line"])

        # Save outputs (same as your notebook but inside results/<cfg>/<flare>/)
        xlsx_path = out_dir / f"{flare_name}_{conc}_results.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame({"Energy": common_energy, "I_coherent": I_coh}).to_excel(
                writer, sheet_name="Coherent", index=False
            )
            df_fluo.to_excel(writer, sheet_name="Fluorescence", index=False)

        save_coherent_spectrum_txt(flare_name, conc, common_energy, I_coh, out_dir=str(out_dir))

        results_dict[conc] = {
            "energy": common_energy,
            "I_coherent": I_coh,
            "fluorescence": df_fluo,
            "out_dir": str(out_dir),
        }

    return results_dict


def run_many(flux_files: list[str | Path], cfg: dict) -> dict:
    all_results = {}
    for f in flux_files:
        all_results[Path(f).stem] = run_one(f, cfg)
    return all_results
