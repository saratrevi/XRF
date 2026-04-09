# src/xrf_model/pipeline.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
# from scipy.interpolate import interp1d

# NOTE: a useful library: xraylib

from .io import (
    load_flux_data,
    # load_element_mass_absorption_data,
    # interpolate_element_full_data,
    # load_form_factor_data,
)
from .physics import compute_coherent_scattering_sample, compute_k_fluorescence
from .physics import ElementProperties, compute_fluorescence_spectrum
from .output import save_coherent_spectrum_txt

def safe_float(x, default=0.0) -> float:
    """Convert x to float; return default if x is missing/NaN/invalid."""
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def run_one(flux_file: str | Path, cfg: dict) -> dict:
    flux_file = Path(flux_file)
    flare_name = flux_file.stem

    # --- Load flare flux
    energy_solar_flare, flux_solar_flare_0 = load_flux_data(str(flux_file))

    # --- Load element parameters
    df = pd.read_excel(cfg["paths"]["element_data"], index_col=0)
    df = df.loc[:, 'mantle_mars':] # NOTE discard all non-concentration columns

    concentration_cols = cfg["concentration_cols"]

    # theta = cfg["theta"] # NOTE: incidence angle w.r.t. the normal vector (which is the POV)
    # phi = cfg["phi"] # NOTE: angle of emission w.r.t. the normal vector
    # altitude = cfg["altitude"] # NOTE: this should become our main distance
    # distance_sun_AU = cfg["distance_sun_AU"] # NOTE: this should not matter in the in-situ setting
    # footprint = cfg["footprint"] # NOTE: this will represent the area of a grid cell

    # Your code uses 1/altitude^2 as "solid angle" factor
    # solid_angle = 1 / (altitude ** 2) # NOTE: we could do everything with this, possibly skipping altitude
    # scattering_angle = np.pi - (theta + phi) # NOTE: this should be correct

    # # --- Build param_dict
    # param_dict = {}
    # for el in cfg["all_matrix_elements"]:
    #     pdata = { # NOTE: this are all fixed properties of elements
    #         "f_rel": df.at[el, "f_REL"],
    #         "fNT": df.at[el, "f_NT"],
    #         "Atomic Number": df.at[el, "Atomic Number"],
    #         "Atomic Mass": df.at[el, "Atomic Mass"],
    #     }
    #     for key in concentration_cols: # NOTE: this will become a primary input (they are currently the last cols in Data_keV.xlsx)
    #         pdata[key] = df.at[el, key]
    #     param_dict[el] = pdata

    # Retrieve the cocentrations
    concentrations = {conc:{} for conc in concentration_cols}
    for el in cfg["all_matrix_elements"]:
        for key in concentration_cols:
            concentrations[key][el] = df.at[el, key]
    
    # Init the element property class
    element_properties = ElementProperties(cfg)

    # # NOTE: these are other element properties (curves of different energy-dependent properties)
    # # --- Load & interpolate mass absorption data 
    # element_xs_data = {}
    # data_dir = cfg["paths"]["absorption_dir"]
    # for el in param_dict:
    #     txt_path = os.path.join(data_dir, f"{el}.txt")
    #     if not os.path.exists(txt_path):
    #         raise FileNotFoundError(f"Missing element file: {txt_path}")
    #     element_xs_data[el] = load_element_mass_absorption_data(txt_path)

    # NOTE: this describes the illumination; the flux is an energy-dependent curve; this is another primary input
    # full_data_dict = {
    #     el: interpolate_element_full_data(xs_data, common_energy)
    #     for el, xs_data in element_xs_data.items()
    # }

    # NEW VERSION: not needed
    # element_properties.realign(energy_solar_flare)

    # NOTE: these are other element properties that depend on the energy-time-angle scalar
    # --- Load form factors
    # form_factor_dict = load_form_factor_data(cfg["paths"]["form_factors"])

    # --- Output dir per flare (prevents overwriting)
    out_dir = Path(cfg["paths"]["output_dir"]) / flare_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {}

    # NOTE: here we computing terms that will later used for the integration
    # --- Loop concentrations
    for conc in concentration_cols:

        print('='*78)
        print(f'SAMPLE: {conc}')
        print('='*78)

        sample_concentrations = concentrations[conc]

        # sample_mass_atten_array = np.zeros_like(common_energy, dtype=float)
        # for el_name, params in param_dict.items():
        #     w = params[conc]
        #     if w > 0:
        #         sample_mass_atten_array += (
        #             w * full_data_dict[el_name]["total_mass_attenuation_interp"]
        #         )


        #scale = (footprint * np.cos(theta)) / (distance_sun_AU ** 2) # NOTE: we'll need to either change this (for now: just move it ealiers)
        #flux_scaled = flux_solar_flare_0 * scale * solid_angle

        #I_coh = compute_coherent_scattering_sample(
        #    energy_array=common_energy,
        #    flux_array=flux_scaled,
        #    sample_mass_atten_array=sample_mass_atten_array,
        #    param_dict=param_dict,
        #    form_factor_dict=form_factor_dict,
        #    full_data_dict=full_data_dict,
        #    concentration_key=conc,
        #    scattering_angle=scattering_angle,
        #    theta=theta,
        #    phi=phi,
        #)

        #fluorescence_rows = []
        #f_atten_interp = interp1d(common_energy, sample_mass_atten_array,
        #                          bounds_error=False, fill_value="extrapolate")

        #for el, params in param_dict.items():
        #    w = params[conc]
        #    if w <= 0 or el not in cfg["fluorescence_elements"]:
        #        continue

        
        #    E_edge = safe_float(df.at[el, "Absorption Edge"], default=0.0)
        #    wK = safe_float(df.at[el, "Fluorescence Yield"], default=0.0)
        #    R  = safe_float(df.at[el, "r-1/r"], default=0.0)

            # --- Kα ---
        #    E_Ka = safe_float(df.at[el, "K-Alpha Line"], default=0.0)
        #    P_Ka = safe_float(df.at[el, "P_kalpha"], default=0.0)
        #    mu_Ka = float(f_atten_interp(E_Ka))

        #    if E_edge > 0 and E_Ka > 0 and P_Ka > 0:
        #        raw_Ka = compute_k_fluorescence(
        #        energy_array=common_energy,
        #        flux_array=flux_scaled,
        #        photoelectric_array_element=full_data_dict[el]["photoelectric_absorption_interp"],
        #        total_atten_sample_array=sample_mass_atten_array,
        #        total_atten_sample_at_line=mu_Ka,
        #        absorption_edge=E_edge,
        #        theta=theta,
        #        phi=phi,
        #        )
        #        I_Ka = (raw_Ka * w * wK * R * P_Ka) / (4 * np.pi)
        #    else:
        #        I_Ka = 0.0
            
            # NOTE: this is already output code
        #    fluorescence_rows.append({
        #        "Element": el, "Concentration": w, "Line": "Kα",
        #        "E_line_keV": E_Ka, "P_line": P_Ka, "Intensity_ph_cm2_s": I_Ka
        #    })
        
            # --- Kβ (optional if present)
        #    E_Kb = safe_float(df.at[el, "K-beta Line"], default=0.0)
        #    P_Kb = safe_float(df.at[el, "P_kbeta"], default=0.0)
        #    if E_Kb > 0.0 and P_Kb > 0.0:
        #        mu_Kb = float(f_atten_interp(E_Kb))
        #        raw_Kb = compute_k_fluorescence(
        #            energy_array=common_energy,
        #            flux_array=flux_scaled,
        #            photoelectric_array_element=full_data_dict[el]["photoelectric_absorption_interp"],
        #            total_atten_sample_array=sample_mass_atten_array,
        #            total_atten_sample_at_line=mu_Kb,
        #            absorption_edge=E_edge,
        #            theta=theta,
        #            phi=phi,
        #        )
        #        I_Kb = (raw_Kb * w * wK * R * P_Kb) / (4 * np.pi)
        #    else:
        #        I_Kb = 0.0

            # NOTE: output code
        #    fluorescence_rows.append({
        #        "Element": el, "Concentration": w, "Line": "Kβ",
        #        "E_line_keV": E_Kb if E_Kb > 0 else np.nan,
        #        "P_line": P_Kb, "Intensity_ph_cm2_s": I_Kb
        #    })

        
        results = compute_fluorescence_spectrum(
            conc=conc,
            conc_df=df,
            energy_solar_flare=energy_solar_flare,
            concentrations=sample_concentrations,
            element_properties=element_properties,
            flux_solar_flare=flux_solar_flare_0,

        )

        # NOTE: output code
        df_fluo = pd.DataFrame(results["rows"]).sort_values(["Element", "Line"])

        # Save outputs (same as your notebook but inside results/<cfg>/<flare>/)
        xlsx_path = out_dir / f"{flare_name}_{conc}_results.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame({"Energy": energy_solar_flare, "I_coherent": results["icoh"]}).to_excel(
                writer, sheet_name="Coherent", index=False
            )
            df_fluo.to_excel(writer, sheet_name="Fluorescence", index=False)

        save_coherent_spectrum_txt(flare_name, conc, energy_solar_flare, results["icoh"], out_dir=str(out_dir))

        results_dict[conc] = {
            "energy": energy_solar_flare,
            "I_coherent": results["icoh"],
            "fluorescence": df_fluo,
            "out_dir": str(out_dir),
        }

    return results_dict


def run_many(flux_files: list[str | Path], cfg: dict) -> dict:
    all_results = {}
    for f in flux_files:
        all_results[Path(f).stem] = run_one(f, cfg)
    return all_results
