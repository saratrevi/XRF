# src/xrf_model/physics.py
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.interpolate import interp1d

import pandas as pd
import os
from .io import (
    load_element_mass_absorption_data,
    interpolate_element_full_data,
    load_form_factor_data,
)

class ElementProperties:
    
    def __init__(self, cfg: dict):
        self.cfg = cfg

        df = pd.read_excel(cfg["paths"]["element_data"], index_col=0)
        self.df = df.loc[:, :'mantle_mars'] # NOTE keep only non-concentration columns

        # Retrieve element properties
        self.param_dict = {}
        for el in cfg["all_matrix_elements"]:
            pdata = {
                "f_rel": self.df.at[el, "f_REL"],
                "fNT": self.df.at[el, "f_NT"],
                "Atomic Number": self.df.at[el, "Atomic Number"],
                "Atomic Mass": self.df.at[el, "Atomic Mass"],
            }
            self.param_dict[el] = pdata

        # Absorption data
        # --- Load & interpolate mass absorption data 
        self.element_xs_data = {}
        data_dir = self.cfg["paths"]["absorption_dir"]
        for el in cfg["all_matrix_elements"]:
            txt_path = os.path.join(data_dir, f"{el}.txt")
            if not os.path.exists(txt_path):
                raise FileNotFoundError(f"Missing element file: {txt_path}")
            self.element_xs_data[el] = load_element_mass_absorption_data(txt_path)

        # Form factor data
        self.form_factor_dict = load_form_factor_data(cfg["paths"]["form_factors"])

    def realign(self, common_energy):
        self.full_data_dict = {
            el: interpolate_element_full_data(xs_data, common_energy)
            for el, xs_data in self.element_xs_data.items()
        }


def compute_fluorescence_spectrum(conc,
                                  conc_df,
                                  energy_solar_flare,
                                  concentrations : dict[str, float],
                                  element_properties : ElementProperties,
                                  flux_solar_flare):
    footprint = element_properties.cfg["footprint"]
    distance_sun_AU = element_properties.cfg["distance_sun_AU"]

    theta = element_properties.cfg["theta"]  # NOTE: incidence angle w.r.t. the normal vector (which is the POV)
    phi = element_properties.cfg["phi"]  # NOTE: angle of emission w.r.t. the normal vector
    solid_angle = 1 / (element_properties.cfg["altitude"] ** 2)  # NOTE: we could do everything with this, possibly skipping altitude
    scattering_angle = np.pi - (theta + phi)

    sample_mass_atten_array = np.zeros_like(energy_solar_flare, dtype=float)
    for el_name, w in concentrations.items():
        if w > 0:
            sample_mass_atten_array += (
                w * element_properties.full_data_dict[el_name]["total_mass_attenuation_interp"]
            )
    scale = (footprint * np.cos(theta)) / (distance_sun_AU ** 2) # NOTE: we'll need to either change this (for now: just move it ealiers)
    flux_scaled = flux_solar_flare * scale * solid_angle

    I_coh = compute_coherent_scattering_sample(
        energy_array=energy_solar_flare,
        flux_array=flux_scaled,
        sample_mass_atten_array=sample_mass_atten_array,
        param_dict=element_properties.param_dict,
        form_factor_dict=element_properties.form_factor_dict,
        full_data_dict=element_properties.full_data_dict,
        concentration_key=conc,
        conc_df=conc_df,
        scattering_angle=scattering_angle,
        theta=theta,
        phi=phi,
    )

    fluorescence_rows = []
    f_atten_interp = interp1d(energy_solar_flare, sample_mass_atten_array,
                              bounds_error=False, fill_value="extrapolate")

    for el, params in element_properties.param_dict.items():
        w = conc_df[conc][el]
        if w <= 0 or el not in element_properties.cfg["fluorescence_elements"]:
            continue

        
        E_edge = float(element_properties.df.at[el, "Absorption Edge"])
        wK = float(element_properties.df.at[el, "Fluorescence Yield"])
        R  = float(element_properties.df.at[el, "r-1/r"])

        # --- Kα ---
        E_Ka = float(element_properties.df.at[el, "K-Alpha Line"])
        P_Ka = float(element_properties.df.at[el, "P_kalpha"])
        mu_Ka = float(f_atten_interp(E_Ka))

        if E_edge > 0 and E_Ka > 0 and P_Ka > 0:
            raw_Ka = compute_k_fluorescence(
            energy_array=energy_solar_flare,
            flux_array=flux_scaled,
            photoelectric_array_element=element_properties.full_data_dict[el]["photoelectric_absorption_interp"],
            total_atten_sample_array=sample_mass_atten_array,
            total_atten_sample_at_line=mu_Ka,
            absorption_edge=E_edge,
            theta=theta,
            phi=phi,
            )
            I_Ka = (raw_Ka * w * wK * R * P_Ka) / (4 * np.pi)
        else:
            I_Ka = 0.0
        
        # NOTE: this is already output code
        fluorescence_rows.append({
            "Element": el, "Concentration": w, "Line": "Kα",
            "E_line_keV": E_Ka, "P_line": P_Ka, "Intensity_ph_cm2_s": I_Ka
        })

        # --- Kβ (optional if present)
        E_Kb = float(element_properties.df.at[el, "K-beta Line"])
        P_Kb = float(element_properties.df.at[el, "P_kbeta"])
        if E_Kb > 0.0 and P_Kb > 0.0:
            mu_Kb = float(f_atten_interp(E_Kb))
            raw_Kb = compute_k_fluorescence(
                energy_array=energy_solar_flare,
                flux_array=flux_scaled,
                photoelectric_array_element=element_properties.full_data_dict[el]["photoelectric_absorption_interp"],
                total_atten_sample_array=sample_mass_atten_array,
                total_atten_sample_at_line=mu_Kb,
                absorption_edge=E_edge,
                theta=theta,
                phi=phi,
            )
            I_Kb = (raw_Kb * w * wK * R * P_Kb) / (4 * np.pi)
        else:
            I_Kb = 0.0

        # NOTE: output code
        fluorescence_rows.append({
            "Element": el, "Concentration": w, "Line": "Kβ",
            "E_line_keV": E_Kb if E_Kb > 0 else np.nan,
            "P_line": P_Kb, "Intensity_ph_cm2_s": I_Kb
        })

        return {"rows": fluorescence_rows, "icoh": I_coh}


def compute_coherent_scattering_sample(
    energy_array,
    flux_array,
    sample_mass_atten_array,
    param_dict,
    form_factor_dict,
    full_data_dict,
    conc_df,
    concentration_key,
    scattering_angle,
    theta,
    phi
):
    """
    Computes the corrected coherent (Rayleigh) scattering spectrum as an array.
    
    For each element i, the effective form factor is computed as:
         Re(f) = f₀(x) + f₁(x) + f_rel - Z + f_NT,
         Im(f) = f₂(x),
         f_eff = sqrt[Re(f)² + Im(f)²],
    where x = sin(scattering_angle/2)*E (with E in keV).
    
    The differential cross section per atom is then:
         dσ/dΩ = r_e² * [1 + cos²(scattering_angle)] * [f_eff]².
    Each element's contribution is weighted by:
         w_i * (N_A / A_i),
    and the sample's overall cross section is the sum over elements.
    
    Finally, the coherent scattering intensity is computed as:
         I_coh(E) = [ flux(E) * σ_coh_sample(E) ] / { sample_mass_attenuation(E) * [1 + (cos_theta/cos_phi)] },
    and then multiplied by (solid_angle / (4π)) to obtain the detected intensity.
    
    Returns:
         I_coh as a numpy array (same shape as energy_array).
    """
    # Physical constants
    r_e = 2.818e-13  # cm
    N_A = 6.022e23   # atoms/mol
    ang_factor = 1.0 + np.cos(scattering_angle)**2

    # Compute x = sin(scattering_angle/2)*E, with E in keV.
    x_values = np.sin(scattering_angle / 2.0) * energy_array

    # Initialize the sample's coherent cross section array.
    sigma_coh_sample = np.zeros_like(energy_array)
    
    for el_name in param_dict:
        if (el_name in form_factor_dict) and (el_name in full_data_dict):
            # f₀ from the form factor Excel file:
            f0_interp = form_factor_dict[el_name]
            f0_vals = f0_interp(x_values)
            
            # f₁ and f₂ from the element's text file:
            f1_vals = full_data_dict[el_name]["f1_interp"]
            f2_vals = full_data_dict[el_name]["f2_interp"]
            # Extra parameters:
            f_rel = param_dict[el_name]["f_rel"]
            fNT = param_dict[el_name]["fNT"]
            Z_val = param_dict[el_name]["Atomic Number"]
            
            # Compute corrected real and imaginary parts.
            Re_f = f0_vals + f1_vals + f_rel - Z_val + fNT
            Im_f = f2_vals
            f_eff = np.sqrt(Re_f**2 + Im_f**2)
            
            # Differential cross section per atom:
            dsigma = ((r_e**2)/2.0) * ang_factor * (f_eff**2)
            # Weight by mass fraction and atoms per gram:
            sigma_eff = conc_df[concentration_key][el_name] * (N_A / param_dict[el_name]["Atomic Mass"]) * dsigma
            # param_dict[el_name][concentration_key]
            sigma_coh_sample += sigma_eff
        else:
            print(f"Warning: Missing data for element {el_name}. Skipping.")
    
    # Attenuation denominator:
    geom_factor = np.cos(theta) / np.cos(phi)
    denominator = sample_mass_atten_array * (1.0 + geom_factor)
    
    # Compute the differential intensity.
    I_coh = flux_array * sigma_coh_sample / denominator
    
    return I_coh


def compute_k_fluorescence(
    energy_array,
    flux_array,
    photoelectric_array_element,
    total_atten_sample_array,
    total_atten_sample_at_line,
    absorption_edge,
    theta,
    phi
):
    mask = energy_array >= absorption_edge
    if np.sum(mask) == 0:
        return 0.0
    E_sub    = energy_array[mask]
    flux_sub = flux_array[mask]
    mu_ph    = photoelectric_array_element[mask]
    mu_tot_E = total_atten_sample_array[mask]
    
    
    geometry_factor = np.cos(theta) / np.cos(phi)
    denom = mu_tot_E + (geometry_factor*total_atten_sample_at_line)
    integrand = np.where(denom > 0.0, flux_sub * mu_ph / denom, 0.0)
    # return np.trapz(integrand, x=E_sub)
    return np.trapezoid(integrand, x=E_sub)
