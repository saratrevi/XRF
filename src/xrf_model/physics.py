# src/xrf_model/physics.py
from __future__ import annotations

import numpy as np

def compute_coherent_scattering_sample(
    energy_array,
    flux_array,
    sample_mass_atten_array,
    param_dict,
    form_factor_dict,
    full_data_dict,
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
            sigma_eff = param_dict[el_name][concentration_key] * (N_A / param_dict[el_name]["Atomic Mass"]) * dsigma
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
    return np.trapz(integrand, x=E_sub)