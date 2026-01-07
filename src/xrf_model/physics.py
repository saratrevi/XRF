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
    # <-- paste your function exactly
    ...

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
    # <-- paste your function exactly
    ...
