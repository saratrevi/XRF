# src/xrf_model/io.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def load_flux_data(txt_filepath: str):
# Build a filtered line iterator for np.loadtxt
    def _filtered_lines(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # Skip QDP/comment lines
                if s[0] in ("!", "#"):
                    continue
                # Skip non-numeric command lines (e.g., "READ SERR 2")
                first = s.split()[0]
                try:
                    float(first)
                except ValueError:
                    continue
                yield line

    data = np.loadtxt(_filtered_lines(txt_filepath), usecols=(0, 1))
    # Ensure 2D when there's only one data row
    if data.ndim == 1:
        data = data[None, :]

    energy_array = data[:, 0]
    flux_array   = data[:, 1]

    # Sort by energy if needed
    sort_indices = np.argsort(energy_array)
    return energy_array[sort_indices], flux_array[sort_indices]

def load_element_mass_absorption_data(txt_filepath: str):
# Use usecols=(0,1,2,3,5) to skip the 4th column
    data = np.loadtxt(txt_filepath, comments='#', usecols=(0,1,2,3,5))

    # data now has shape (N, 5)
    #  - data[:, 0] = Energy
    #  - data[:, 1] = f1
    #  - data[:, 2] = f2
    #  - data[:, 3] = Photoelectric mass absorption
    #  - data[:, 4] = Total mass attenuation

    energies = data[:, 0]
    f1       = data[:, 1]
    f2       = data[:, 2]
    photoelectric_absorption = data[:, 3]
    total_mass_attenuation   = data[:, 4]

    # Sort by ascending energy
    idx_sort = np.argsort(energies)

    energies           = energies[idx_sort]
    f1                 = f1[idx_sort]
    f2                 = f2[idx_sort]
    photoelectric_absorption     = photoelectric_absorption[idx_sort]
    total_mass_attenuation   = total_mass_attenuation[idx_sort]

    return {
        "energy": energies,
        "f1": f1,
        "f2": f2,
        "photoelectric_absorption": photoelectric_absorption,
        "total_mass_attenuation": total_mass_attenuation
    }

    
def interpolate_element_full_data(element_dict, common_energy):
    """
    Interpolates an element's data onto a common energy grid.
    
    Assumes element_dict contains the following keys:
       "energy" : np.array of energies (keV)
       "photoelectric_absorption_interp"      : np.array of photoelectric mass absorption coefficients
       "total_mass_attenuation_interp"        : np.array of total mass attenuation coefficients
       "f1" : np.array of f1 values (non-relativistic anomalous dispersion correction)
       "f2" : np.array of f2 values (imaginary correction)
    
    Returns a dictionary with keys:
       "photoelectric" : interpolated photoelectric mass absorption (same length as common_energy)
       "total_attenuation" : interpolated total mass attenuation (same length as common_energy)
       "f1" : interpolated f1 values
       "f2" : interpolated f2 values
    """
    e_orig = element_dict["energy"]
    
    f_photo_interp = interp1d(e_orig, element_dict["photoelectric_absorption"], kind='linear', bounds_error=False ,fill_value="extrapolate" )
    f_total_interp = interp1d(e_orig, element_dict["total_mass_attenuation"], kind='linear',bounds_error=False ,fill_value="extrapolate")
    f_f1_interp = interp1d(e_orig, element_dict["f1"], kind='linear', bounds_error=False ,fill_value="extrapolate")
    f_f2_interp = interp1d(e_orig, element_dict["f2"], kind='linear', bounds_error=False ,fill_value="extrapolate")
    
    photo_interp = f_photo_interp(common_energy)
    total_interp = f_total_interp(common_energy)
    f1_interp = f_f1_interp(common_energy)
    f2_interp = f_f2_interp(common_energy)
    
    return {
        "photoelectric_absorption_interp": photo_interp,
        "total_mass_attenuation_interp": total_interp,
        "f1_interp": f1_interp,
        "f2_interp": f2_interp
    }


def load_form_factor_data(form_factor_filepath: str):
    """
    Loads atomic form factor data from an Excel file.
    In this version, the second column (index=1) is assumed to be the
    x-values (e.g., sin(theta/2)*E points), and subsequent columns
    contain the form-factor data for each element.

    The Excel file might look like:
      | (some unused col) | sin(theta/2)*E | Element1 | Element2 | ...
      |        ...        |       ...      |   ...    |   ...    | ...

    Returns a dictionary mapping each element column name to an
    interpolation function f(x).
    """

    # Read the entire Excel file
    df_ff = pd.read_excel(form_factor_filepath)

    # Define x-values from the SECOND column (index=1)
    x_values = df_ff.iloc[:, 1].values

    form_factor_dict = {}

    # Loop over the remaining columns (starting from index=2) 
    for col in df_ff.columns[2:]:
        y_values = df_ff[col].values
        # Create an interpolation function for each element’s form factor
        f_interp = interp1d(x_values, y_values, kind='linear',bounds_error=False ,fill_value="extrapolate")
        form_factor_dict[col] = f_interp

    return form_factor_dict