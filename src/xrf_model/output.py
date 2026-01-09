# src/xrf_model/output.py
from __future__ import annotations

import os
import numpy as np

def save_coherent_spectrum_txt(flare_name, conc, energy_mid, I_diff, out_dir="."):
    """
    energy_mid : 1D array of mid-point energies [keV], length N
    I_diff     : 1D array of differential intensity [photons/cm2/s/keV], length N
    """
    # 1) build N+1 bin edges array
    edges = np.empty(len(energy_mid) + 1, dtype=float)
    # inner edges are midpoints between adjacent energies
    edges[1:-1] = 0.5 * (energy_mid[:-1] + energy_mid[1:])
    # first/last edges by extending the first/last bin width
    edges[0]       = energy_mid[0] - (edges[1] - energy_mid[0])
    edges[-1]      = energy_mid[-1] + (energy_mid[-1] - edges[-2])

    # 2) bin widths in keV
    widths = edges[1:] - edges[:-1]

    # 3) photons/cm2/s per bin
    I_bin = I_diff * widths

    # stack into columns: E_low, E_high, I_bin
    out = np.column_stack((edges[:-1], edges[1:], I_bin))

    # write to txt
    fname = f"{flare_name}_{conc}_coherent_spectrum.txt"
    out_path = os.path.join(out_dir, fname)
    header = "E_low_keV    E_high_keV    I_photons_per_cm2_s"
    np.savetxt(out_path, out, header=header, fmt="%.6e", delimiter="\t")
    print(f"→ Saved coherent spectrum to {out_path}")