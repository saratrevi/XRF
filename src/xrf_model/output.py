# src/xrf_model/output.py
from __future__ import annotations

import os
import numpy as np
from datetime import datetime


def save_mca(
    energy_kev: np.ndarray,
    spectrum: np.ndarray,
    path: str,
    description: str = "",
    live_time: float = 1.0,
) -> None:
    """
    Write the simulated spectrum as an ORTEC PMCA .mca file readable by PyMCA.

    The spectrum (ph/s/keV) is converted to integer counts per channel by:
        counts[i] = max(0, round(spectrum[i] * dE_keV * live_time))

    Energy calibration stored in the file header (eV units, linear):
        E(ch) = A + B * ch
    where A = energy_kev[0] * 1000 [eV] and B = channel width [eV/ch].

    Parameters
    ----------
    energy_kev  : 1D array of photon energies [keV], must be uniformly spaced
    spectrum    : 1D array [ph/s/keV] on energy_kev — output of build_spectrum
    path        : full output file path (should end in .mca)
    description : free-text label written into the DESCRIPTION header field
    live_time   : simulated acquisition time [s] used to scale counts (default 1 s)
    """
    dE_keV = float(np.mean(np.diff(energy_kev)))   # keV per channel
    dE_eV  = dE_keV * 1000.0

    A_eV = float(energy_kev[0]) * 1000.0           # offset  [eV]
    B_eV = dE_eV                                   # gain    [eV/channel]

    # ph/s/keV → counts per channel (integer, non-negative)
    counts = np.maximum(0, np.round(spectrum * dE_keV * live_time)).astype(int)

    n_channels = len(counts)
    timestamp  = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    with open(path, "w") as fh:
        fh.write("<<PMCA SPECTRUM>>\n")
        fh.write("TAG - \n")
        fh.write(f"DESCRIPTION - {description}\n")
        fh.write("GAIN - 1\n")
        fh.write("THRESHOLD - 0\n")
        fh.write("LIVE_MODE - 0\n")
        fh.write("PRESET_TIME - Real\n")
        fh.write(f"LIVE_TIME - {live_time:.1f}\n")
        fh.write(f"REAL_TIME - {live_time:.1f}\n")
        fh.write(f"START_TIME - {timestamp}\n")
        fh.write("SERIAL_NUMBER - \n")
        fh.write("<<CALIBRATION>>\n")
        fh.write("LABEL - Energy\\Channel\n")
        # Format: index  A[eV]  B[eV/ch]  C[eV/ch²]
        fh.write(f"1 {A_eV:.6f} {B_eV:.6f} 0.000000\n")
        fh.write("<<DATA>>\n")
        for c in counts:
            fh.write(f"{c}\n")
        fh.write("<<END>>\n")

    print(f"→ Saved MCA file ({n_channels} channels, dE={dE_eV:.1f} eV/ch, "
          f"live_time={live_time:.1f} s) to {path}")

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