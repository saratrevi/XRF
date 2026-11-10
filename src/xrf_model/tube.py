# src/xrf_model/tube.py
"""X-ray tube source spectrum, generated with SpekPy at run time."""
from __future__ import annotations

import numpy as np
import spekpy


def tube_source_term(tube: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    On-axis source term of a transmission-target X-ray tube.

    Returns (energy_keV, P) with P in ph/s/keV/sr, so the flux reaching a
    surface element at distance R and incidence angle Φ is P · sinΦ / R².

    SpekPy gives the fluence in ph/cm²/keV for a chosen mAs at a distance z
    from the focal spot, along the beam axis (this z is not the height of the
    tube above the sample). The fluence falls as 1/z², so fluence × z² is the
    same at any z and is the number of photons per sr. Multiplying by the
    current in mA (mA·s per second) gives ph/s/keV/sr.

    The same on-axis spectrum is used for every grid point, with the tube
    assumed aimed at the grid centre. SpekPy can compute off-axis spectra, but
    on the lab_xrf.yml grid (at most 6.3° off axis) per-point spectra changed
    Si Kα by 0.16% and Fe Kα by 0.04%, and each direction needs its own SpekPy
    run of about 10 s (checked 2026-11-10).

    tube keys
    ---------
    kvp                        tube voltage [kV]
    target                     anode material, SpekPy name (e.g. "Ag")
    transmission_thickness_um  transmission target thickness [µm]
    filters                    {material: thickness_mm}, e.g. {"Be": 0.125}
    energy_step_keV            spectrum bin width [keV]
    current_mA                 tube current [mA]
    """
    z_cm = 1.0   # SpekPy evaluation distance; any value gives the same result
    s = spekpy.Spek(
        kvp=float(tube["kvp"]),
        targ=str(tube["target"]),
        trans=True,
        thick=float(tube["transmission_thickness_um"]),
        dk=float(tube["energy_step_keV"]),
        z=z_cm,
        mas=1.0,
    )
    for material, thickness_mm in tube.get("filters", {}).items():
        s.filter(str(material), float(thickness_mm))
    energy_kev, fluence = s.get_spectrum(edges=False)   # ph/cm²/keV at z_cm, per mAs
    per_sr = fluence * z_cm ** 2                         # ph/keV/sr, per mAs
    return energy_kev, per_sr * float(tube["current_mA"])   # ph/s/keV/sr
