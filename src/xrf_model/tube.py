# src/xrf_model/tube.py
"""X-ray tube source spectrum, generated with SpekPy at run time (and cached)."""
from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import spekpy


def tube_source_term(tube: dict, cache_dir=None) -> tuple[np.ndarray, np.ndarray]:
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

    Cache
    -----
    With cache_dir, the SpekPy spectrum (per mAs, so independent of the
    current) is saved there and reused by later runs with the same settings,
    skipping SpekPy's ~7 s.  The file name shows the settings; a short code at
    the end also covers the SpekPy version, so a new version computes afresh.
    Deleting the folder clears the cache.

    tube keys
    ---------
    kvp                        tube voltage [kV]
    target                     anode material, SpekPy name (e.g. "Ag")
    transmission_thickness_um  transmission target thickness [µm]
    filters                    {material: thickness_mm}, e.g. {"Be": 0.125}
    energy_step_keV            spectrum bin width [keV]
    current_mA                 tube current [mA]
    """
    settings = _spectrum_settings(tube)
    if cache_dir is None:
        energy_kev, per_sr = _spekpy_per_sr(settings)
    else:
        energy_kev, per_sr = _cached_per_sr(settings, Path(cache_dir))
    return energy_kev, per_sr * float(tube["current_mA"])   # ph/s/keV/sr


def _spectrum_settings(tube: dict) -> dict:
    """Everything that changes the SpekPy spectrum shape (not the current)."""
    return {
        "kvp": float(tube["kvp"]),
        "target": str(tube["target"]),
        "transmission_thickness_um": float(tube["transmission_thickness_um"]),
        "filters": {str(m): float(t) for m, t in sorted(tube.get("filters", {}).items())},
        "energy_step_keV": float(tube["energy_step_keV"]),
        "spekpy": version("spekpy"),
    }


def _spekpy_per_sr(settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """SpekPy spectrum in ph/keV/sr per mAs (1 mAs, evaluated at 1 cm)."""
    z_cm = 1.0   # SpekPy evaluation distance; any value gives the same result
    s = spekpy.Spek(
        kvp=settings["kvp"],
        targ=settings["target"],
        trans=True,
        thick=settings["transmission_thickness_um"],
        dk=settings["energy_step_keV"],
        z=z_cm,
        mas=1.0,
    )
    for material, thickness_mm in settings["filters"].items():
        s.filter(material, thickness_mm)
    energy_kev, fluence = s.get_spectrum(edges=False)   # ph/cm²/keV at z_cm, per mAs
    return energy_kev, fluence * z_cm ** 2              # ph/keV/sr, per mAs


def cache_file_name(settings: dict) -> str:
    """Readable file name from the tube settings, plus a short code of all of them."""
    key = json.dumps(settings, sort_keys=True)
    code = hashlib.sha256(key.encode()).hexdigest()[:10]
    filters = "-".join(f"{m}{t:g}mm" for m, t in settings["filters"].items()) or "nofilter"
    return (f"tube_{settings['kvp']:g}kV_{settings['target']}_"
            f"{settings['transmission_thickness_um']:g}um_{filters}_"
            f"{settings['energy_step_keV'] * 1000:g}eV_{code}.npz")


def _cached_per_sr(settings: dict, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    key = json.dumps(settings, sort_keys=True)
    path = cache_dir / cache_file_name(settings)
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if str(f["settings"]) == key:
                print(f"Tube spectrum read from cache: {path.name}")
                return f["energy_keV"], f["per_sr_per_mAs"]
    energy_kev, per_sr = _spekpy_per_sr(settings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")     # complete file, or none at all
    np.savez(tmp, energy_keV=energy_kev, per_sr_per_mAs=per_sr, settings=key)
    os.replace(tmp, path)
    print(f"Tube spectrum computed with SpekPy and cached: {path.name}")
    return energy_kev, per_sr
