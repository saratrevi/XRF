# src/xrf_model/io.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import xraylib
import yaml


def load_composition(path) -> dict:
    """
    Read a sample composition file (YAML): {element symbol: mass fraction}.

    The file has a `composition:` mapping, e.g.
        composition:
          O: 0.437
          Si: 0.221
    Other keys (source, notes) are allowed and ignored.  Elements with a
    fraction of 0 are skipped.  Only the ratios between the fractions matter
    for this model, so the sum does not have to be exactly 1.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Composition file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    comp = data.get("composition")
    if not isinstance(comp, dict) or not comp:
        raise ValueError(f"{path}: needs a non-empty 'composition:' mapping")
    out = {}
    for el, w in comp.items():
        try:
            xraylib.SymbolToAtomicNumber(str(el))
        except ValueError:
            raise ValueError(f"{path}: '{el}' is not an element symbol") from None
        w = float(w)
        if w < 0:
            raise ValueError(f"{path}: {el} has a negative mass fraction ({w})")
        if w > 0:
            out[str(el)] = w
    return out


def load_flux_data(txt_filepath: str):
    """
    Read a flux spectrum: energy [keV] and flux from the first two columns.

    Comment lines (# or !, as in .qdp files), empty lines and non-numeric
    command lines (e.g. "READ SERR 2") are skipped.  Returns
    (energy_array, flux_array) sorted by energy.
    """
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
