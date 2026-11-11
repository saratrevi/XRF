# src/xrf_model/io.py
from __future__ import annotations

import numpy as np


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
