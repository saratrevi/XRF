# src/xrf_model/energy_grid.py
"""Energy-bin helpers shared by the Compton redistribution and the .mca output."""
from __future__ import annotations

import numpy as np


def bin_edges(centres: np.ndarray) -> np.ndarray:
    """Edges halfway between bin centres, ends extended by half a bin."""
    mid = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[2 * centres[0] - mid[0]], mid, [2 * centres[-1] - mid[-1]]])


def is_uniform(centres: np.ndarray, rtol: float = 1e-6) -> bool:
    """True if all bins have the same width."""
    steps = np.diff(centres)
    return bool(np.allclose(steps, steps[0], rtol=rtol, atol=0.0))


def rebin(centres: np.ndarray, density: np.ndarray, new_edges: np.ndarray) -> np.ndarray:
    """
    Counts in each new bin from a density [per keV] given at bin centres.

    The density is taken as constant inside each input bin, so the counts are
    exact for that picture and their total is kept.  New bins outside the input
    range get nothing.
    """
    edges = bin_edges(centres)
    cumulative = np.concatenate([[0.0], np.cumsum(density * np.diff(edges))])
    return np.diff(np.interp(new_edges, edges, cumulative))
