"""Tests for the sample attenuation at line energies.

μ at a line's energy must be xraylib's value at exactly that energy. If it
were read between two beam-grid points, a line just below an absorption edge
(Al Kβ at 1.554 keV, Al K edge at 1.560 keV) would get a value mixed with the
far side of the edge, and the result would change with where the grid points
fall.  So: the same tube spectrum on two grids shifted by half a step must
give the same Al Kβ1/Kα1 ratio (both lines are excited the same way, only
their way out differs).
"""
from pathlib import Path

import numpy as np
import pytest
import xraylib as xl

from xrf_model.config import load_config
from xrf_model.io import load_flux_data
from xrf_model.pipeline import run_one_grid

ROOT = Path(__file__).resolve().parents[1]
TUBE_FILE = ROOT / "data" / "flux" / "spectrum_xraytube.txt"


def test_exact_mu_below_an_edge_does_not_feel_the_edge():
    from xrf_model.physics import mass_attenuation
    params = {"Al": {"Atomic Number": 13}, "O": {"Atomic Number": 8}}
    conc = {"Al": 0.5, "O": 0.5}
    E_kb = xl.LineEnergy(13, xl.KB1_LINE)                  # 1.554 keV, below the Al K edge
    expected = 0.5 * xl.CS_Total(13, E_kb) + 0.5 * xl.CS_Total(8, E_kb)
    assert mass_attenuation(params, conc, E_kb) == pytest.approx(expected, rel=1e-12)
    assert mass_attenuation(params, conc, 0.525) > 0        # below the beam grid
    assert mass_attenuation(params, conc, 93.35) > 0        # above the beam grid


def ratios(tmp_path, flux_file, label):
    cfg = load_config(ROOT / "configs" / "lab_xrf.yml")
    local = ("tube", "detector_response", "mca_calibration", "mca_channels")
    cfg = {k: v for k, v in cfg.items() if k not in local}
    cfg.update(src_reference_distance_cm=0.56, grid_n_x=1, grid_n_y=1, atmosphere="vacuum")
    cfg["paths"] = dict(cfg["paths"], output_dir=str(tmp_path / label))
    f = run_one_grid(flux_file, cfg)[cfg["samples"][0]]["fluorescence"]
    I = lambda el, line: float(f[(f.Element == el) & (f.Line == line)].Intensity_ph_s.iloc[0])
    return {"Al Kβ1/Kα1": I("Al", "Kβ1") / I("Al", "Kα1"),
            "Y Lβ17/Lα1": I("Y", "Lβ17") / I("Y", "Lα1")}


def test_results_do_not_depend_on_where_the_grid_points_fall(tmp_path):
    E, F = load_flux_data(str(TUBE_FILE))
    shifted = E[:-1] + 0.5 * (E[1] - E[0])                  # half a step: 1.040, 1.070, …
    shifted_file = tmp_path / "shifted.txt"
    np.savetxt(shifted_file, np.column_stack([shifted, np.interp(shifted, E, F)]),
               header="Energy_keV  SpectralFluence_photons_cm-2_keV-1")
    a = ratios(tmp_path, TUBE_FILE, "grid")
    b = ratios(tmp_path, shifted_file, "shifted")
    # old (interpolated) μ: +8.5% and −9.7%.  Now: Al < 0.001%; Y 0.16%, because
    # Lβ17 (from L2) and Lα1 (from L3) are also excited slightly differently
    for key in a:
        assert b[key] == pytest.approx(a[key], rel=1e-2), key
