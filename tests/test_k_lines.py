"""Tests for the K lines, computed one by one instead of merged per group.

The lines of a group must add up to the old group (same total rate), each
line must sit at its own xraylib energy, and in the spectrum Ba Kα1 and Kα2
(377 eV apart, more than the peak width) must show up as two peaks.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xraylib as xl

from xrf_model.config import load_config
from xrf_model.pipeline import run_one_grid
from xrf_model.spectrum import build_spectrum

ROOT = Path(__file__).resolve().parents[1]
TUBE_FILE = ROOT / "data" / "flux" / "spectrum_xraytube.txt"


@pytest.mark.parametrize("Z", [20, 26, 40, 56])
def test_lines_add_up_to_the_groups(Z):
    from xrf_model.pipeline import _k_group, _k_lines
    lines = {l["name"]: l for l in _k_lines(Z)}
    for prefix, attrs in (("Kα", ["KA1_LINE", "KA2_LINE"]),
                          ("Kβ", ["KB1_LINE", "KB2_LINE", "KB3_LINE"])):
        group = [l for n, l in lines.items() if n.startswith(prefix)]
        E_mean, P_total = _k_group(Z, attrs)
        assert sum(l["P"] for l in group) == pytest.approx(P_total, rel=1e-12)
        assert (sum(l["E"] * l["P"] for l in group) / P_total
                == pytest.approx(E_mean, rel=1e-12))


def test_each_line_has_its_own_energy():
    from xrf_model.pipeline import _k_lines
    lines = {l["name"]: l["E"] for l in _k_lines(40)}
    for name, attr in (("Kα1", xl.KA1_LINE), ("Kα2", xl.KA2_LINE), ("Kβ1", xl.KB1_LINE),
                       ("Kβ3", xl.KB3_LINE), ("Kβ2", xl.KB2_LINE)):
        assert lines[name] == pytest.approx(xl.LineEnergy(40, attr), abs=1e-9)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    cfg = load_config(ROOT / "configs" / "lab_xrf.yml")
    local = ("tube", "detector_response", "mca_calibration", "mca_channels")
    cfg = {k: v for k, v in cfg.items() if k not in local}
    cfg.update(src_reference_distance_cm=0.56, grid_n_x=1, grid_n_y=1, atmosphere="vacuum")
    cfg["paths"] = dict(cfg["paths"], output_dir=str(tmp_path_factory.mktemp("k")))
    return run_one_grid(TUBE_FILE, cfg)[cfg["samples"][0]]["fluorescence"]


def test_ba_k_alpha_shows_two_peaks(table):
    ba = table[(table.Element == "Ba") & table.Line.str.startswith("Kα")]
    assert set(ba.Line) == {"Kα1", "Kα2"}
    grid = 31.0 + 0.005 * np.arange(401)                    # 31 … 33 keV
    S = build_spectrum(grid, np.zeros_like(grid), ba, be_mm=0.0127, si_active_um=450.0,
                       fano=0.118, noise_ev=12.7, w_ev=3.65)
    peaks = [grid[i] for i in range(1, grid.size - 1) if S[i] > S[i - 1] and S[i] >= S[i + 1]]
    assert len(peaks) == 2
    assert peaks[0] == pytest.approx(xl.LineEnergy(56, xl.KA2_LINE), abs=0.01)
    assert peaks[1] == pytest.approx(xl.LineEnergy(56, xl.KA1_LINE), abs=0.01)
