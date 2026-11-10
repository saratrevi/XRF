"""Tests for the SpekPy tube source term and how the pipeline picks its source.

data/flux/spectrum_xraytube.txt was made with SpekPy for this tube at 0.56 cm
and 0.002 mAs, and the pipeline used it as file × 0.56². Generating the
spectrum in the project at 0.002 mA must give the same source term, so the
switch changes no result.
"""
from pathlib import Path

import numpy as np
import pytest

from xrf_model.io import load_flux_data
from xrf_model.pipeline import run_one_grid
from xrf_model.tube import tube_source_term

ROOT = Path(__file__).resolve().parents[1]
TUBE_FILE = ROOT / "data" / "flux" / "spectrum_xraytube.txt"

LAB_TUBE = {
    "kvp": 50,
    "target": "Ag",
    "transmission_thickness_um": 1.0,
    "filters": {"Be": 0.125},
    "energy_step_keV": 0.03,
    "current_mA": 0.002,
}


def test_generated_spectrum_matches_the_old_tube_file():
    energy, P = tube_source_term(LAB_TUBE)
    file_energy, file_flux = load_flux_data(str(TUBE_FILE))

    np.testing.assert_allclose(energy, file_energy, rtol=0, atol=1e-9)
    np.testing.assert_allclose(P, file_flux * 0.56 ** 2, rtol=1e-9)


def test_tube_block_and_flux_file_together_is_an_error():
    with pytest.raises(ValueError, match="tube"):
        run_one_grid(TUBE_FILE, {"tube": LAB_TUBE})


def test_no_source_spectrum_is_an_error():
    with pytest.raises(ValueError, match="No source spectrum"):
        run_one_grid(None, {})
