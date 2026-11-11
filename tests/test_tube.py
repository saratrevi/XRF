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


def test_removed_luminosity_option_is_an_error():
    # ph_per_s_per_keV used to pass a total output on without dividing by 4π
    with pytest.raises(ValueError, match="flux_units"):
        run_one_grid(TUBE_FILE, {"flux_units": "ph_per_s_per_keV"})


def test_no_source_spectrum_is_an_error():
    with pytest.raises(ValueError, match="No source spectrum"):
        run_one_grid(None, {})


# ── Cache of SpekPy spectra ───────────────────────────────────────────────────
# A stand-in for SpekPy that counts how often it is asked for a spectrum, so
# the caching can be checked in milliseconds (the real spectrum is checked by
# test_generated_spectrum_matches_the_old_tube_file above).

class FakeSpek:
    calls = 0

    def __init__(self, kvp, targ, trans, thick, dk, z, mas):
        FakeSpek.calls += 1
        self.kvp, self.dk, self.attenuation = kvp, dk, 1.0

    def filter(self, material, thickness_mm):
        self.attenuation *= np.exp(-thickness_mm)

    def get_spectrum(self, edges=False):
        E = np.arange(1.0, self.kvp, self.dk)
        return E, self.attenuation * (self.kvp - E)


@pytest.fixture
def fake_spekpy(monkeypatch):
    import xrf_model.tube as tube_module
    FakeSpek.calls = 0
    monkeypatch.setattr(tube_module.spekpy, "Spek", FakeSpek)
    return FakeSpek


def test_cached_spectrum_is_exactly_the_spekpy_one(tmp_path, fake_spekpy):
    E0, P0 = tube_source_term(LAB_TUBE)                       # no cache
    E1, P1 = tube_source_term(LAB_TUBE, cache_dir=tmp_path)   # computed, then saved
    E2, P2 = tube_source_term(LAB_TUBE, cache_dir=tmp_path)   # read from the file
    assert fake_spekpy.calls == 2                              # the third call skipped SpekPy
    for E, P in ((E1, P1), (E2, P2)):
        np.testing.assert_array_equal(E, E0)
        np.testing.assert_array_equal(P, P0)


def test_a_new_current_reuses_the_cached_spectrum(tmp_path, fake_spekpy):
    _, P1 = tube_source_term(LAB_TUBE, cache_dir=tmp_path)
    _, P2 = tube_source_term(dict(LAB_TUBE, current_mA=0.01), cache_dir=tmp_path)
    assert fake_spekpy.calls == 1 and len(list(tmp_path.glob("*.npz"))) == 1
    np.testing.assert_allclose(P2, P1 * 5.0, rtol=1e-12)


def test_every_other_setting_gets_its_own_file(tmp_path, fake_spekpy):
    for tube in (LAB_TUBE, dict(LAB_TUBE, kvp=40), dict(LAB_TUBE, target="Rh"),
                 dict(LAB_TUBE, filters={"Be": 0.25}), dict(LAB_TUBE, energy_step_keV=0.01)):
        tube_source_term(tube, cache_dir=tmp_path)
    assert fake_spekpy.calls == 5 and len(list(tmp_path.glob("*.npz"))) == 5


def test_cache_file_name_shows_the_tube_settings(tmp_path, fake_spekpy):
    tube_source_term(LAB_TUBE, cache_dir=tmp_path)
    (name,) = [p.name for p in tmp_path.glob("*.npz")]
    assert name.startswith("tube_50kV_Ag_1um_Be0.125mm_30eV_")
