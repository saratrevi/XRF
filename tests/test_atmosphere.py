"""Tests for air absorption between the tube, the sample and the detector.

A single grid cell at the origin makes both path lengths known, so the effect
of air on each output can be checked exactly against exp(-mu_air · L).
"""
from pathlib import Path

import numpy as np
import pytest

from xrf_model.atmosphere import attenuation_coefficient
from xrf_model.config import load_config
from xrf_model.pipeline import run_one_grid

ROOT = Path(__file__).resolve().parents[1]
TUBE_FILE = ROOT / "data" / "flux" / "spectrum_xraytube.txt"


def test_air_matches_the_nist_table():
    # NIST (Hubbell & Seltzer) mu/rho for dry air [cm²/g], density 1.205e-3 g/cm³
    mu = attenuation_coefficient("air")
    for energy, mu_rho in ((5.0, 40.27), (10.0, 5.120), (20.0, 0.7779)):
        assert float(mu(energy)) == pytest.approx(mu_rho * 1.205e-3, rel=1e-3)


def test_vacuum_absorbs_nothing():
    assert np.all(attenuation_coefficient("vacuum")(np.array([1.0, 5.0, 20.0])) == 0.0)


def test_unknown_atmosphere_is_an_error():
    with pytest.raises(ValueError, match="atmosphere"):
        attenuation_coefficient("helium")


@pytest.fixture(scope="module")
def one_cell(tmp_path_factory):
    cfg = load_config(ROOT / "configs" / "lab_xrf.yml")
    # no tube (file route), no local response matrix or MCA calibration:
    # tests use repo files only
    local = ("tube", "detector_response", "mca_calibration", "mca_channels")
    base = {k: v for k, v in cfg.items() if k not in local}
    base.update(src_reference_distance_cm=0.56, grid_n_x=1, grid_n_y=1,
                atmosphere="vacuum", src_vacuum_path_cm=0.0, det_vacuum_path_cm=0.0)
    base["paths"] = dict(cfg["paths"], output_dir=str(tmp_path_factory.mktemp("air")))
    sample = cfg["concentration_cols"][0]

    def run(**changes):
        return run_one_grid(TUBE_FILE, dict(base, **changes))[sample]

    big = 100.0   # longer than any path: no air on that side
    return {
        "R_src": float(np.linalg.norm([cfg["src_x"], cfg["src_y"], cfg["src_z"]])),
        "R_det": float(np.linalg.norm([cfg["det_x"], cfg["det_y"], cfg["det_z"]])),
        "vacuum": run(),
        "out_only": run(atmosphere="air", src_vacuum_path_cm=big),
        "in_only": run(atmosphere="air", det_vacuum_path_cm=big),
        "both_in_housings": run(atmosphere="air", src_vacuum_path_cm=big,
                                det_vacuum_path_cm=big),
    }


def test_air_to_the_detector_scales_each_line_at_its_own_energy(one_cell):
    mu = attenuation_coefficient("air")
    f0 = one_cell["vacuum"]["fluorescence"]
    f1 = one_cell["out_only"]["fluorescence"]
    m = f0.Intensity_ph_s.values > 0
    expected = np.exp(-mu(f0.E_line_keV.values[m]) * one_cell["R_det"])
    np.testing.assert_allclose(
        f1.Intensity_ph_s.values[m] / f0.Intensity_ph_s.values[m], expected, rtol=1e-9)


def test_air_to_the_detector_scales_the_scattered_continua(one_cell):
    mu = attenuation_coefficient("air")
    E = one_cell["vacuum"]["energy"]
    expected = np.exp(-mu(E) * one_cell["R_det"])
    for key in ("I_coherent", "I_incoherent"):
        v0, v1 = one_cell["vacuum"][key], one_cell["out_only"][key]
        m = v0 > 0
        np.testing.assert_allclose(v1[m] / v0[m], expected[m], rtol=1e-9)


def test_air_from_the_tube_scales_the_incoming_spectrum(one_cell):
    # Rayleigh keeps the photon energy, so it shows the incoming loss bin by bin
    mu = attenuation_coefficient("air")
    E = one_cell["vacuum"]["energy"]
    v0, v1 = one_cell["vacuum"]["I_coherent"], one_cell["in_only"]["I_coherent"]
    m = v0 > 0
    np.testing.assert_allclose(v1[m] / v0[m], np.exp(-mu(E[m]) * one_cell["R_src"]),
                               rtol=1e-9)
    f0 = one_cell["vacuum"]["fluorescence"].Intensity_ph_s.values
    f1 = one_cell["in_only"]["fluorescence"].Intensity_ph_s.values
    assert np.all(f1[f0 > 0] < f0[f0 > 0])


def test_paths_inside_the_housings_are_not_in_air(one_cell):
    f0 = one_cell["vacuum"]["fluorescence"].Intensity_ph_s.values
    f1 = one_cell["both_in_housings"]["fluorescence"].Intensity_ph_s.values
    np.testing.assert_array_equal(f1, f0)
