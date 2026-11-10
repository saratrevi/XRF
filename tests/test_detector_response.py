"""Tests for the detector spectrum built through a Monte Carlo response matrix.

Synthetic matrices are used, so the right answer is known: a "delta" matrix
puts every photon in the channel of its own energy, so after broadening it
must give the same result as the Gaussian path.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xrf_model.config import load_config
from xrf_model.detector_response import (
    broaden, build_spectrum_with_response, fold, load_response,
)
from xrf_model.pipeline import run_one_grid
from xrf_model.spectrum import build_spectrum, qe_active_si, transmission_be

ROOT = Path(__file__).resolve().parents[1]
TUBE_FILE = ROOT / "data" / "flux" / "spectrum_xraytube.txt"

WIDTH = 0.005                                  # keV per channel
E_IN = 1.0 + 0.1 * np.arange(141)              # 1.0 … 15.0 keV, on channel centres
FANO, W_EV, NOISE_EV = 0.118, 3.65, 12.7
DETECTOR = dict(be_mm=0.125, si_active_um=450.0, fano=FANO, noise_ev=NOISE_EV, w_ev=W_EV)


def write_matrix(path, n_channels=4000, escape=0.0, scale=1.0):
    """Delta response (optionally with a Si escape peak), saved in the .npz format."""
    edges = -WIDTH / 2 + WIDTH * np.arange(n_channels + 1)
    resp = np.zeros((E_IN.size, n_channels))
    for i, e in enumerate(E_IN):
        resp[i, int(round(e / WIDTH))] = scale * (1.0 - escape)
        if escape and e - 1.74 > 0:
            resp[i, int(round((e - 1.74) / WIDTH))] = scale * escape
    np.savez(path, energy_in_keV=E_IN, channel_edges_keV=edges, response=resp)
    return path


@pytest.fixture
def delta(tmp_path):
    return load_response(write_matrix(tmp_path / "delta.npz"))


def centroid(response, counts):
    c = response["channel_centres_keV"]
    return float((c * counts).sum() / counts.sum())


def test_line_between_rows_lands_at_its_own_energy(delta):
    counts = fold(delta, [6.4237], [1.0])
    assert counts.sum() == pytest.approx(1.0, abs=1e-12)
    assert centroid(delta, counts) == pytest.approx(6.4237, abs=1e-9)
    assert np.count_nonzero(counts) <= 2      # not widened by mixing two rows


def test_broadening_follows_the_two_term_formula(delta):
    c = delta["channel_centres_keV"]
    out = broaden(c, fold(delta, [6.4], [1.0]), FANO, NOISE_EV / 1000, W_EV / 1000)
    mean = centroid(delta, out)
    variance = float(((c - mean) ** 2 * out).sum() / out.sum())
    expected = FANO * W_EV / 1000 * 6.4 + (NOISE_EV / 1000) ** 2
    assert out.sum() == pytest.approx(1.0, rel=1e-12)
    assert variance == pytest.approx(expected, rel=1e-2)


def inputs():
    energy = 1.025 + 0.03 * np.arange(466)                     # 1.025 … 14.975 keV
    continuum = 100.0 * np.exp(-energy / 4.0)                   # ph/s/keV
    lines = pd.DataFrame({"E_line_keV": [3.69, 6.40, np.nan],
                          "Intensity_ph_s": [50.0, 200.0, 7.0]})
    return energy, continuum, lines


def test_counts_are_conserved_per_detected_photon(delta):
    energy, continuum, lines = inputs()
    E, S = build_spectrum_with_response(energy, continuum, lines, delta,
                                        rows="per_detected_photon", **DETECTOR)
    eff = lambda e: (transmission_be(np.atleast_1d(e), 0.125)
                     * qe_active_si(np.atleast_1d(e), 450.0, include_compton=True))
    expected = (np.sum(continuum * np.gradient(energy) * eff(energy))
                + 50.0 * eff(3.69)[0] + 200.0 * eff(6.40)[0])
    assert np.sum(S) * WIDTH == pytest.approx(expected, rel=1e-9)


def test_per_incident_photon_rows_skip_window_and_efficiency(tmp_path):
    resp = load_response(write_matrix(tmp_path / "eff.npz", scale=0.25))
    energy, continuum, lines = inputs()
    _, S = build_spectrum_with_response(energy, continuum, lines, resp,
                                        rows="per_incident_photon", **DETECTOR)
    expected = 0.25 * (np.sum(continuum * np.gradient(energy)) + 250.0)
    assert np.sum(S) * WIDTH == pytest.approx(expected, rel=1e-9)


def test_escape_peak_moves_with_the_line(tmp_path):
    resp = load_response(write_matrix(tmp_path / "esc.npz", escape=0.01))
    counts = fold(resp, [6.4237], [1.0])
    low = resp["channel_centres_keV"] < 5.5
    assert counts[low].sum() == pytest.approx(0.01, rel=1e-9)
    assert centroid(resp, np.where(low, counts, 0.0)) == pytest.approx(6.4237 - 1.74, abs=1e-9)


def test_delta_matrix_matches_the_gaussian_path_for_the_continuum(delta):
    # below 12 keV the two efficiencies (photoelectric, photoelectric + Compton)
    # agree to better than 0.2%, so the two paths must give the same continuum
    energy, continuum, _ = inputs()
    no_lines = pd.DataFrame({"E_line_keV": [], "Intensity_ph_s": []})
    E, S = build_spectrum_with_response(energy, continuum, no_lines, delta,
                                        rows="per_detected_photon", **DETECTOR)
    gauss = build_spectrum(energy, continuum, no_lines, **DETECTOR)
    band = (E > 3.0) & (E < 12.0)
    np.testing.assert_allclose(S[band], np.interp(E[band], energy, gauss), rtol=5e-3)


def test_bad_matrix_files_are_rejected(tmp_path):
    e = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="same width"):
        np.savez(tmp_path / "a.npz", energy_in_keV=e,
                 channel_edges_keV=np.array([0.0, 1.0, 3.0]), response=np.ones((2, 2)))
        load_response(tmp_path / "a.npz")
    with pytest.raises(ValueError, match="shape"):
        np.savez(tmp_path / "b.npz", energy_in_keV=e,
                 channel_edges_keV=np.array([0.0, 1.0, 2.0]), response=np.ones((3, 2)))
        load_response(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="missing"):
        np.savez(tmp_path / "c.npz", energy_in_keV=e, response=np.ones((2, 2)))
        load_response(tmp_path / "c.npz")


def test_unknown_rows_option_is_an_error(delta):
    energy, continuum, lines = inputs()
    with pytest.raises(ValueError, match="rows"):
        build_spectrum_with_response(energy, continuum, lines, delta,
                                     rows="normalised", **DETECTOR)


def test_old_noise_key_is_an_error():
    with pytest.raises(ValueError, match="detector_noise_eV"):
        run_one_grid(None, {"detector_enc_electrons": 8.0})


def test_pipeline_writes_the_spectrum_on_the_matrix_channels(tmp_path):
    # channels up to 55 keV, so the 50 kV tube spectrum and its broadening fit
    matrix = write_matrix(tmp_path / "pipe.npz", n_channels=11000)
    cfg = load_config(ROOT / "configs" / "lab_xrf.yml")
    cfg = {k: v for k, v in cfg.items() if k not in ("tube", "mca_calibration", "mca_channels")}
    cfg.update(src_reference_distance_cm=0.56, grid_n_x=1, grid_n_y=1,
               detector_response={"file": str(matrix), "rows": "per_detected_photon"})
    cfg["paths"] = dict(cfg["paths"], output_dir=str(tmp_path / "out"))
    sample = cfg["concentration_cols"][0]
    r = run_one_grid(TUBE_FILE, cfg)[sample]

    spec = pd.read_excel(next(Path(r["out_dir"]).glob(f"*_{sample}_spectrum.xlsx")))
    assert len(spec) == 11000
    assert np.allclose(np.diff(spec.Energy_keV), WIDTH)

    E = r["energy"]
    eff = lambda e: (transmission_be(e, cfg["be_window_mm"])
                     * qe_active_si(e, cfg["si_active_layer_um"], include_compton=True))
    f = r["fluorescence"]
    f = f[np.isfinite(f.E_line_keV) & (f.Intensity_ph_s > 0)]
    expected = (np.sum((r["I_coherent"] + r["I_incoherent"]) * np.gradient(E) * eff(E))
                + np.sum(f.Intensity_ph_s.values * eff(f.E_line_keV.values)))
    assert spec.Intensity_ph_s_per_keV.sum() * WIDTH == pytest.approx(expected, rel=1e-6)
