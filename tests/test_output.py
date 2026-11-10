"""Tests for the .mca export.

The counts in each channel must be the spectrum integrated over that channel's
energy range, whatever grid the spectrum comes on, and the calibration in the
header must say where each channel really is.
"""
from pathlib import Path

import numpy as np
import pytest

from xrf_model.io import load_flux_data
from xrf_model.output import channel_energies, read_calibration, save_mca
from xrf_model.pipeline import run_one_grid

ROOT = Path(__file__).resolve().parents[1]
CALIB = (0.01, 0.005, 2e-9)          # made-up second-order calibration [keV, keV/ch, keV/ch²]


def read_mca(path):
    text = Path(path).read_text().splitlines()
    cal = text[text.index("<<CALIBRATION>>") + 2].split()
    data = text[text.index("<<DATA>>") + 1:text.index("<<END>>")]
    A, B, C = (float(v) / 1000.0 for v in cal[1:4])          # eV → keV
    return (A, B, C), np.array([int(v) for v in data])


def gaussian(E, centre, sigma, area):
    return area * np.exp(-0.5 * ((E - centre) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def write_calib(path, order=2):
    lines = ["", "[cal_test_|_mca 1_|_1_|_1_|_1]", f"order = {order}",
             f"A = {CALIB[0]}", f"B = {CALIB[1]}", f"C = {CALIB[2]}"]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_equal_steps_give_the_same_file_as_before(tmp_path):
    E = 1.025 + 0.03 * np.arange(200)
    S = 100.0 * np.random.default_rng(1).random(200)
    save_mca(E, S, str(tmp_path / "a.mca"), live_time=10.0)
    cal, counts = read_mca(tmp_path / "a.mca")
    assert cal == pytest.approx((1.025, 0.03, 0.0))
    np.testing.assert_array_equal(counts, np.round(S * 0.03 * 10.0).astype(int))


def test_unequal_steps_are_rebinned_to_equal_channels(tmp_path):
    E, _ = load_flux_data(str(ROOT / "data" / "flux" / "A3_1000bin.qdp"))   # 2.3–23 eV steps
    S = 1.0 + gaussian(E, 3.166, 0.03, 50.0)
    live = 1e6
    save_mca(E, S, str(tmp_path / "q.mca"), live_time=live)
    (A, B, C), counts = read_mca(tmp_path / "q.mca")
    assert B == pytest.approx(np.min(np.diff(E)), rel=1e-6) and C == 0.0
    edges = np.concatenate([[1.5 * E[0] - 0.5 * E[1]], 0.5 * (E[1:] + E[:-1]),
                            [1.5 * E[-1] - 0.5 * E[-2]]])
    expected_total = np.sum(S * np.diff(edges)) * live
    assert counts.sum() == pytest.approx(expected_total, rel=1e-3)
    # the input steps near 3.17 keV are wider than the new channels, so the peak
    # has a flat top: check its centroid above the flat 1 ph/s/keV background
    energy = A + B * np.arange(counts.size)
    near = np.abs(energy - 3.166) < 0.2
    signal = counts[near] - 1.0 * B * live
    centroid = np.sum(energy[near] * signal) / signal.sum()
    assert centroid == pytest.approx(3.166, abs=1e-3)        # was 5.499 keV before the fix


def test_line_lands_in_the_calibrated_channel(tmp_path):
    E = 0.0025 + 0.005 * np.arange(9000)                     # 5 eV grid, 0 … 45 keV
    S = gaussian(E, 6.40, 0.06, 1000.0)
    n = 8192
    save_mca(E, S, str(tmp_path / "c.mca"), live_time=1000.0, calibration=CALIB, n_channels=n)
    cal, counts = read_mca(tmp_path / "c.mca")
    assert cal == pytest.approx(CALIB, rel=1e-6)
    assert counts.size == n
    peak_ch = int(np.argmax(counts))
    assert peak_ch == int(np.argmin(np.abs(channel_energies(CALIB, n) - 6.40)))
    centroid = np.sum(channel_energies(CALIB, n) * counts) / counts.sum()
    assert centroid == pytest.approx(6.40, abs=1e-3)
    assert counts.sum() == pytest.approx(1000.0 * 1000.0, rel=1e-3)


def test_reads_pymca_calibration_files(tmp_path):
    assert read_calibration(write_calib(tmp_path / "two.calib")) == pytest.approx(CALIB)
    A, B, C = read_calibration(write_calib(tmp_path / "one.calib", order=1))
    assert (A, B, C) == pytest.approx((CALIB[0], CALIB[1], 0.0))


def test_calibration_needs_the_number_of_channels(tmp_path):
    with pytest.raises(ValueError, match="mca_channels"):
        run_one_grid(None, {"mca_calibration": str(write_calib(tmp_path / "x.calib"))})
