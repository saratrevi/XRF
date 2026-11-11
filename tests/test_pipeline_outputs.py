"""Tests that run_one_grid returns the same detector spectrum it writes to disk.

The notebook and user scripts plot results["spectrum"] instead of rebuilding
it, so it must be exactly the spectrum in the .xlsx file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from xrf_model.config import load_config
from xrf_model.pipeline import run_one_grid

ROOT = Path(__file__).resolve().parents[1]
TUBE_FILE = ROOT / "data" / "flux" / "spectrum_xraytube.txt"


def test_returned_spectrum_is_the_saved_one(tmp_path):
    cfg = load_config(ROOT / "configs" / "lab_xrf.yml")
    local = ("tube", "detector_response", "mca_calibration", "mca_channels")
    cfg = {k: v for k, v in cfg.items() if k not in local}
    cfg.update(src_reference_distance_cm=0.56, grid_n_x=1, grid_n_y=1)
    cfg["paths"] = dict(cfg["paths"], output_dir=str(tmp_path))
    sample = cfg["samples"][0]
    r = run_one_grid(TUBE_FILE, cfg)[sample]
    saved = pd.read_excel(next(Path(r["out_dir"]).glob(f"*_{sample}_spectrum.xlsx")))
    # .xlsx keeps ~15 significant digits, so compare to 1e-12, not bit for bit
    np.testing.assert_allclose(r["spectrum_energy"], saved.Energy_keV.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(r["spectrum"], saved.Intensity_ph_s_per_keV.to_numpy(), rtol=1e-12)


def test_cs_photo_is_computed_once_per_run_not_per_cell(tmp_path, monkeypatch):
    # xraylib.CS_Photo depends only on the element and the energy, so the number
    # of calls must not grow with the number of grid cells
    import xraylib
    calls = {"n": 0}
    original = xraylib.CS_Photo

    def counting(Z, E):
        calls["n"] += 1
        return original(Z, E)

    monkeypatch.setattr(xraylib, "CS_Photo", counting)
    cfg = load_config(ROOT / "configs" / "lab_xrf.yml")
    local = ("tube", "detector_response", "mca_calibration", "mca_channels")
    cfg = {k: v for k, v in cfg.items() if k not in local}
    cfg.update(src_reference_distance_cm=0.56, atmosphere="vacuum")
    counts = {}
    for n in (1, 2):
        calls["n"] = 0
        c = dict(cfg, grid_n_x=n, grid_n_y=n)
        c["paths"] = dict(cfg["paths"], output_dir=str(tmp_path / f"g{n}"))
        run_one_grid(TUBE_FILE, c)
        counts[n] = calls["n"]
    assert counts[2] == counts[1], counts
