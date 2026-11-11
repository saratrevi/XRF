"""Tests for the per-sample composition files (data/compositions/<sample>.yml).

Each file must load into {element: mass fraction}; mistakes (a wrong symbol,
a negative value, a missing `composition:` key) must stop with a clear error;
and the order in which a file lists the elements must not change the results.
"""
from pathlib import Path

import pytest
import yaml

from xrf_model.config import load_config
from xrf_model.io import load_composition
from xrf_model.pipeline import run_one_grid

ROOT = Path(__file__).resolve().parents[1]
COMPOSITIONS = ROOT / "data" / "compositions"
TUBE_FILE = ROOT / "data" / "flux" / "spectrum_xraytube.txt"


def write(tmp_path, comp, name="s"):
    path = tmp_path / f"{name}.yml"
    path.write_text(yaml.safe_dump({"composition": comp}, sort_keys=False))
    return path


def test_every_shipped_composition_loads():
    files = sorted(COMPOSITIONS.glob("*.yml"))
    assert len(files) >= 12
    for f in files:
        comp = load_composition(f)
        assert comp and all(w > 0 for w in comp.values()), f.name
    soil = load_composition(COMPOSITIONS / "soil_14163.yml")
    assert len(soil) == 22 and soil["O"] == pytest.approx(0.4368, abs=1e-4)


def test_zero_fractions_are_skipped(tmp_path):
    assert load_composition(write(tmp_path, {"O": 0.5, "Si": 0.5, "Fe": 0.0})) == {"O": 0.5, "Si": 0.5}


@pytest.mark.parametrize("comp, message", [
    ({"O": 0.5, "Xx": 0.5}, "not an element symbol"),
    ({"O": 0.5, "Si": -0.1}, "negative"),
])
def test_mistakes_stop_with_a_clear_error(tmp_path, comp, message):
    with pytest.raises(ValueError, match=message):
        load_composition(write(tmp_path, comp))


def test_missing_composition_key_is_an_error(tmp_path):
    path = tmp_path / "s.yml"
    path.write_text("O: 0.5\nSi: 0.5\n")
    with pytest.raises(ValueError, match="composition"):
        load_composition(path)


def test_element_order_in_the_file_does_not_change_results(tmp_path):
    soil = load_composition(COMPOSITIONS / "soil_14163.yml")
    write(tmp_path, soil, "as_shipped")
    write(tmp_path, dict(reversed(list(soil.items()))), "reversed")
    cfg = load_config(ROOT / "configs" / "lab_xrf.yml")
    local = ("tube", "detector_response", "mca_calibration", "mca_channels")
    cfg = {k: v for k, v in cfg.items() if k not in local}
    cfg.update(src_reference_distance_cm=0.56, grid_n_x=1, grid_n_y=1,
               samples=["as_shipped", "reversed"])
    cfg["paths"] = dict(cfg["paths"], compositions=str(tmp_path), output_dir=str(tmp_path / "out"))
    r = run_one_grid(TUBE_FILE, cfg)
    a, b = r["as_shipped"]["fluorescence"], r["reversed"]["fluorescence"]
    assert list(a.Intensity_ph_s) == list(b.Intensity_ph_s)
    assert (r["as_shipped"]["I_incoherent"] == r["reversed"]["I_incoherent"]).all()
