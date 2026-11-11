"""Tests that paths do not depend on the folder Python is started in.

The repository is found from the package's own location, and every relative
path (the config, the paths inside it, a flux file) is taken from there.  The
old code guessed the repository from the current folder, so it failed from
another folder and could be fooled by any folder with a `data` sub-folder.
"""
from pathlib import Path

from xrf_model.config import load_config, repo_path, repo_root
from xrf_model.pipeline import run_one_grid

ROOT = Path(__file__).resolve().parents[1]


def test_repository_is_found_from_the_package():
    assert repo_root() == ROOT
    assert repo_path("configs/lab_xrf.yml") == ROOT / "configs" / "lab_xrf.yml"
    assert repo_path("/an/absolute/path") == Path("/an/absolute/path")


def test_config_loads_from_any_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("configs/lab_xrf.yml")
    assert Path(cfg["paths"]["compositions"]) == ROOT / "data" / "compositions"


def test_a_stray_data_folder_does_not_fool_it(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    monkeypatch.chdir(tmp_path)
    cfg = load_config("configs/lab_xrf.yml")
    assert Path(cfg["_repo_root"]) == ROOT
    assert Path(cfg["paths"]["output_dir"]).is_relative_to(ROOT)


def test_relative_flux_file_is_found_from_any_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("configs/lab_xrf.yml")
    local = ("tube", "detector_response", "mca_calibration", "mca_channels")
    cfg = {k: v for k, v in cfg.items() if k not in local}
    cfg.update(src_reference_distance_cm=0.56, grid_n_x=1, grid_n_y=1)
    cfg["paths"] = dict(cfg["paths"], output_dir=str(tmp_path / "out"))
    r = run_one_grid(Path("data/flux/spectrum_xraytube.txt"), cfg)
    assert r[cfg["samples"][0]]["fluorescence"].Intensity_ph_s.sum() > 0
