# src/xrf_model/config.py
from __future__ import annotations

from pathlib import Path
import yaml

# This file is <repo>/src/xrf_model/config.py, so the repository is two
# folders up, whatever folder Python is started in.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """The XRF repository folder (the one with data/ and configs/)."""
    if not ((_REPO_ROOT / "data").is_dir() and (_REPO_ROOT / "configs").is_dir()):
        raise RuntimeError(f"Expected the XRF repository at {_REPO_ROOT} "
                           "(with data/ and configs/), but they are not there.")
    return _REPO_ROOT


def repo_path(path: str | Path) -> Path:
    """Absolute paths stay as they are; relative ones are taken from the repository."""
    p = Path(path)
    return p if p.is_absolute() else (repo_root() / p).resolve()


def load_config(yaml_path: str | Path) -> dict:
    """
    Read a YAML config.  The config path and every path inside it are taken
    relative to the repository unless absolute, so the result does not depend
    on the folder Python is started in.
    """
    root = repo_root()
    yaml_path = repo_path(yaml_path)

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    # Lab-geometry keys
    for k in ["src_x", "src_y", "src_z",
              "det_x", "det_y", "det_z",
              "det_area",
              "grid_x_min", "grid_x_max",
              "grid_y_min", "grid_y_max",
              "src_reference_distance_cm",
              "src_vacuum_path_cm", "det_vacuum_path_cm"]:
        if k in cfg:
            cfg[k] = float(cfg[k])

    for k in ["grid_n_x", "grid_n_y", "mca_channels"]:
        if k in cfg:
            cfg[k] = int(cfg[k])

    # det_normal stays as a list — normalised later in GridGeometry

    # Detector response matrix path, relative to the repo root unless absolute
    response = cfg.get("detector_response")
    if response:
        response["file"] = str(repo_path(response["file"]))

    # MCA calibration file (PyMca .calib), relative to the repo root unless absolute
    if cfg.get("mca_calibration"):
        cfg["mca_calibration"] = str(repo_path(cfg["mca_calibration"]))

    # Resolve paths
    paths = cfg.get("paths", {})
    cfg["paths"] = {
        "compositions": str(repo_path(paths["compositions"])),
        "output_dir": str(repo_path(paths["output_dir"])),
    }

    cfg["_repo_root"] = str(root)
    return cfg
