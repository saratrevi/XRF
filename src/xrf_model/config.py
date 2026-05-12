# src/xrf_model/config.py
from __future__ import annotations

from pathlib import Path
import yaml

def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by looking for a 'data' folder."""
    p = start or Path.cwd()
    p = p.resolve()
    for _ in range(6):
        if (p / "data").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root (folder containing 'data'). Run from repo or notebooks/.")

def load_config(yaml_path: str | Path) -> dict:
    root = find_repo_root()
    yaml_path = (root / yaml_path).resolve() if not Path(yaml_path).is_absolute() else Path(yaml_path)

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    for k in ["theta", "phi", "altitude", "distance_sun_AU", "footprint"]:
        if k in cfg:
            cfg[k] = float(cfg[k])

    # New lab-geometry keys (optional — only present in lab configs)
    for k in ["src_x", "src_y", "src_z",
              "det_x", "det_y", "det_z",
              "det_area",
              "grid_x_min", "grid_x_max",
              "grid_y_min", "grid_y_max",
              "src_reference_distance_cm"]:
        if k in cfg:
            cfg[k] = float(cfg[k])

    for k in ["grid_n_x", "grid_n_y"]:
        if k in cfg:
            cfg[k] = int(cfg[k])

    # det_normal stays as a list — normalised later in GridGeometry

    # Resolve paths
    paths = cfg.get("paths", {})
    cfg["paths"] = {
        "element_data": str((root / paths["element_data"]).resolve()),
        "form_factors": str((root / paths["form_factors"]).resolve()),
        "absorption_dir": str((root / paths["absorption_dir"]).resolve()),
        "output_dir": str((root / paths["output_dir"]).resolve()),
    }

    cfg["_repo_root"] = str(root)
    return cfg
