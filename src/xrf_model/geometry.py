# src/xrf_model/geometry.py
from __future__ import annotations
import numpy as np


class GridGeometry:
    """
    2D grid of sample surface points with per-point geometry.

    Coordinate system
    -----------------
    Sample surface = Z = 0 plane.  Z axis points upward (away from sample).
    All distances in cm.

    Angle convention (Shiraiwa-Fujino)
    -----------------------------------
    sinΦ = z_src / R_src  (incidence: elevation angle from sample surface)
    sinΨ = z_det / R_det  (emission:  elevation angle from sample surface)
    """

    def __init__(self, cfg: dict):
        self.r_src = np.array([cfg["src_x"], cfg["src_y"], cfg["src_z"]], dtype=float)
        self.r_det = np.array([cfg["det_x"], cfg["det_y"], cfg["det_z"]], dtype=float)

        n = np.array(cfg["det_normal"], dtype=float)
        self.n_det = n / np.linalg.norm(n)

        self.det_area = float(cfg["det_area"])
        self.r_det_radius = np.sqrt(self.det_area / np.pi)   # equivalent circular radius (cm)

        # Build 2D grid over sample surface.
        # For n=1: place the single point at the grid centre and assign the full
        # range as dS — linspace with n=1 returns x_min (a corner), which is
        # geometrically wrong for a symmetric instrument.
        if cfg["grid_n_x"] > 1:
            xs = np.linspace(cfg["grid_x_min"], cfg["grid_x_max"], cfg["grid_n_x"])
            dx = xs[1] - xs[0]
        else:
            xs = np.array([(cfg["grid_x_min"] + cfg["grid_x_max"]) / 2.0])
            dx = cfg["grid_x_max"] - cfg["grid_x_min"]

        if cfg["grid_n_y"] > 1:
            ys = np.linspace(cfg["grid_y_min"], cfg["grid_y_max"], cfg["grid_n_y"])
            dy = ys[1] - ys[0]
        else:
            ys = np.array([(cfg["grid_y_min"] + cfg["grid_y_max"]) / 2.0])
            dy = cfg["grid_y_max"] - cfg["grid_y_min"]

        self.XX, self.YY = np.meshgrid(xs, ys)   # shape (n_y, n_x)
        self.dS = dx * dy                         # cm² per cell

    def compute_local_geometry(self):
        """
        Returns per-grid-point arrays (shape n_y × n_x):

        sinPhi    : sinΦ = z_src/R_src  (incidence angle from surface normal)
        sinPsi    : sinΨ = z_det/R_det  (emission angle from surface normal)
        R_src     : source-to-point distance (cm)
        R_det     : point-to-detector distance (cm)
        Omega     : exact solid angle of detector (sr), using 2π(1−cosα)
        valid     : bool — True where point can be seen by both source and detector
        """
        # Source vectors (z_src is scalar, same for all grid points)
        dx_s = self.r_src[0] - self.XX
        dy_s = self.r_src[1] - self.YY
        dz_s = self.r_src[2]              # z_src > 0 means source above surface
        R_src = np.sqrt(dx_s**2 + dy_s**2 + dz_s**2)
        sinPhi = dz_s / R_src

        # Detector vectors
        dx_d = self.r_det[0] - self.XX
        dy_d = self.r_det[1] - self.YY
        dz_d = self.r_det[2]
        R_det = np.sqrt(dx_d**2 + dy_d**2 + dz_d**2)
        sinPsi = dz_d / R_det

        # Tilt correction: det_normal points FROM detector TOWARD sample,
        # so sample→detector direction is antiparallel to it.
        # cos_det = dot(sample→det_unit, -n_det)
        cos_det = -(dx_d * self.n_det[0] + dy_d * self.n_det[1] + dz_d * self.n_det[2]) / R_det
        cos_det = np.clip(cos_det, 0.0, 1.0)

        # Exact solid angle with detector tilt projection.
        # 2π(1−cosα) is the face-on solid angle; multiplying by cos_det gives
        # the effective solid angle for a tilted detector face.
        alpha = np.arctan(self.r_det_radius / R_det)
        Omega = 2.0 * np.pi * (1.0 - np.cos(alpha)) * cos_det   # steradians

        # Valid: source above surface, detector above surface, sample visible to detector face
        valid = (sinPhi > 1e-6) & (sinPsi > 1e-6) & (cos_det > 1e-6)

        return sinPhi, sinPsi, R_src, R_det, Omega, valid
