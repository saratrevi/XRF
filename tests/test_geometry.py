"""Tests for the sample-surface grid in GridGeometry.

Each point must stand for its own cell, so the cell areas add up to exactly the
configured region whatever the grid size. Points on both edges with a full cell
area each would overcount the region by (n/(n-1))^2.
"""
import numpy as np
import pytest

from xrf_model.geometry import GridGeometry


def make_cfg(n_x=2, n_y=2, x_range=(-0.4, 0.4), y_range=(-0.4, 0.4)):
    return {
        "src_x": -3.0, "src_y": 0.0, "src_z": 3.0,
        "det_x": 3.0, "det_y": 0.0, "det_z": 3.0,
        "det_normal": [-0.707, 0.0, -0.707],
        "det_area": 0.17,
        "grid_x_min": x_range[0], "grid_x_max": x_range[1], "grid_n_x": n_x,
        "grid_y_min": y_range[0], "grid_y_max": y_range[1], "grid_n_y": n_y,
    }


@pytest.mark.parametrize("n_x, n_y", [(1, 1), (2, 2), (3, 3), (10, 10), (2, 5)])
def test_cell_areas_add_up_to_the_configured_region(n_x, n_y):
    geo = GridGeometry(make_cfg(n_x, n_y))
    assert geo.XX.size * geo.dS == pytest.approx(0.8 * 0.8)


def test_points_sit_at_cell_centres():
    geo = GridGeometry(make_cfg(n_x=4, n_y=1, x_range=(0.0, 1.0), y_range=(-0.5, 0.5)))
    assert geo.XX[0] == pytest.approx(np.array([0.125, 0.375, 0.625, 0.875]))
    assert geo.YY[:, 0] == pytest.approx(np.array([0.0]))


def test_coarse_grid_agrees_with_a_fine_grid():
    """A smooth geometric signal summed on 2x2 must match the 40x40 result."""

    def signal(cfg):
        geo = GridGeometry(cfg)
        sinPhi, _, R_src, _, Omega, _ = geo.compute_local_geometry()
        return np.sum(Omega * sinPhi / R_src**2) * geo.dS

    assert signal(make_cfg(2, 2)) == pytest.approx(signal(make_cfg(40, 40)), rel=0.01)
