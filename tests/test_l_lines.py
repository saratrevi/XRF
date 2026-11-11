"""Tests for the L-line cross sections, including the K→L cascade.

Below the K edge an L vacancy comes only from direct L1/L2/L3 ionisation plus
Coster-Kronig transfers, so the cross sections must equal that formula built
here from xraylib's sub-shell pieces.  Above the K edge each K vacancy adds
L vacancies: at least those from radiative Kα decay, at most two per vacancy.
"""
import numpy as np
import pytest
import xraylib as xl

from xrf_model.physics import l_line_cross_sections

Z = 40                     # Zr: L1 edge 2.53 keV, K edge 18.0 keV
LINES = {"Lα1": (xl.LA1_LINE, 2), "Lβ1": (xl.LB1_LINE, 1), "Lβ3": (xl.LB3_LINE, 0)}
SHELLS = (xl.L1_SHELL, xl.L2_SHELL, xl.L3_SHELL)


def ck(trans):
    return xl.CosKronTransProb(Z, trans)


def l_vacancies_direct(E):
    """Vacancies in L1, L2, L3 per gram from direct ionisation + Coster-Kronig."""
    s1, s2, s3 = (xl.CS_Photo_Partial(Z, sh, E) for sh in SHELLS)
    n1 = s1
    n2 = s2 + ck(xl.FL12_TRANS) * n1
    n3 = s3 + ck(xl.FL23_TRANS) * n2 + ck(xl.FL13_TRANS) * n1
    return n1, n2, n3


def line_from_vacancies(n, name):
    line, shell = LINES[name]
    return n[shell] * xl.FluorYield(Z, SHELLS[shell]) * xl.RadRate(Z, line)


def sigma_at(E, name):
    lines = {l["line_name"]: l for l in l_line_cross_sections(Z, np.array([E]))}
    return lines[name]["sigma"][0]


@pytest.mark.parametrize("E", [3.0, 5.0, 10.0, 17.0])
@pytest.mark.parametrize("name", list(LINES))
def test_below_the_k_edge_matches_the_subshell_formula(E, name):
    expected = line_from_vacancies(l_vacancies_direct(E), name)
    assert sigma_at(E, name) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("E", [20.0, 25.0, 40.0])
def test_above_the_k_edge_l_alpha_gains_the_k_cascade(E):
    direct = line_from_vacancies(l_vacancies_direct(E), "Lα1")
    extra = sigma_at(E, "Lα1") - direct
    per_l3_vacancy = xl.FluorYield(Z, xl.L3_SHELL) * xl.RadRate(Z, xl.LA1_LINE)
    sigma_k = xl.CS_Photo_Partial(Z, xl.K_SHELL, E)
    # radiative decay alone: Kα1 leaves an L3 hole, Kα2 an L2 hole that may move to L3
    radiative_l3 = xl.FluorYield(Z, xl.K_SHELL) * (
        xl.RadRate(Z, xl.KL3_LINE) + xl.RadRate(Z, xl.KL2_LINE) * ck(xl.FL23_TRANS))
    assert extra >= 0.99 * sigma_k * radiative_l3 * per_l3_vacancy
    assert extra <= 2.0 * sigma_k * per_l3_vacancy          # at most 2 L holes per K hole


def test_light_elements_have_no_l_lines_on_the_grid():
    assert l_line_cross_sections(26, np.array([5.0, 10.0])) == []
