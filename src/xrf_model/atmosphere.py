# src/xrf_model/atmosphere.py
"""Absorption in the gas between the tube, the sample and the detector."""
from __future__ import annotations

import numpy as np
import xraylib

AIR = "Air, Dry (near sea level)"   # NIST compound in xraylib, 1.205e-3 g/cm³


def attenuation_coefficient(atmosphere: str):
    """
    Return mu(E), the linear attenuation coefficient of the gas [1/cm].

    vacuum  no absorption, mu = 0 at every energy.
    air     dry air at sea level, NIST composition and density from xraylib.
            The total cross section is used: photons that are absorbed and
            photons that are scattered both leave the beam.

    mu accepts a scalar or an array of energies [keV].
    """
    if atmosphere == "vacuum":
        return lambda E: np.zeros(np.shape(E))
    if atmosphere == "air":
        rho = xraylib.GetCompoundDataNISTByName(AIR)["density"]
        cs_total = np.vectorize(lambda e: xraylib.CS_Total_CP(AIR, float(e)))
        return lambda E: rho * cs_total(E)
    raise ValueError(f"Unknown atmosphere: {atmosphere!r} (options: vacuum, air)")
