# src/xrf_model/output.py
from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
from datetime import datetime

from .energy_grid import is_uniform, rebin


def read_calibration(path) -> tuple[float, float, float]:
    """
    Energy calibration (A, B, C) from a PyMca .calib file:
        E(ch) = A + B·ch + C·ch²   [keV]
    The file has one section with keys order, A, B (and C for order 2).
    """
    parser = configparser.ConfigParser()
    if not parser.read(Path(path)):
        raise FileNotFoundError(f"Calibration file not found: {path}")
    sections = parser.sections()
    if len(sections) != 1:
        raise ValueError(f"{path}: expected one calibration section, found {len(sections)}")
    sec = parser[sections[0]]
    try:
        A, B = float(sec["A"]), float(sec["B"])
    except KeyError as err:
        raise ValueError(f"{path}: missing calibration coefficient {err}") from None
    C = float(sec["C"]) if int(sec.get("order", "1")) >= 2 else 0.0
    return A, B, C


def channel_energies(calibration, n_channels: int, ch: np.ndarray | None = None) -> np.ndarray:
    """E(ch) = A + B·ch + C·ch² [keV]; ch defaults to 0 … n_channels − 1."""
    A, B, C = calibration
    ch = np.arange(n_channels, dtype=float) if ch is None else ch
    return A + B * ch + C * ch ** 2


def save_mca(
    energy_kev: np.ndarray,
    spectrum: np.ndarray,
    path: str,
    description: str = "",
    live_time: float = 1.0,
    calibration: tuple[float, float, float] | None = None,
    n_channels: int | None = None,
) -> None:
    """
    Write the simulated spectrum as an ORTEC PMCA .mca file readable by PyMCA.

    The spectrum (ph/s/keV) becomes integer counts per channel:
        counts[ch] = max(0, round(rate[ch] * live_time))
    where rate[ch] is the spectrum integrated over the channel's energy range,
    so the total number of counts is kept whatever the input grid.

    Channels
    --------
    calibration = (A, B, C) and n_channels given: the channels of a real MCA,
        E(ch) = A + B·ch + C·ch² [keV] for ch = 0 … n_channels − 1, with each
        channel reaching halfway to its neighbours.  The simulated file then
        lines up channel by channel with a measurement on that MCA.
    calibration = None: the spectrum's own grid if its steps are equal (one
        channel per point, as before); otherwise equal channels as narrow as
        the smallest step, so no detail is lost.

    Energy calibration stored in the file header (eV units):
        E(ch) = A + B * ch + C * ch²

    Parameters
    ----------
    energy_kev  : 1D array of photon energies [keV] (bin centres)
    spectrum    : 1D array [ph/s/keV] on energy_kev — output of build_spectrum
    path        : full output file path (should end in .mca)
    description : free-text label written into the DESCRIPTION header field
    live_time   : simulated acquisition time [s] used to scale counts (default 1 s)
    calibration : (A, B, C) in keV, keV/ch, keV/ch², e.g. from read_calibration
    n_channels  : number of MCA channels; required with calibration
    """
    if calibration is not None:
        if not n_channels:
            raise ValueError("save_mca: n_channels is required with a calibration")
        half = np.arange(n_channels + 1, dtype=float) - 0.5
        edges = channel_energies(calibration, n_channels, half)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("save_mca: calibration energies must increase with channel")
        rate = rebin(energy_kev, spectrum, edges)
    elif is_uniform(energy_kev):
        step = float(np.mean(np.diff(energy_kev)))
        calibration = (float(energy_kev[0]), step, 0.0)
        rate = spectrum * step
    else:
        step = float(np.min(np.diff(energy_kev)))
        n = int(np.floor((energy_kev[-1] - energy_kev[0]) / step)) + 1
        calibration = (float(energy_kev[0]), step, 0.0)
        edges = energy_kev[0] + step * (np.arange(n + 1) - 0.5)
        rate = rebin(energy_kev, spectrum, edges)

    A_eV, B_eV, C_eV = (1000.0 * c for c in calibration)
    dE_eV = B_eV

    # ph/s per channel → counts per channel (integer, non-negative)
    counts = np.maximum(0, np.round(rate * live_time)).astype(int)

    n_channels = len(counts)
    timestamp  = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    with open(path, "w") as fh:
        fh.write("<<PMCA SPECTRUM>>\n")
        fh.write("TAG - \n")
        fh.write(f"DESCRIPTION - {description}\n")
        fh.write("GAIN - 1\n")
        fh.write("THRESHOLD - 0\n")
        fh.write("LIVE_MODE - 0\n")
        fh.write("PRESET_TIME - Real\n")
        fh.write(f"LIVE_TIME - {live_time:.1f}\n")
        fh.write(f"REAL_TIME - {live_time:.1f}\n")
        fh.write(f"START_TIME - {timestamp}\n")
        fh.write("SERIAL_NUMBER - \n")
        fh.write("<<CALIBRATION>>\n")
        fh.write("LABEL - Energy\\Channel\n")
        # Format: index  A[eV]  B[eV/ch]  C[eV/ch²]
        fh.write(f"1 {A_eV:.6f} {B_eV:.6f} {C_eV:.6e}\n")
        fh.write("<<DATA>>\n")
        for c in counts:
            fh.write(f"{c}\n")
        fh.write("<<END>>\n")

    print(f"→ Saved MCA file ({n_channels} channels, dE={dE_eV:.1f} eV/ch, "
          f"live_time={live_time:.1f} s) to {path}")