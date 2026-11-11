# XRF — Fundamental-Parameters X-ray fluorescence simulator

A Python pipeline that simulates the X-ray fluorescence (XRF) spectrum of a
multi-element sample using the **Fundamental Parameters (FP)** method with a
**spatially-resolved surface grid** and a **realistic detector response**.

Given an excitation spectrum (an X-ray tube or a solar-flare flux) and a sample
composition, it computes line intensities and a synthetic detector spectrum that
can be opened directly in **PyMCA**.

## What it models

- **Primary K fluorescence** (Kα1, Kα2, Kβ1, Kβ3, Kβ2, each at its own energy) via the Shiraiwa–Fujino integral
- **Secondary (inter-element) K fluorescence** — enhancement of a lighter element
  by the lines of a heavier one (toggle `include_secondary`)
- **L-shell fluorescence** (all L lines with non-zero radiative rate), with
  Coster–Kronig transfers and the K→L vacancy cascade
- **Coherent (Rayleigh) scattering** using anomalous form factors
- **Incoherent (Compton) scattering** via Klein–Nishina × incoherent scatter
  function, Doppler-broadened with the sample's shell Compton profiles
  (impulse approximation, binding-energy cutoffs)
- **Source** — an X-ray tube spectrum generated with SpekPy (`tube:` block),
  or a spectrum read from a file
- **Air** between tube, sample and detector (`atmosphere: air`), or vacuum
- **Detector response** — Be entrance-window transmission, Si efficiency, and
  Fano + electronic-noise broadening; with an optional Monte Carlo response
  matrix (`detector_response:`) for escape peaks, the low-energy shelf and
  Compton events in the Si, otherwise pure Gaussian peaks
- **Geometry** — a 2D grid over the sample surface, with per-point source/detector
  distances, incidence/emission angles, and detector solid angle

All atomic data (edge energies, fluorescence yields, jump factors, line energies,
radiative rates, cross-sections) is fetched at run time from
[**xraylib**](https://github.com/tschoonj/xraylib) — no spreadsheet look-ups.
All distances are in **cm**; cross-sections are in **cm²/g**.

## Inputs

| Input | Location | Notes |
|-------|----------|-------|
| Excitation spectrum | `tube:` block in the config, or `data/flux/*.qdp` | X-ray tube spectrum generated with [SpekPy](https://bitbucket.org/spekpy/spekpy_release), or flare spectra from a file |
| Sample compositions | `data/compositions/<sample>.yml` | one file per sample: the mass fraction of each element under `composition:` |
| Instrument config | `configs/*.yml` | `lab_xrf.yml` |

To add a sample, copy a file in `data/compositions/`, give it the sample's
name, edit the fractions and add the name to `samples` in the config. Oxide
analyses must be converted to element fractions first. Only the ratios between
the fractions matter, so the sum need not be exactly 1; a known light component
missing from the list (water, organic carbon) should be added as elements.
Other keys in the file (for example `source:` or `notes:`) are ignored.

With a `tube:` block (as in `lab_xrf.yml`) the spectrum is computed at run time
from the tube settings (voltage, target, window, current) and no flux file is
passed. The SpekPy result is saved in `.cache/tube/` (git-ignored), one file per
set of tube settings, so later runs with the same tube skip SpekPy (about 7 s);
a different current reuses the same file. Deleting the folder clears the cache. `data/flux/spectrum_xraytube.txt` is the same tube spectrum at 0.002 mA,
kept as a reference for the tests.

A flux file gives `ph/cm²/s/keV` on the beam axis at the distance
`src_reference_distance_cm` from the source. The code multiplies it by that
distance squared to get the source term in `ph/s/keV/sr`, which does not
depend on distance.

## Outputs

For each sample, results are written under `results/<config>/<run>_grid/`
(the `results/` tree is git-ignored). `run_one_grid` also returns them, per
sample: `fluorescence` (line table), `energy`, `I_coherent`, `I_incoherent`,
and the detector spectrum `spectrum_energy` / `spectrum` (the same one as in
the files below).

| File | Contents |
|------|----------|
| `*_grid_results.xlsx` | line intensities `[ph/s]` (Fluorescence sheet) + coherent/incoherent continua (Scattering sheet) |
| `*_spectrum.xlsx` | full detector spectrum, `Energy_keV` vs `Intensity_ph_s_per_keV` |
| `*_spectrum.dat` | same spectrum as PyMCA ASCII (`#L Energy Intensity`) |
| `*_spectrum.mca` | ORTEC PMCA file — **integer counts per channel** for a given live time, readable by PyMCA |
| `*_spectrum.png` | plot of the spectrum with annotated geometry |

> **Note on the `.mca` file:** the spectrum is a density in `ph/s/keV`, while an
> `.mca` stores integer counts per channel. Each channel gets the spectrum
> integrated over its energy range × live time, rounded (see
> [`output.py`](src/xrf_model/output.py)). With the default `live_time = 1 s`,
> channels below ~0.5 ph/s round to zero — increase `mca_live_time_s` in the
> config to preserve faint features.

## Project layout

```
configs/                 YAML instrument configs (paths, geometry, detector, samples)
data/
  compositions/          one YAML file per sample (element mass fractions)
  flux/                  excitation spectra (*.qdp, spectrum_xraytube.txt)
notebooks/
  xrf.ipynb              end-to-end example + plots
src/xrf_model/
  config.py              load_config, repo-root resolution
  io.py                  flux-spectrum loader
  tube.py                X-ray tube spectrum generated with SpekPy
  geometry.py            GridGeometry — surface grid, angles, solid angle
  atmosphere.py          air absorption between tube, sample and detector
  physics.py             cross-sections, K/L primary & secondary fluorescence
  compton.py             Doppler broadening of Compton scattering
  spectrum.py            build_spectrum — Gaussian detector response (Be, QE, Fano + noise)
  detector_response.py   detector response from a Monte Carlo matrix
  energy_grid.py         bin edges and count-conserving rebinning
  plot_spectrum.py       spectrum plotting
  output.py              save_mca (PyMca calibrations), ASCII/coherent exports
  pipeline.py            run_one_grid — orchestrates the whole run
tests/                   pytest suite (see Tests)
results/                 outputs (git-ignored)
```

## Setup

Requires **Python ≥ 3.11**. `xraylib` is the one non-pip-trivial dependency
(a C library with Python bindings).

Using the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with the `pyproject.toml` (Poetry / PEP 621):

```bash
pip install .
```

If `pip install xraylib` fails on your platform, install it via conda:

```bash
conda install -c conda-forge xraylib
```

## Quick start

```python
from pathlib import Path
from xrf_model.config import load_config
from xrf_model.pipeline import run_one_grid

cfg = load_config("configs/lab_xrf.yml")

# lab_xrf.yml has a `tube:` block, so no flux file is passed
results = run_one_grid(None, cfg)
# → writes .xlsx / .dat / .mca / .png under results/lab_xrf/tube_50kV_Ag_grid/

# for a spectrum from a file (config without a `tube:` block):
# results = run_one_grid(Path("data/flux/<spectrum>.qdp"), cfg)
```

From a shell (make the package importable first):

```bash
export PYTHONPATH="$(pwd)/src"
python -c "from xrf_model.config import load_config; \
from xrf_model.pipeline import run_one_grid; \
run_one_grid(None, load_config('configs/lab_xrf.yml'))"
```

## Tests

From the repo folder (pytest is in the `dev` dependency group; with pip,
`pip install pytest`), or from anywhere as `pytest <repo>/tests`:

```bash
pytest
```

The suite takes about 20 s and uses only files in the repo, not the local
response matrix or MCA calibration. Most tests compare the code with something
independent of it: xraylib's own functions, NIST tables, exact results, the
shipped tube file, or synthetic detector matrices with a known answer.

| File | Checks |
|------|--------|
| `test_rayleigh.py` | Rayleigh amplitude against xraylib `DCS_Rayl` |
| `test_geometry.py` | grid cells add up to the configured area |
| `test_tube.py` | SpekPy tube spectrum, choice of source |
| `test_atmosphere.py` | air absorption on both paths |
| `test_detector_response.py` | response-matrix folding and broadening |
| `test_efficiency.py` | interactions counted by the Si efficiency |
| `test_compton.py` | Doppler broadening of Compton scattering |
| `test_output.py` | `.mca` channels and calibration |

## Notebook

```bash
source .venv/bin/activate
pip install notebook ipykernel
python -m ipykernel install --user --name xrf-venv --display-name "Python (XRF .venv)"
jupyter notebook notebooks/xrf.ipynb
```

Select the **Python (XRF .venv)** kernel. The notebook runs `lab_xrf.yml` and
shows the strongest lines and the detector spectrum of the first sample. Its
setup cell adds `src/` to the path:

```python
from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
```

## Configuration reference

Key fields in a `configs/*.yml` (see [`configs/lab_xrf.yml`](configs/lab_xrf.yml)).
Relative paths, in the config and for a flux file passed to `run_one_grid`, are
taken from the repository folder, whatever folder Python is started in;
absolute paths are used as they are.

| Field | Meaning |
|-------|---------|
| `paths.*` | input/output locations |
| `samples` | which sample file(s) in `paths.compositions` to run |
| `include_secondary` | include inter-element secondary K fluorescence |
| `tube.*` | X-ray tube for the SpekPy spectrum: `kvp`, `target`, `transmission_thickness_um`, `filters` (mm), `energy_step_keV`, `current_mA` |
| `src_reference_distance_cm` | distance at which a flux file was evaluated (only without `tube:`); `flux_units` is optional, its only value is `ph_per_cm2_per_s_per_keV` |
| `src_x/y/z`, `det_x/y/z`, `det_normal`, `det_area` | source & detector geometry (cm) |
| `atmosphere`, `src_vacuum_path_cm`, `det_vacuum_path_cm` | gas between tube, sample and detector (`vacuum`, the default, or `air`) and the part of each path inside the housings (cm) |
| `grid_{x,y}_{min,max}`, `grid_n_{x,y}` | sample surface integration grid |
| `be_window_mm`, `si_active_layer_um` | detector window and Si efficiency |
| `detector_fano`, `detector_w_eV`, `detector_noise_eV` | peak broadening, σ² = F·w·E + σ_el² |
| `detector_response.file`, `detector_response.rows` | optional Monte Carlo response matrix (`.npz`, deposited energy before broadening; format in `lab_xrf.yml`). Without it, peaks are pure Gaussians. Pile-up and dead time are not modelled. |
| `mca_live_time_s` | acquisition time used to scale the `.mca` counts (default 1 s) |
| `mca_calibration`, `mca_channels` | optional: write the `.mca` on a real MCA's channels, from a PyMca `.calib` file (E = A + B·ch + C·ch², keV). Without them the `.mca` uses the spectrum's own grid, rebinned to equal channels if its steps are unequal |
