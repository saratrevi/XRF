# XRF — Fundamental-Parameters X-ray fluorescence simulator

A Python pipeline that simulates the X-ray fluorescence (XRF) spectrum of a
multi-element sample using the **Fundamental Parameters (FP)** method with a
**spatially-resolved surface grid** and a **realistic detector response**.

Given an excitation spectrum (an X-ray tube or a solar-flare flux) and a sample
composition, it computes line intensities and a synthetic detector spectrum that
can be opened directly in **PyMCA**.

## What it models

- **Primary K fluorescence** (Kα, Kβ groups) via the Shiraiwa–Fujino integral
- **Secondary (inter-element) K fluorescence** — enhancement of a lighter element
  by the lines of a heavier one (toggle `include_secondary`)
- **L-shell fluorescence** (all L lines with non-zero radiative rate)
- **Coherent (Rayleigh) scattering** using anomalous form factors
- **Incoherent (Compton) scattering** via Klein–Nishina × incoherent scatter
  function, with the correct energy shift and Jacobian
- **Detector response** — Be entrance-window transmission, active-Si quantum
  efficiency, and Fano + electronic-noise Gaussian broadening (SDD)
- **Geometry** — a 2D grid over the sample surface, with per-point source/detector
  distances, incidence/emission angles, and detector solid angle

All atomic data (edge energies, fluorescence yields, jump factors, line energies,
radiative rates, cross-sections) is fetched at run time from
[**xraylib**](https://github.com/tschoonj/xraylib) — no spreadsheet look-ups.
All distances are in **cm**; cross-sections are in **cm²/g**.

## Inputs

| Input | Location | Notes |
|-------|----------|-------|
| Excitation flux spectrum | `data/flux/*.qdp`, `data/flux/spectrum_xraytube.txt` | flare spectra or X-ray-tube output |
| Sample compositions | `data/element_params/Data_keV.xlsx` | element rows, one column per sample; used for concentrations |
| Coherent form factors | `data/form_factors/form_factor_coherent_scatter.xlsx` | anomalous scattering factors |
| Instrument config | `configs/*.yml` | `deimos.yml`, `kaguya.yml`, `lab_xrf.yml` |

The flux column can be in `ph/cm²/s/keV` at a reference distance
(`src_reference_distance_cm`) or already a source luminosity in `ph/s/keV`,
selected via `flux_units` in the config.

## Outputs

For each sample column, results are written under `results/<config>/<run>_grid/`
(the `results/` tree is git-ignored):

| File | Contents |
|------|----------|
| `*_grid_results.xlsx` | line intensities `[ph/s]` (Fluorescence sheet) + coherent/incoherent continua (Scattering sheet) |
| `*_spectrum.xlsx` | full detector spectrum, `Energy_keV` vs `Intensity_ph_s_per_keV` |
| `*_spectrum.dat` | same spectrum as PyMCA ASCII (`#L Energy Intensity`) |
| `*_spectrum.mca` | ORTEC PMCA file — **integer counts per channel** for a given live time, readable by PyMCA |
| `*_spectrum.png` | plot of the spectrum with annotated geometry |

> **Note on the `.mca` file:** the spectrum is a density in `ph/s/keV`, while an
> `.mca` stores integer counts per channel. The conversion is
> `counts = round(spectrum × dE_keV × live_time)` (see
> [`output.py`](src/xrf_model/output.py)). With the default `live_time = 1 s`,
> channels below ~0.5 ph/s round to zero — increase `mca_live_time_s` in the
> config to preserve faint features.

## Project layout

```
configs/                 YAML instrument configs (paths, geometry, detector, samples)
data/
  absorption_coeffs/     per-element mass-attenuation .txt files
  element_params/        Data_keV.xlsx  (compositions)
  form_factors/          coherent-scatter form factors
  flux/                  excitation spectra (*.qdp, spectrum_xraytube.txt)
notebooks/
  xrf.ipynb              end-to-end example + plots
src/xrf_model/
  config.py              load_config, repo-root resolution
  io.py                  flux / attenuation / form-factor loaders
  geometry.py            GridGeometry — surface grid, angles, solid angle
  physics.py             cross-sections, K/L primary & secondary fluorescence
  spectrum.py            build_spectrum — detector response (Be, QE, Fano/ENC)
  plot_spectrum.py       spectrum plotting
  output.py              save_mca, ASCII/coherent exports
  pipeline.py            run_one_grid — orchestrates the whole run
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

cfg       = load_config("configs/lab_xrf.yml")
flux_file = Path("data/flux/spectrum_xraytube.txt")

results = run_one_grid(flux_file, cfg, debugging=True)
# → writes .xlsx / .dat / .mca / .png under results/lab_xrf/<run>_grid/
```

From a shell (make the package importable first):

```bash
export PYTHONPATH="$(pwd)/src"
python -c "from pathlib import Path; from xrf_model.config import load_config; \
from xrf_model.pipeline import run_one_grid; \
run_one_grid(Path('data/flux/spectrum_xraytube.txt'), load_config('configs/lab_xrf.yml'))"
```

## Notebook

```bash
source .venv/bin/activate
pip install notebook ipykernel
python -m ipykernel install --user --name xrf-venv --display-name "Python (XRF .venv)"
jupyter notebook notebooks/xrf.ipynb
```

Select the **Python (XRF .venv)** kernel. The first cell adds `src/` to the path:

```python
from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
```

## Configuration reference

Key fields in a `configs/*.yml` (see [`configs/lab_xrf.yml`](configs/lab_xrf.yml)):

| Field | Meaning |
|-------|---------|
| `paths.*` | input/output locations |
| `concentration_cols` | which sample column(s) of `Data_keV.xlsx` to run |
| `include_secondary` | include inter-element secondary K fluorescence |
| `flux_units`, `src_reference_distance_cm` | flux normalisation |
| `src_x/y/z`, `det_x/y/z`, `det_normal`, `det_area` | source & detector geometry (cm) |
| `grid_{x,y}_{min,max}`, `grid_n_{x,y}` | sample surface integration grid |
| `be_window_mm`, `si_active_layer_um`, `detector_fano`, `detector_enc_electrons` | detector response |
| `mca_live_time_s` | acquisition time used to scale the `.mca` counts (default 1 s) |
