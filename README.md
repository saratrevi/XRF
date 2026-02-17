# XRF - Fundamental Parameters (FP) method

Python pipeline to compute:

* coherent scattering spectrum on the flare energy grid
* Primary K-fluorescence line intensities (Kα / Kβ)

Inputs:

* flare flux spectra (`data/flux/*.qdp`)
* element parameters table (`data/element_params/Data_keV.xlsx`)
* mass attenuation coefficients per element (`data/absorption_coeffs/*.txt`)
* coherent scattering form factors (`data/form_factors/form_factor_coherent_scatter.xlsx`)

Configs are stored in `configs/*.yml` (e.g. `deimos.yml`, `kaguya.yml`). Outputs are written under `results/` (ignored by git).

---

## Layout

```
configs/                 YAML configs (paths + geometry + element lists)
data/
  absorption_coeffs/     element .txt files (e.g. Fe.txt, Si.txt, ...)
  element_params/        Data_keV.xlsx
  form_factors/          form_factor_coherent_scatter.xlsx
  flux/                  input flux spectra (*.qdp)
notebooks/               notebooks / analysis
src/xrf_model/           package (io, physics, pipeline, output)
results/                 outputs (ignored by git)
```


## Setup
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

## Run 

```bash
export PYTHONPATH="$(pwd)/src"
python -c "from xrf_model.config import load_config; from xrf_model.pipeline import run_one; cfg=load_config('configs/deimos.yml'); run_one('data/flux/M1_1000bin.qdp', cfg)"
```

Batch:

```bash
export PYTHONPATH="$(pwd)/src"
python -c "from pathlib import Path; from xrf_model.config import load_config; from xrf_model.pipeline import run_many; cfg=load_config('configs/deimos.yml'); run_many(sorted(Path('data/flux').glob('*.qdp')), cfg)"
```

## Notebooks (Jupyter)

```bash
source .venv/bin/activate
pip install notebook ipykernel
python -m ipykernel install --user --name xrf-venv --display-name "Python (XRF .venv)"
jupyter notebook
```
Select kernel: **Python (XRF .venv)**

Notebook import helper (top cell):

```python
from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "data").exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
```

```
```
