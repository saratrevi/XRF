from pathlib import Path
import sys
from xrf_model.config import load_config
from xrf_model.pipeline import run_one, run_many
import time

ROOT = Path("../..")
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "src"))

print("CWD :", Path.cwd())
print("ROOT:", ROOT)
print("flux dir exists:", (ROOT / "data" / "flux").exists())


cfg = load_config("configs/deimos.yml")
flux_file = ROOT / "data" / "flux" / "M1_1000bin.qdp"
print("flux exists:", flux_file.exists(), flux_file)

TEST_SIZE = 200
tic = time.time()
for _ in range(TEST_SIZE):
    res_one = run_one(flux_file, cfg)
toc = time.time()

print("Time taken:", toc - tic)
print("Average time taken:", (toc - tic) / TEST_SIZE)

