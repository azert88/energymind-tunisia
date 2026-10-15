import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_dataset.config import Config
from baseline_dataset.pipeline import run

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="EnergyMind Tunisia - clean LV neighbourhood dataset")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--resolution", type=int, default=15, help="minutes per step (1,5,10,15,30,60)")
    ap.add_argument("--workers", type=int, default=1, help="parallel (needs ~1 GB RAM each) power-flow processes")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    cfg = Config(days=a.days, resolution_min=a.resolution, seed=a.seed)
    if a.out:
        cfg.out_dir = Path(a.out)
    run(cfg, workers=a.workers, plots=not a.no_plots)
