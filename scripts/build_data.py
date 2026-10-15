"""Build all canonical parquet files: python scripts/build_data.py [--eco 1 2 ..] [--no-ampds] [--force] [--out data/processed]"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energymind.data_ampds import build_ampds  # noqa: E402
from energymind.data_eco import build_eco  # noqa: E402


def _eco(a):
    build_eco(a[0], a[1], verbose=False)
    print(f"eco_0{a[0]} written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eco", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--no-ampds", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild existing outputs")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    todo = [h for h in a.eco if a.force or not (Path(a.out) / f"eco_0{h}.parquet").exists()]
    with Pool(3) as p:
        p.map(_eco, [(h, a.out) for h in todo])
    if not a.no_ampds and (a.force or not (Path(a.out) / "ampds2.parquet").exists()):
        build_ampds(a.out)
