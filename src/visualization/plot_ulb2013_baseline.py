from __future__ import annotations
import argparse
from pathlib import Path
from src.visualization.ulb2013_baseline_plots import generate_all

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None

    paths = generate_all(run_dir=run_dir, out_dir=out_dir)
    print("[OK] Figures:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()