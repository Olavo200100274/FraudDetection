"""
Run all remaining FT-Transformer experiments sequentially.
Launch once, walk away — everything runs in order.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Always run from the src/ directory, regardless of where the script is launched
SRC_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Threshold ULB",      ["threshold_study.py", "--dataset", "ulb", "--models", "fttransformer"]),
    ("Threshold BAF",      ["threshold_study.py", "--dataset", "baf_base", "--models", "fttransformer"]),
    ("Cross-domain",       ["cross_domain.py", "--models", "fttransformer"]),
]


def main():
    print("=" * 60)
    print("  FT-Transformer — Sequential Runner")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps  : {len(STEPS)}")
    print("=" * 60)

    for i, (name, cmd) in enumerate(STEPS, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(STEPS)}] {name}")
        print(f"  Command: python {' '.join(cmd)}")
        print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        print("─" * 60)

        t0 = time.time()
        full_cmd = [sys.executable, str(SRC_DIR / cmd[0])] + cmd[1:]
        result = subprocess.run(full_cmd)

        elapsed = time.time() - t0
        minutes = elapsed / 60

        if result.returncode != 0:
            print(f"\n  FAILED: {name} (exit code {result.returncode})")
            print(f"  Elapsed: {minutes:.1f} min")
            print("  Stopping — fix the error and re-run.")
            sys.exit(1)

        print(f"\n  DONE: {name} ({minutes:.1f} min)")

    print("\n" + "=" * 60)
    print(f"  All steps complete!")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
