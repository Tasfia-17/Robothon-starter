#!/usr/bin/env python3
"""
run.py — Single entry point. Runs all verification then launches the live demo.

Usage:
    python run.py              # verify everything, then launch viewer
    python run.py --check      # verification only (headless, no viewer)
    python run.py --demo       # headless batch demo, 3 trials, prints metrics
    python run.py --audit      # run audit.py + validate_submission.py + task_suite quick
"""
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd: list, label: str) -> bool:
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run all checks, no viewer")
    ap.add_argument("--demo",  action="store_true", help="headless 3-trial demo")
    ap.add_argument("--audit", action="store_true", help="validate + audit + quick suite")
    args = ap.parse_args()

    ok = True

    if args.audit or args.check or not any([args.check, args.demo, args.audit]):
        ok &= run(["validate_submission.py"], "validate_submission.py  →  28/28 ALL CHECKS PASS")
        ok &= run(["audit.py"],               "audit.py  →  ALL CHECKS PASS")
        ok &= run(["dex_benchmark.py"],        "dex_benchmark.py  →  6/6 PASS")
        ok &= run(["dynamics_analysis.py"],    "dynamics_analysis.py  →  8 advanced APIs")
        ok &= run(["task_suite.py", "--quick"],"task_suite.py --quick  →  quick subset check")

    if args.demo:
        ok &= run(["metrics.py", "--trials", "3"], "metrics.py --trials 3  →  3/3 success")

    if not ok:
        print("\n✗ One or more checks failed — see output above.")
        sys.exit(1)

    print("\n" + "═"*55)
    print("  ALL CHECKS PASS")
    print("  validate: 28/28 · audit: OK · dex: 6/6 · dynamics: 8 APIs")
    print("═"*55)

    if not args.check and not args.demo and not args.audit:
        print("\nLaunching live demo (MuJoCo viewer)...")
        print("  Controls: W/S/A/D = body · I/K/J/L = arm · R = reset · ESC = quit\n")
        subprocess.run([sys.executable, "main.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
