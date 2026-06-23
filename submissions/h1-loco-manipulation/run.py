#!/usr/bin/env python3
"""
run.py — single entry point for H1 Loco-Manipulation.

Usage:
    python run.py               # interactive sim (viewer + teleop)
    python run.py --eval        # full task suite + ablation → results/
    python run.py --audit       # 7-check honesty audit → ALL CHECKS PASS
    python run.py --demo        # regenerate demo.mp4 + SRT + GIF
    python run.py --dex         # dexterous gripper benchmark → 6/6 PASS
    python run.py --dynamics    # advanced MuJoCo API analysis → dynamics_report.json
    python run.py --quick       # fast smoke-test (3 tasks, no video)
"""
import argparse, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="H1 Loco-Manipulation")
    p.add_argument("--eval",     action="store_true", help="Full task suite + ablation")
    p.add_argument("--audit",    action="store_true", help="7-check honesty audit")
    p.add_argument("--demo",     action="store_true", help="Regenerate demo video")
    p.add_argument("--dex",      action="store_true", help="Dexterous gripper benchmark")
    p.add_argument("--dynamics", action="store_true", help="Advanced MuJoCo API analysis")
    p.add_argument("--quick",    action="store_true", help="Fast smoke-test")
    args = p.parse_args()

    if args.audit:
        import audit as _a
        sys.exit(0)

    elif args.demo:
        import record_demo as _r
        _r.main()

    elif args.dex:
        import dex_benchmark as _d
        _d.run()

    elif args.dynamics:
        import dynamics_analysis as _dy
        _dy.main() if hasattr(_dy, 'main') else exec(open('dynamics_analysis.py').read())

    elif args.eval or args.quick:
        from task_suite import run_suite
        run_suite(quick=args.quick)

    else:
        # Interactive sim
        import main as _m
        _m.main() if hasattr(_m, 'main') else exec(open('main.py').read())


if __name__ == "__main__":
    main()
