#!/usr/bin/env python3
"""Graduated training validation stages from the migration plan."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES = [
    ("smoke_env", ["python", "scripts/smoke_test.py", "--num_envs", "2", "--task", "WMP-Go2-AMP-v0"]),
    ("domain_rand", ["python", "scripts/acceptance_checks.py", "--check", "domain_rand", "--num_envs", "16"]),
    ("runner_16x10", [
        "python", "scripts/train_wmp.py", "--num_envs", "16", "--max_iterations", "10",
        "--log_dir", "logs/go2_amp_semantic_v2",
    ]),
    ("reward_64x100", [
        "python", "scripts/train_wmp.py", "--num_envs", "64", "--max_iterations", "100",
        "--log_dir", "logs/go2_amp_semantic_v2",
    ]),
    ("wm_256x500", [
        "python", "scripts/train_wmp.py", "--num_envs", "256", "--max_iterations", "500",
        "--log_dir", "logs/go2_amp_semantic_v2",
    ]),
]


def main():
    parser = argparse.ArgumentParser(description="Run graduated WMP validation stages.")
    parser.add_argument("--stage", type=int, default=None, help="Run a single stage (0-based index)")
    parser.add_argument("--list", action="store_true", help="List stages and exit")
    args = parser.parse_args()

    if args.list:
        for i, (name, cmd) in enumerate(STAGES):
            print(f"{i}: {name} -> {' '.join(cmd)}")
        return

    stages = STAGES if args.stage is None else [STAGES[args.stage]]
    for name, cmd in stages:
        print(f"\n=== stage {name} ===", flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            print(f"stage {name} failed with exit code {rc}", file=sys.stderr, flush=True)
            raise SystemExit(rc)
    print("\nall requested stages passed")


if __name__ == "__main__":
    main()
