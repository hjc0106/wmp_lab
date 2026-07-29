#!/usr/bin/env python3
"""Prune old WMP checkpoints while preserving pre-semantic-fix files."""
from __future__ import annotations

import argparse
from pathlib import Path

from wmp_lab.checkpoint import prune_old_checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--keep_last_n", type=int, default=5)
    args = parser.parse_args()
    removed = prune_old_checkpoints(args.log_dir, args.keep_last_n)
    for path in removed:
        print(f"removed {path}")
    print(f"kept last {args.keep_last_n}; removed {len(removed)} file(s)")


if __name__ == "__main__":
    main()
