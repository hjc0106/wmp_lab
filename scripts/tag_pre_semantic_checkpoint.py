#!/usr/bin/env python3
"""Tag legacy checkpoints as pre-semantic-fix (inference-only)."""
from __future__ import annotations

import argparse
from pathlib import Path

from wmp_lab.checkpoint import tag_pre_semantic_fix_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="+", help="Checkpoint path(s) to tag")
    args = parser.parse_args()
    for ckpt in args.checkpoint:
        tag_pre_semantic_fix_checkpoint(Path(ckpt).expanduser().resolve())


if __name__ == "__main__":
    main()
