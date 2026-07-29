"""Checkpoint format and training-semantics versioning for WMP migration."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

# Bump CHECKPOINT_FORMAT_VERSION when the on-disk dict layout changes.
CHECKPOINT_FORMAT_VERSION = 1
# Bump TRAINING_SEMANTICS_VERSION when reward/obs/DR semantics change (invalidates resume).
TRAINING_SEMANTICS_VERSION = 1

# Checkpoints saved before semantic fixes (missing discriminator/normalizer, wrong reward scale).
PRE_SEMANTIC_FIX_TAG = "pre-semantic-fix"
PRE_SEMANTIC_FIX_MAX_SEMANTICS_VERSION = 0


def checkpoint_metadata(
    *,
    seed: int | None = None,
    config: dict | None = None,
    curriculum_state: dict | None = None,
) -> dict[str, Any]:
    meta = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
    }
    if seed is not None:
        meta["seed"] = seed
    if config is not None:
        meta["config"] = config
    if curriculum_state is not None:
        meta["curriculum_state"] = curriculum_state
    return meta


def is_pre_semantic_fix_checkpoint(loaded: dict[str, Any]) -> bool:
    """True for legacy checkpoints that must not be used to resume training."""
    tag = loaded.get("checkpoint_tag")
    if tag == PRE_SEMANTIC_FIX_TAG:
        return True
    semantics = loaded.get("training_semantics_version")
    if semantics is None:
        return True
    return int(semantics) <= PRE_SEMANTIC_FIX_MAX_SEMANTICS_VERSION


def validate_checkpoint_for_resume(loaded: dict[str, Any], *, allow_inference_only: bool = False) -> None:
    """Raise ``ValueError`` if checkpoint cannot be used for training resume."""
    if allow_inference_only:
        return
    if is_pre_semantic_fix_checkpoint(loaded):
        raise ValueError(
            "Checkpoint is tagged pre-semantic-fix or lacks training_semantics_version. "
            "Use it for inference/play only; start a fresh training run instead."
        )
    fmt = loaded.get("format_version")
    if fmt is not None and int(fmt) > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version={fmt} is newer than supported {CHECKPOINT_FORMAT_VERSION}."
        )
    missing = []
    for key in ("discriminator_state_dict", "amp_normalizer"):
        if key not in loaded or loaded[key] is None:
            missing.append(key)
    if missing:
        raise ValueError(
            f"Checkpoint missing required keys for resume: {missing}. "
            "Inference-only load is still allowed via allow_inference_only=True."
        )


def tag_pre_semantic_fix_checkpoint(path: str | Path) -> None:
    """Mark an existing checkpoint as pre-semantic-fix (inference-only)."""
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    data["checkpoint_tag"] = PRE_SEMANTIC_FIX_TAG
    data["training_semantics_version"] = PRE_SEMANTIC_FIX_MAX_SEMANTICS_VERSION
    torch.save(data, path)
    warnings.warn(f"Tagged {path} as {PRE_SEMANTIC_FIX_TAG} (inference-only).")


def prune_old_checkpoints(log_dir: str | Path, keep_last_n: int) -> list[Path]:
    """Delete oldest periodic checkpoints, never touching manually tagged files."""
    log_dir = Path(log_dir)
    models = sorted(log_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if keep_last_n <= 0 or len(models) <= keep_last_n:
        return []
    to_remove = models[: len(models) - keep_last_n]
    removed = []
    for path in to_remove:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if data.get("checkpoint_tag") == PRE_SEMANTIC_FIX_TAG:
                continue
        except Exception:
            pass
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed
