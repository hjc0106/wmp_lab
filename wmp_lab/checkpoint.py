"""Checkpoint format and training-semantics versioning for WMP migration."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Bump CHECKPOINT_FORMAT_VERSION when the on-disk dict layout changes.
CHECKPOINT_FORMAT_VERSION = 2
# Bump TRAINING_SEMANTICS_VERSION when reward/obs/DR semantics change (invalidates resume).
TRAINING_SEMANTICS_VERSION = 1

# Checkpoints saved before semantic fixes (missing discriminator/normalizer, wrong reward scale).
PRE_SEMANTIC_FIX_TAG = "pre-semantic-fix"
PRE_SEMANTIC_FIX_MAX_SEMANTICS_VERSION = 0
AMP_NORMALIZER_FORMAT_VERSION = 1


def install_numpy_pickle_compat() -> None:
    """Allow unpickling NumPy-2 checkpoints under NumPy 1.x.

    NumPy 2 renamed ``numpy.core`` -> ``numpy._core``. Checkpoints that embed
    ndarray/normalizer state pickled with NumPy 2 fail on 1.x with
    ``ModuleNotFoundError: No module named 'numpy._core'``.
    """
    try:
        import numpy._core.multiarray  # noqa: F401

        return
    except Exception:
        pass

    import numpy.core as _core

    sys.modules["numpy._core"] = _core
    for name in (
        "multiarray",
        "umath",
        "_multiarray_umath",
        "numeric",
        "fromnumeric",
        "_dtype_ctypes",
        "_internal",
    ):
        try:
            mod = __import__(f"numpy.core.{name}", fromlist=["*"])
            sys.modules[f"numpy._core.{name}"] = mod
        except Exception:
            continue


def torch_load_checkpoint(path: str | Path, map_location=None, **kwargs):
    """``torch.load`` with NumPy 1/2 pickle path compatibility."""
    install_numpy_pickle_compat()
    kwargs.setdefault("weights_only", False)
    return torch.load(path, map_location=map_location, **kwargs)


def serialize_amp_normalizer(normalizer: Any) -> dict[str, Any] | Any:
    """Convert ``Normalizer`` into a NumPy-version-agnostic payload."""
    if normalizer is None or isinstance(normalizer, dict):
        return normalizer
    if not hasattr(normalizer, "mean") or not hasattr(normalizer, "var") or not hasattr(normalizer, "count"):
        return normalizer
    return {
        "__class__": "Normalizer",
        "format_version": AMP_NORMALIZER_FORMAT_VERSION,
        "epsilon": float(getattr(normalizer, "epsilon", 1e-4)),
        "clip_obs": float(getattr(normalizer, "clip_obs", 10.0)),
        "mean": np.asarray(normalizer.mean, dtype=np.float64).tolist(),
        "var": np.asarray(normalizer.var, dtype=np.float64).tolist(),
        "count": float(normalizer.count),
    }


def deserialize_amp_normalizer(payload: Any) -> Any:
    """Rebuild ``Normalizer`` from a serializable payload."""
    if payload is None or hasattr(payload, "normalize_torch"):
        return payload
    if not isinstance(payload, dict):
        return payload

    is_serialized = payload.get("__class__") == "Normalizer" or {"mean", "var", "count"}.issubset(payload)
    if not is_serialized:
        return payload

    from rsl_rl.utils.utils import Normalizer

    mean = np.asarray(payload["mean"], dtype=np.float64)
    normalizer = Normalizer(
        mean.shape,
        epsilon=float(payload.get("epsilon", 1e-4)),
        clip_obs=float(payload.get("clip_obs", 10.0)),
    )
    normalizer.mean = mean
    normalizer.var = np.asarray(payload["var"], dtype=np.float64)
    normalizer.count = float(payload["count"])
    return normalizer


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
    data = torch_load_checkpoint(path, map_location="cpu")
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
            data = torch_load_checkpoint(path, map_location="cpu")
            if data.get("checkpoint_tag") == PRE_SEMANTIC_FIX_TAG:
                continue
        except Exception:
            pass
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed
