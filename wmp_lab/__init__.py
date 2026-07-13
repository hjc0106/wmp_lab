"""IsaacLab task package for the migrated Go2 WMP project."""

import importlib

# --- configclass validate safety patch --------------------------------------
# IsaacLab's ``configclass._validate`` recurses into any object exposing
# ``__dict__``. The Go2 env registers ``RayCasterCfg``/``RayCasterCameraCfg``
# whose ``class_type`` field is the *class* ``RayCaster`` (a ``type``). After the
# first env is built, ``RayCaster`` carries a class-level ``meshes`` dict that
# holds Warp meshes whose ``device.runtime.device_map`` is self-referential.
# Recursing into the class therefore spirals into infinite recursion on the
# *second* env's ``cfg.validate()``. Validating class objects is never useful
# (a ``class_type`` field is just a constructor), so we short-circuit ``type``
# instances here. This must run before any ``DirectRLEnv.__init__`` validation.
try:
    _cc = importlib.import_module("isaaclab.utils.configclass")
    _orig_validate = _cc._validate

    def _patched_validate(obj, prefix=""):
        if isinstance(obj, type):
            return []
        return _orig_validate(obj, prefix)

    _cc._validate = _patched_validate
except Exception:  # pragma: no cover - best effort
    pass

from . import tasks  # noqa: F401,E402