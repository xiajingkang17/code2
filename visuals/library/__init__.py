from __future__ import annotations

from .builder import build_from_spec
from .registry import list_components, register

# Trigger component registrations.
from . import scene  # noqa: F401
from . import primitives  # noqa: F401
from . import primitives3d  # noqa: F401
from . import bodies  # noqa: F401
from . import physics  # noqa: F401
from . import paths  # noqa: F401
from . import tracks  # noqa: F401

__all__ = [
    "build_from_spec",
    "list_components",
    "register",
]
