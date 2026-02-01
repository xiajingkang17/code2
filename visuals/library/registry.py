from __future__ import annotations

from typing import Callable, Dict, Optional

from .types import BuildResult

ComponentBuilder = Callable[..., BuildResult]
_COMPONENTS: Dict[str, ComponentBuilder] = {}


def register(name: str) -> Callable[[ComponentBuilder], ComponentBuilder]:
    def decorator(fn: ComponentBuilder) -> ComponentBuilder:
        _COMPONENTS[name] = fn
        return fn

    return decorator


def get(name: str) -> Optional[ComponentBuilder]:
    return _COMPONENTS.get(name)


def list_components() -> list[str]:
    return sorted(_COMPONENTS.keys())
