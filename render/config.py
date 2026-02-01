from dataclasses import dataclass
from typing import Optional


@dataclass
class RenderConfig:
    quality: str = "-ql"
    renderer: str = "cairo"
    scene: str = "ProblemScene"
    module: str = "render/scene.py"
    output: Optional[str] = None
