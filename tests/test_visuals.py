from template.visuals import build_visual
from layout import Theme


def test_visual_placeholder_when_missing() -> None:
    result = build_visual(None, width=4.0, height=3.0, theme=Theme())
    assert result.mobject is not None


def test_visual_inclined_plane() -> None:
    spec = {
        "objects": [
            {"id": "plane", "type": "plane", "angle": 30},
            {"id": "block", "type": "block", "pos": 0.5},
            {"id": "F", "type": "force", "dir": "up_slope", "label": "F"},
        ]
    }
    result = build_visual(spec, width=4.0, height=3.0, theme=Theme())
    assert result.mobject is not None
