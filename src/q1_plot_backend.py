"""Small deterministic PNG backend used when matplotlib is unavailable.

The project is normally compatible with matplotlib, but the execution image used
for this project does not ship it.  This module therefore renders a small set of
primitive chart elements through Windows System.Drawing.  Chinese labels are
drawn with Microsoft YaHei when available; the data and chart definitions remain
in Python and are reproducible.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from q1_common import ROOT


def _clean(value: Any) -> Any:
    """Convert numpy-like scalar values to JSON-compatible Python values."""

    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def text(elements: list[dict[str, Any]], x: float, y: float, value: str, *, size: int = 24,
         color: str = "#222222", align: str = "left", angle: float = 0) -> None:
    """Append a text primitive."""

    elements.append({"kind": "text", "x": float(x), "y": float(y), "text": str(value),
                     "size": int(size), "color": color, "align": align,
                     "angle": float(angle)})


def line(elements: list[dict[str, Any]], x1: float, y1: float, x2: float, y2: float,
         *, color: str = "#333333", width: int = 2) -> None:
    """Append a line primitive."""

    elements.append({"kind": "line", "x1": float(x1), "y1": float(y1), "x2": float(x2),
                     "y2": float(y2), "color": color, "width": int(width)})


def rect(elements: list[dict[str, Any]], x: float, y: float, width: float, height: float,
         *, fill: str = "#ffffff", outline: str | None = None, line_width: int = 1) -> None:
    """Append a rectangle primitive."""

    elements.append({"kind": "rect", "x": float(x), "y": float(y), "width": float(width),
                     "height": float(height), "fill": fill, "outline": outline,
                     "line_width": int(line_width)})


def circle(elements: list[dict[str, Any]], x: float, y: float, radius: float,
           *, fill: str = "#ffffff", outline: str | None = None, line_width: int = 1) -> None:
    """Append a circle primitive."""

    elements.append({"kind": "ellipse", "x": float(x - radius), "y": float(y - radius),
                     "width": float(2 * radius), "height": float(2 * radius), "fill": fill,
                     "outline": outline, "line_width": int(line_width)})


def save_png(path: Path, width: int, height: int, elements: Iterable[dict[str, Any]]) -> None:
    """Render primitives to a PNG and fail loudly if the renderer fails."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output": str(path),
        "width": int(width),
        "height": int(height),
        "elements": [{key: _clean(value) for key, value in element.items()}
                     for element in elements],
    }
    script = ROOT / "src" / "q1_render_primitives.ps1"
    if not script.exists():
        raise FileNotFoundError(f"Plot renderer is missing: {script}")
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False)
        spec_path = Path(handle.name)
    try:
        command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                   str(script), "-Spec", str(spec_path)]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode != 0 or not path.exists():
            raise RuntimeError(
                "PNG rendering failed: " + (completed.stderr or completed.stdout or "unknown error")
            )
    finally:
        spec_path.unlink(missing_ok=True)
