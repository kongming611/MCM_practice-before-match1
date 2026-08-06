"""Small deterministic Matplotlib backend for the question-one figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


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
    """Render the primitive chart description with Matplotlib."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")
    axis.set_facecolor("white")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    for element in elements:
        kind = element["kind"]
        if kind == "text":
            axis.text(
                element["x"],
                element["y"],
                element["text"],
                fontsize=element["size"],
                color=element["color"],
                ha=element["align"],
                va="top",
                rotation=element.get("angle", 0),
            )
        elif kind == "line":
            axis.plot(
                [element["x1"], element["x2"]],
                [element["y1"], element["y2"]],
                color=element["color"],
                linewidth=element["width"],
            )
        elif kind == "rect":
            axis.add_patch(
                Rectangle(
                    (element["x"], element["y"]),
                    element["width"],
                    element["height"],
                    facecolor=element["fill"],
                    edgecolor=element.get("outline") or element["fill"],
                    linewidth=element.get("line_width", 1),
                )
            )
        elif kind == "ellipse":
            axis.add_patch(
                Ellipse(
                    (element["x"] + element["width"] / 2, element["y"] + element["height"] / 2),
                    element["width"],
                    element["height"],
                    facecolor=element["fill"],
                    edgecolor=element.get("outline") or element["fill"],
                    linewidth=element.get("line_width", 1),
                )
            )
        else:
            raise ValueError(f"unknown plotting primitive: {kind}")

    figure.savefig(path, dpi=100, facecolor="white")
    plt.close(figure)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"PNG rendering failed: {path}")
