from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Read an image from a Windows path that may contain non-ASCII text."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image to a Windows path that may contain non-ASCII text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(path)
