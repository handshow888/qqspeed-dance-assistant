from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .config import resolve_project_path
from .detector import Detection
from .image_io import write_image


CLASS_NAMES = (
    "up_unpressed",
    "up_pressed",
    "down_unpressed",
    "down_pressed",
    "left_unpressed",
    "left_pressed",
    "right_unpressed",
    "right_pressed",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def detection_class_id(detection: Detection) -> int:
    key = f"{detection.direction.lower()}_{detection.appearance.lower()}"
    try:
        return CLASS_IDS[key]
    except KeyError as error:
        raise ValueError(f"无法转换为 YOLO 类别：{key}") from error


def detection_to_yolo_line(
    detection: Detection,
    image_width: int,
    image_height: int,
) -> str:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图像尺寸必须大于 0")

    left = min(max(float(detection.x), 0.0), float(image_width))
    top = min(max(float(detection.y), 0.0), float(image_height))
    right = min(
        max(float(detection.x + detection.width), left),
        float(image_width),
    )
    bottom = min(
        max(float(detection.y + detection.height), top),
        float(image_height),
    )
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("检测框不在图像范围内")

    center_x = (left + right) / 2.0 / image_width
    center_y = (top + bottom) / 2.0 / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return (
        f"{detection_class_id(detection)} "
        f"{center_x:.6f} {center_y:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


class YoloDatasetCollector:
    """Save template-detector results as reviewable YOLO training samples."""

    def __init__(self, config: dict | None = None):
        values = config or {}
        self.enabled = bool(values.get("enabled", False))
        self.output_dir = resolve_project_path(
            str(values.get("output_dir", "datasets/yolo_arrows"))
        )
        self.split = str(values.get("split", "train")).strip().lower()
        if self.split not in {"train", "val"}:
            raise ValueError("dataset.split 只能是 train 或 val")
        extension = str(values.get("image_extension", ".png")).strip().lower()
        self.image_extension = extension if extension.startswith(".") else f".{extension}"
        if self.image_extension not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("dataset.image_extension 仅支持 png、jpg 或 jpeg")

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.saved_count = 0
        if self.enabled:
            self._prepare_layout()

    @property
    def status(self) -> str:
        if not self.enabled:
            return "DATASET OFF"
        return f"DATASET {self.split.upper()} {self.saved_count}"

    def disable(self) -> None:
        self.enabled = False

    def _prepare_layout(self) -> None:
        for split in ("train", "val"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        classes_path = self.output_dir / "classes.txt"
        classes_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

        names = "\n".join(
            f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)
        )
        data_yaml = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"{names}\n"
        )
        (self.output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    def save_event(
        self,
        frame: np.ndarray,
        detections: Iterable[Detection],
        *,
        game_mode: str,
        ui_mode: str,
        reasons: Iterable[str],
    ) -> tuple[Path, Path] | None:
        if not self.enabled:
            return None
        if frame.ndim != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("数据集帧必须是非空彩色图像")

        items = sorted(list(detections), key=lambda item: item.center_x)
        event_names = sorted({_safe_name(value) for value in reasons if value.strip()})
        if not event_names:
            raise ValueError("保存数据集样本时必须提供触发原因")

        height, width = frame.shape[:2]
        lines = [detection_to_yolo_line(item, width, height) for item in items]
        self.saved_count += 1
        event_slug = "+".join(event_names)
        game_directory = _safe_name(game_mode)
        ui_directory = _safe_name(ui_mode)
        stem = (
            f"{self.session_id}_{game_directory}_"
            f"{ui_directory}_{event_slug}_{self.saved_count:06d}"
        )
        image_path = (
            self.output_dir
            / "images"
            / self.split
            / game_directory
            / ui_directory
            / f"{stem}{self.image_extension}"
        )
        label_path = (
            self.output_dir
            / "labels"
            / self.split
            / game_directory
            / ui_directory
            / f"{stem}.txt"
        )
        try:
            write_image(image_path, frame)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
        except Exception:
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            self.saved_count -= 1
            raise

        return image_path, label_path
