from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import resolve_project_path
from .detector import ArrowDetector, Detection, select_main_row
from .yolo_dataset import ARROW_CLASS_NAMES, CLASS_NAMES


DEFAULT_MODEL_PATH = "runs/yolo_arrow/yolo26n_rhythm_10class/weights/best.pt"


@dataclass(frozen=True)
class YoloObjectDetection:
    class_name: str
    score: float
    x: int
    y: int
    width: int
    height: int

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass(frozen=True)
class YoloFrameDetections:
    arrows: tuple[Detection, ...]
    slider: YoloObjectDetection | None


def yolo_rows_to_detections(
    rows: Iterable[Sequence[float]],
    class_names: Sequence[str],
    image_width: int,
    image_height: int,
    *,
    keep_main_row: bool = True,
) -> list[Detection]:
    detections: list[Detection] = []
    for row in rows:
        if len(row) < 6:
            raise ValueError("YOLO 检测结果字段不足")
        left, top, right, bottom, score, raw_class_id = row[:6]
        class_id = int(raw_class_id)
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(f"YOLO 返回了未知类别编号：{class_id}")
        class_name = str(class_names[class_id])
        if class_name not in ARROW_CLASS_NAMES:
            continue
        try:
            direction, appearance = class_name.rsplit("_", 1)
        except ValueError as error:
            raise ValueError(f"无法解析 YOLO 类别：{class_name}") from error
        direction = direction.upper()
        if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError(f"无法解析 YOLO 方向：{class_name}")
        if appearance not in {"pressed", "unpressed"}:
            raise ValueError(f"无法解析 YOLO 按下状态：{class_name}")

        x1 = min(max(0, round(float(left))), image_width)
        y1 = min(max(0, round(float(top))), image_height)
        x2 = min(max(x1, round(float(right))), image_width)
        y2 = min(max(y1, round(float(bottom))), image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                direction=direction,
                appearance=appearance,
                score=float(score),
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
            )
        )

    if keep_main_row:
        detections = select_main_row(detections)
    return sorted(detections, key=lambda item: item.center_x)


def yolo_rows_to_frame_detections(
    rows: Iterable[Sequence[float]],
    class_names: Sequence[str],
    image_width: int,
    image_height: int,
    *,
    arrow_confidence: float = 0.5,
    slider_confidence: float = 0.3,
    keep_main_row: bool = True,
) -> YoloFrameDetections:
    materialized = [tuple(row) for row in rows]
    arrow_rows = [
        row
        for row in materialized
        if len(row) >= 6 and float(row[4]) >= arrow_confidence
    ]
    arrows = tuple(
        yolo_rows_to_detections(
            arrow_rows,
            class_names,
            image_width,
            image_height,
            keep_main_row=keep_main_row,
        )
    )

    slider: YoloObjectDetection | None = None
    for row in materialized:
        if len(row) < 6:
            continue
        left, top, right, bottom, score, raw_class_id = row[:6]
        class_id = int(raw_class_id)
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(f"YOLO 返回了未知类别编号：{class_id}")
        class_name = str(class_names[class_id])
        if class_name != "slider" or float(score) < slider_confidence:
            continue
        x1 = min(max(0, round(float(left))), image_width)
        y1 = min(max(0, round(float(top))), image_height)
        x2 = min(max(x1, round(float(right))), image_width)
        y2 = min(max(y1, round(float(bottom))), image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        candidate = YoloObjectDetection(
            class_name=class_name,
            score=float(score),
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
        )
        if slider is None or candidate.score > slider.score:
            slider = candidate
    return YoloFrameDetections(
        arrows=arrows,
        slider=slider,
    )


class YoloArrowDetector:
    """Run one YOLO pass for arrows and the moving slider."""

    def __init__(self, config: dict | None = None):
        values = config or {}
        self.model_path = resolve_project_path(
            str(values.get("model", DEFAULT_MODEL_PATH))
        ).resolve()
        if not self.model_path.exists():
            raise ValueError(f"找不到 YOLO 权重：{self.model_path}")
        self.confidence = min(max(float(values.get("confidence", 0.5)), 0.01), 1.0)
        self.slider_confidence = min(
            max(float(values.get("slider_confidence", 0.3)), 0.01), 1.0
        )
        self.iou = min(max(float(values.get("iou", 0.45)), 0.01), 1.0)
        self.image_size = max(320, int(values.get("image_size", 640)))
        self.device = str(values.get("device", "0"))
        use_half = bool(values.get("half", self.device != "cpu"))
        self.quantize = 16 if use_half else None
        self.max_detections = max(1, int(values.get("max_detections", 16)))

        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO 依赖未安装，请运行："
                r".\.venv\Scripts\python.exe -m pip install -r requirements-train.txt"
            ) from error
        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("YOLO 已配置使用显卡，但 PyTorch 当前无法使用 CUDA")

        self.model = YOLO(str(self.model_path), task="detect")
        raw_names = self.model.names
        if isinstance(raw_names, dict):
            self.class_names = tuple(
                str(raw_names[index]) for index in sorted(raw_names)
            )
        else:
            self.class_names = tuple(str(name) for name in raw_names)
        if self.class_names not in {ARROW_CLASS_NAMES, CLASS_NAMES}:
            raise ValueError(
                "YOLO 权重类别不匹配，期望8类箭头，或现有10类箭头/滑块模型"
            )
        self.supports_slider_class = self.class_names == CLASS_NAMES

    def warmup(self) -> None:
        """Move the model to its inference device before the UI becomes interactive."""
        # The automatic ROI spans the full game-window width and only its lower band.
        dummy = np.zeros((256, 1600, 3), dtype=np.uint8)
        self.detect_frame(dummy)

    def detect(self, image: np.ndarray, keep_main_row: bool = True) -> list[Detection]:
        return list(self.detect_frame(image, keep_main_row=keep_main_row).arrows)

    def detect_frame(
        self, image: np.ndarray, keep_main_row: bool = True
    ) -> YoloFrameDetections:
        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=min(self.confidence, self.slider_confidence),
            iou=self.iou,
            device=self.device,
            quantize=self.quantize,
            max_det=self.max_detections,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return YoloFrameDetections((), None)
        rows = results[0].boxes.data.detach().cpu().tolist()
        height, width = image.shape[:2]
        return yolo_rows_to_frame_detections(
            rows,
            self.class_names,
            width,
            height,
            arrow_confidence=self.confidence,
            slider_confidence=self.slider_confidence,
            keep_main_row=keep_main_row,
        )

    annotate = staticmethod(ArrowDetector.annotate)
