from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import resolve_project_path
from .image_io import read_image


COLORS = {
    "UP": (80, 220, 80),
    "DOWN": (40, 180, 255),
    "LEFT": (255, 180, 40),
    "RIGHT": (220, 80, 220),
}
DIRECTION_ABBREVIATIONS = {
    "UP": "U",
    "DOWN": "D",
    "LEFT": "L",
    "RIGHT": "R",
}
LOGGER = logging.getLogger("dance_tool.runtime")


def detection_preview_label(index: int, detection: "Detection") -> str:
    direction = DIRECTION_ABBREVIATIONS.get(
        detection.direction, detection.direction[:1]
    )
    state = "P" if detection.appearance == "pressed" else "U"
    return f"{index}:{direction}/{state} {detection.score:.2f}"


@dataclass(frozen=True)
class Detection:
    direction: str
    score: float
    x: int
    y: int
    width: int
    height: int
    appearance: str = "unknown"

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass(frozen=True)
class Template:
    direction: str
    source: Path
    edges: np.ndarray


def edge_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, 55, 155)


def crop_template_to_arrow(image: np.ndarray) -> np.ndarray:
    """Remove most of the purple template background while retaining the arrow."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 145), (179, 120, 255))
    cyan = cv2.inRange(hsv, (72, 80, 120), (108, 255, 255))
    mask = cv2.bitwise_or(white, cyan)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return image

    image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
    candidates: list[tuple[float, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        distance = float(np.linalg.norm(centroids[index] - image_center))
        candidates.append((area / (1.0 + distance), index))

    if not candidates:
        return image

    _, best = max(candidates)
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    width = int(stats[best, cv2.CC_STAT_WIDTH])
    height = int(stats[best, cv2.CC_STAT_HEIGHT])
    margin = 4
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + width + margin)
    y2 = min(image.shape[0], y + height + margin)
    return image[y1:y2, x1:x2]


def intersection_over_union(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union


def non_maximum_suppression(
    detections: Iterable[Detection], iou_threshold: float
) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(intersection_over_union(candidate, existing) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def select_main_row(detections: list[Detection]) -> list[Detection]:
    """Keep the strongest horizontal row when a full screenshot contains extra icons."""
    if len(detections) <= 1:
        return detections

    rows: list[list[Detection]] = []
    for detection in sorted(detections, key=lambda item: item.center_y):
        matching_row = next(
            (
                row
                for row in rows
                if abs(np.median([item.center_y for item in row]) - detection.center_y)
                <= max(detection.height, np.median([item.height for item in row])) * 0.55
            ),
            None,
        )
        if matching_row is None:
            rows.append([detection])
        else:
            matching_row.append(detection)

    return max(rows, key=lambda row: (len(row), sum(item.score for item in row)))


def classify_appearance(image: np.ndarray, detection: Detection) -> str:
    """Blue/cyan fill means unpressed; pressed arrows lose that cyan fill."""
    patch = image[
        detection.y : detection.y + detection.height,
        detection.x : detection.x + detection.width,
    ]
    if patch.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(hsv, (72, 80, 170), (108, 255, 255))
    cyan_pixels = cv2.countNonZero(cyan)
    minimum_cyan = max(30, round(patch.shape[0] * patch.shape[1] * 0.03))
    return "unpressed" if cyan_pixels >= minimum_cyan else "pressed"


class ArrowDetector:
    def __init__(self, config: dict):
        recognition = config["recognition"]
        self.threshold = float(recognition["match_threshold"])
        self.scales = [float(value) for value in recognition["scales"]]
        self.iou_threshold = float(recognition["nms_iou_threshold"])
        self.templates = self._load_templates(config["templates"])

    @staticmethod
    def _load_templates(template_config: dict[str, list[str]]) -> list[Template]:
        loaded: list[Template] = []
        missing_directions: list[str] = []
        for direction, paths in template_config.items():
            direction_count = 0
            for configured_path in paths:
                path = resolve_project_path(configured_path)
                try:
                    image = read_image(path)
                except (OSError, ValueError) as error:
                    LOGGER.warning(
                        "arrow_template_skipped direction=%s path=%s error=%s",
                        direction,
                        path,
                        error,
                    )
                    print(f"警告：跳过无法读取的箭头模板：{path}")
                    continue
                cropped = crop_template_to_arrow(image)
                edges = edge_image(cropped)
                loaded.append(Template(direction=direction, source=path, edges=edges))
                direction_count += 1
            if direction_count == 0:
                missing_directions.append(direction)
        if not loaded:
            raise ValueError("没有配置箭头模板")
        if missing_directions:
            raise ValueError(
                "以下方向没有任何可用模板：" + "、".join(missing_directions)
            )
        return loaded

    def detect(self, image: np.ndarray, keep_main_row: bool = True) -> list[Detection]:
        search = edge_image(image)
        candidates: list[Detection] = []

        for template in self.templates:
            for scale in self.scales:
                width = max(8, round(template.edges.shape[1] * scale))
                height = max(8, round(template.edges.shape[0] * scale))
                if width >= search.shape[1] or height >= search.shape[0]:
                    continue

                resized = cv2.resize(template.edges, (width, height), interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
                local_max = result == cv2.dilate(result, np.ones((9, 9), np.uint8))
                ys, xs = np.where((result >= self.threshold) & local_max)
                for x, y in zip(xs.tolist(), ys.tolist()):
                    candidates.append(
                        Detection(
                            direction=template.direction,
                            score=float(result[y, x]),
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                        )
                    )

        detections = non_maximum_suppression(candidates, self.iou_threshold)
        if keep_main_row:
            detections = select_main_row(detections)
        detections = [
            replace(item, appearance=classify_appearance(image, item))
            for item in detections
        ]
        return sorted(detections, key=lambda item: item.center_x)

    @staticmethod
    def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        canvas = image.copy()
        for index, detection in enumerate(detections, start=1):
            color = COLORS[detection.direction]
            cv2.rectangle(
                canvas,
                (detection.x, detection.y),
                (detection.x + detection.width, detection.y + detection.height),
                color,
                2,
            )
            label = detection_preview_label(index, detection)
            cv2.putText(
                canvas,
                label,
                (detection.x, max(18, detection.y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas
